"""HGB + encoder pairs on the blurred-ball bench, against the first pair. LIVE_DEAD=1 BALL_BLUR=1.

    LIVE_DEAD=1 BALL_BLUR=1 python possession/experiments/pair_score.py \
        ctrl_sharp3+encoder_sharp_s2027 ctrl_mixed_all3+encoder ctrl_kick3+encoder_kick_s2027

As variant_score.py (mixture weight and Viterbi costs nested over the other six matches, Metrica at
their medians, control labels only, recall by true spell length), but both halves of the pair may
change. The first pair is the reference: today's recipe, trained on the sharp ball, facing the
blurred one. Writes pairs.json.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import idsse, train  # noqa: E402
import improve  # noqa: E402
import live_review  # noqa: E402
import metrica_exp  # noqa: E402
import short_review  # noqa: E402
import variant_score as vs  # noqa: E402

vs.GAINS = (0.0,)


def idsse_job(args):
    pair, m, gap = args
    hgb, enc = pair.split("+")
    with np.load(improve.cache_path(m, "uniform")) as f:
        y, period = f["y_ctrl"], f["period"]
    found = vs.paths(vs.load_oof(hgb, m, gap)[0], vs.load_oof(enc, m, gap)[0], None, period)
    return pair, m, gap, {k: train.score(y, p, period)[0] for k, p in found.items()}, \
        {k: short_review.recall(y, p, np.ones((len(y), 3))) for k, p in found.items()}


def metrica_job(args):
    pair, game, repeat, gap = args
    hgb, enc = pair.split("+")
    def lp_of(a):
        with np.load(os.path.join(metrica_exp.OUT, "lp_{}_{}_{}_{}.npz".format(a, game, repeat, gap))) as f:
            return f["lp"].astype(np.float64), f["y"], f["period"]
    lp_hgb, y, period = lp_of(hgb)
    lp_enc = lp_of(enc)[0]
    y = live_review.metrica_control(y, *live_review.metrica_events(game))
    return pair, gap, {k: train.score(y, p, period)[0] for k, p in vs.paths(lp_hgb, lp_enc, None, period).items()}


def main():
    pairs = tuple(sys.argv[1:])
    for hgb in dict.fromkeys(p.split("+")[0] for p in pairs):
        live_review.metrica_probabilities(hgb, improve.ARMS[hgb]["blocks"])
    with ProcessPoolExecutor(6) as pool:
        id_done = list(pool.map(idsse_job, [(p, m, g) for p in pairs for m in idsse.MATCHES for g in improve.GAPS]))
        me_done = list(pool.map(metrica_job, [(p, game, r, g) for p in pairs for game in metrica_exp.GAMES
                                              for r in range(metrica_exp.REPEATS) for g in improve.GAPS]))
    summary = {}
    for gap in improve.GAPS:
        res = {}
        for pair in pairs:
            by = {m: t for a, m, g, t, _ in id_done if a == pair and g == gap}
            rec = {m: r for a, m, g, _, r in id_done if a == pair and g == gap}
            folds, picked, key = vs.nested(by)
            me = [t for a, g, t in me_done if a == pair and g == gap]
            rows = np.array([row for m, k in zip(idsse.MATCHES, picked) for row in rec[m][k]], dtype=float)
            res[pair] = dict(folds=folds, picked=picked, metrica=float(np.mean([t[key] for t in me])), key=key,
                             recall=[float(rows[rows[:, 0] == b, 1].mean()) for b in range(5)])
        print("\nhuecos {}  (recall por duracion: {})".format(gap, ", ".join(short_review.NAMES)))
        base = res[pairs[0]]
        for pair in pairs:
            r = res[pair]
            diff = r["folds"] - base["folds"]
            summary["{}|{}".format(pair, gap)] = dict(score=float(r["folds"].mean()), diff=float(diff.mean()),
                                                      up=int(np.sum(diff > 0)), metrica=r["metrica"],
                                                      metrica_diff=r["metrica"] - base["metrica"], key=r["key"],
                                                      picked=r["picked"], recall=r["recall"])
            print("  {:<34} IDSSE {:.3f} {:+.4f} ({}/7)  Metrica {:.3f} {:+.4f}  [{}]  recall {}".format(
                pair, r["folds"].mean(), diff.mean(), int(np.sum(diff > 0)), r["metrica"],
                r["metrica"] - base["metrica"], r["key"], " ".join("{:.0%}".format(v) for v in r["recall"])))
    with open(os.path.join(improve.OUT, "pairs.json"), "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
