"""Run rescore on v4: cut with a cheaper Viterbi, relabel each run from its pooled evidence. LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/rescore_exp.py

Each run of the decoded path takes the class with the highest mean probability (`mean`) or the
highest summed log-probability (`loglik`); neighbours that end up equal merge. Costs for a match
are picked on the other six; Metrica uses their median. Scored against control labels only, paired
with v4 at its production costs (40, 40). No training: the out-of-fold probabilities are reused.
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

ARM = "mixed_all3"
V4 = (40.0, 40.0)
SWITCH = (2.5, 5.0, 10.0, 14.0, 20.0, 28.0, 40.0)
DEAD = (20.0, 40.0, 80.0)
MODES = ("mean", "loglik")


def rescore(lp, path, mode):
    out = path.copy()
    evidence = np.exp(lp) if mode == "mean" else lp
    for s, e, _ in zip(*improve.runs(path)):
        out[s:e] = int(np.argmax(evidence[s:e].mean(axis=0)))
    return out


def table(lp, y, period):
    rows = {"v4": train.score(y, improve.decode(lp, None, period, V4[0], 0.0, V4[1]), period)[0]}
    for s in SWITCH:
        for d in DEAD:
            path = improve.decode(lp, None, period, s, 0.0, d)
            for mode in MODES:
                rows["{}_{:g}_{:g}".format(mode, s, d)] = train.score(y, rescore(lp, path, mode), period)[0]
    return rows


def idsse_job(args):
    held, gap = args
    with np.load(os.path.join(improve.OUT, "oof", ARM, held + ".npz")) as f:
        lp = f["lp_" + gap].astype(np.float64)
    with np.load(improve.cache_path(held, "uniform")) as f:
        y, period = f["y_ctrl"], f["period"]
    return held, gap, table(lp, y, period)


def metrica_job(args):
    game, repeat, gap = args
    with np.load(os.path.join(metrica_exp.OUT, "lp_{}_{}_{}_{}.npz".format(ARM, game, repeat, gap))) as f:
        lp, y, period = f["lp"].astype(np.float64), f["y"], f["period"]
    y_ctrl = live_review.metrica_control(y, *live_review.metrica_events(game))
    return game, gap, table(lp, y_ctrl, period)


def main():
    with ProcessPoolExecutor(6) as pool:
        id_done = list(pool.map(idsse_job, [(m, g) for m in idsse.MATCHES for g in improve.GAPS]))
        me_done = list(pool.map(metrica_job, [(g, r, gap) for g in metrica_exp.GAMES
                                              for r in range(metrica_exp.REPEATS) for gap in improve.GAPS]))
    summary = {}
    for gap in improve.GAPS:
        by_match = {held: rows for held, g, rows in id_done if g == gap}
        metrica = [rows for _, g, rows in me_done if g == gap]
        for mode in MODES:
            keys = [k for k in next(iter(by_match.values())) if k.startswith(mode)]
            folds, picked = [], []
            for held, rows in by_match.items():
                others = [m for m in by_match if m != held]
                best = max(keys, key=lambda k: np.mean([by_match[m][k] for m in others]))
                folds.append(rows[best] - rows["v4"])
                picked.append(best)
            s = np.median([float(k.split("_")[1]) for k in picked])
            d = np.median([float(k.split("_")[2]) for k in picked])
            key = "{}_{:g}_{:g}".format(mode, min(SWITCH, key=lambda v: abs(v - s)), min(DEAD, key=lambda v: abs(v - d)))
            me = [r[key] - r["v4"] for r in metrica]
            summary["{}|{}".format(mode, gap)] = dict(
                v4=float(np.mean([r["v4"] for r in by_match.values()])), diff=float(np.mean(folds)),
                up=int(np.sum(np.array(folds) > 0)), folds=[round(float(v), 4) for v in folds], picked=picked,
                metrica_key=key, metrica_v4=float(np.mean([r["v4"] for r in metrica])), metrica_diff=float(np.mean(me)))
    with open(os.path.join(improve.OUT, "rescore.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    for k, v in summary.items():
        print("{:<16} IDSSE v4 {:.3f} {:+.4f} ({}/7)  Metrica v4 {:.3f} {:+.4f} [{}]  costes {}".format(
            k, v["v4"], v["diff"], v["up"], v["metrica_v4"], v["metrica_diff"], v["metrica_key"], sorted(set(v["picked"]))))


if __name__ == "__main__":
    main()
