"""Does a second provider in training help a match nobody trained on? LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/metrica_mix.py train     # HGB folds, ~1 h
    LIVE_DEAD=1 python possession/experiments/metrica_mix.py average <out> <arm> ...   # seeds, in probability
    LIVE_DEAD=1 python possession/experiments/metrica_mix.py score <hgb+enc[:hgbB+encB]> ...

Protocol, fixed before any score was seen. Same degradation, control labels and HGB recipe as
ctrl_mixed_all3 (v5's HGB); Metrica games 1-3 enter as three more matches (3 draws x 2 gap
patterns each, both views), labels from their events through the same control rule.
  A  IDSSE held out (main criterion): 6 IDSSE (+ 3 Metrica) -> the 7th. Arm ctrl_metrica3.
  B  Metrica held out (secondary, same provider as two training games): 7 IDSSE (+ the other two
     Metrica) -> the third. Reference: the HGB fitted on all IDSSE.
  C  Transfer: Metrica only -> every IDSSE match, against IDSSE only -> Metrica in B.
Pairs mix an HGB with an encoder as v5 does; weight and costs nested on A, Metrica at the medians.
"""
import json
import os
import pickle
import sys
import time
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

BASE = "ctrl_mixed_all3"
MIX = "ctrl_metrica3"
METONLY = "ctrl_metonly3"
BLOCKS = improve.ARMS[BASE]["blocks"]
KEY = "y_ctrl"
vs.GAINS = (0.0,)


def metrica_match(game, repeat, gap):
    with np.load(os.path.join(metrica_exp.OUT, "g{}_{}_{}.npz".format(game, repeat, gap))) as f:
        return dict(X=np.hstack([f[b] for b in BLOCKS]), X_swap=np.hstack([f[b + "_swap"] for b in BLOCKS]),
                    y_ctrl=f["y_ctrl"], y=f["y"], period=f["period"])


def metrica_pool(games):
    return [metrica_match(g, r, gap) for g in games for r in range(metrica_exp.REPEATS) for gap in improve.GAPS]


def idsse_pool(matches):
    return [improve.load_match(m, g, BLOCKS, d) for d in range(3) for g in improve.GAPS for m in matches]


def fit(pool):
    return improve.Fitted(dict(improve.ARMS[BASE], _extra={}), pool, KEY, train.SEED)


def lp_path(arm, game, repeat, gap):
    return os.path.join(metrica_exp.OUT, "lp_{}_{}_{}_{}.npz".format(arm, game, repeat, gap))


def write_metrica(arm, model, games):
    for g in games:
        for r in range(metrica_exp.REPEATS):
            for gap in improve.GAPS:
                m = metrica_match(g, r, gap)
                np.savez(lp_path(arm, g, r, gap), lp=improve.log_p(model.proba(m["X"])).astype(np.float64),
                         y=m["y"], period=m["period"])


def write_idsse(arm, model, held):
    folder = os.path.join(improve.OUT, "oof", arm)
    os.makedirs(folder, exist_ok=True)
    out = {"lp_" + g: improve.log_p(model.proba(improve.load_match(held, g, BLOCKS)["X"])) for g in improve.GAPS}
    np.savez(os.path.join(folder, held + ".npz"), **out)


def train_all():
    t = time.time()
    met = metrica_pool(metrica_exp.ALL_GAMES)
    # A: IDSSE held out, with all three Metrica games in the pool.
    for held in idsse.MATCHES:
        if os.path.isfile(os.path.join(improve.OUT, "oof", MIX, held + ".npz")):
            continue
        model = fit(idsse_pool([m for m in idsse.MATCHES if m != held]) + met)
        write_idsse(MIX, model, held)
        print("A", held, round(time.time() - t), flush=True)
    # B reference: today's HGB fitted on all IDSSE, on every Metrica game.
    with open(os.path.join(metrica_exp.OUT, BASE + ".pkl"), "rb") as handle:
        base = pickle.load(handle)[0]
    write_metrica(BASE, base, metrica_exp.ALL_GAMES)
    del base
    # B: each Metrica game held out from a pool of all IDSSE and the other two.
    every = idsse_pool(idsse.MATCHES)
    for held in metrica_exp.ALL_GAMES:
        if os.path.isfile(lp_path(MIX, held, 0, "uniform")):
            continue
        others = [g for g in metrica_exp.ALL_GAMES if g != held]
        model = fit(every + [m for g in others for m in metrica_pool([g])])
        write_metrica(MIX, model, [held])
        print("B", held, round(time.time() - t), flush=True)
    del every
    # C: Metrica only, tested on every IDSSE match.
    if not os.path.isfile(os.path.join(improve.OUT, "oof", METONLY, idsse.MATCHES[-1] + ".npz")):
        model = fit(met)
        for held in idsse.MATCHES:
            write_idsse(METONLY, model, held)
        print("C", round(time.time() - t), flush=True)


def idsse_job(args):
    pair, m, gap = args
    hgb, enc = pair.split("+")
    with np.load(improve.cache_path(m, "uniform")) as f:
        y, period = f["y_ctrl"], f["period"]
    lp_h = vs.load_oof(hgb, m, gap)[0]
    lp_e = vs.load_oof(enc, m, gap)[0] if enc != "none" else lp_h
    found = vs.paths(lp_h, lp_e, None, period)
    return pair, m, gap, {k: train.score(y, p, period)[0] for k, p in found.items()}, \
        {k: short_review.recall(y, p, np.ones((len(y), 3))) for k, p in found.items()}


def metrica_job(args):
    pair, game, repeat, gap = args
    hgb, enc = pair.split("+")
    def load(a):
        with np.load(lp_path(a, game, repeat, gap)) as f:
            return f["lp"].astype(np.float64), f["y"], f["period"]
    lp_h, y, period = load(hgb)
    lp_e = load(enc)[0] if enc != "none" else lp_h
    y = live_review.metrica_control(y, *live_review.metrica_events(game))
    return pair, game, gap, {k: train.score(y, p, period)[0] for k, p in vs.paths(lp_h, lp_e, None, period).items()}


def score(pairs):
    """pairs: 'hgb+encoder' for A, with the B arms given after a colon: 'hgbA+encA:hgbB+encB'."""
    a_pairs = [p.split(":")[0] for p in pairs]
    b_pairs = [p.split(":")[-1] for p in pairs]
    with ProcessPoolExecutor(6) as pool:
        id_done = list(pool.map(idsse_job, [(p, m, g) for p in dict.fromkeys(a_pairs) for m in idsse.MATCHES
                                            for g in improve.GAPS]))
        me_done = list(pool.map(metrica_job, [(p, game, r, g) for p in dict.fromkeys(b_pairs)
                                              for game in metrica_exp.ALL_GAMES
                                              for r in range(metrica_exp.REPEATS) for g in improve.GAPS]))
    summary = {}
    for gap in improve.GAPS:
        print("\nhuecos {}   IDSSE fuera (A) | Metrica fuera (B), por partido 1/2/3 | recall < 2 s, 2-4 s".format(gap))
        ref = None
        for a, b in zip(a_pairs, b_pairs):
            by = {m: t for p, m, g, t, _ in id_done if p == a and g == gap}
            rec = {m: r for p, m, g, _, r in id_done if p == a and g == gap}
            folds, picked, key = vs.nested(by)
            per_game = [float(np.mean([t[key] for p, game, g, t in me_done if p == b and g == gap and game == x]))
                        for x in metrica_exp.ALL_GAMES]
            rows = np.array([row for m, k in zip(idsse.MATCHES, picked) for row in rec[m][k]], dtype=float)
            recall = [float(rows[rows[:, 0] == i, 1].mean()) for i in (0, 1)]
            if ref is None:
                ref = (folds, np.array(per_game))
            d, dm = folds - ref[0], np.array(per_game) - ref[1]
            summary["{}|{}|{}".format(a, b, gap)] = dict(idsse=float(folds.mean()), diff=float(d.mean()),
                                                         up=int(np.sum(d > 0)), metrica=per_game,
                                                         metrica_diff=float(dm.mean()), key=key, recall=recall)
            print("  {:<36} {:.3f} {:+.4f} ({}/7) | {:.3f} {:+.4f}  [{}]  [{}]  {:.0%} {:.0%}".format(
                a if a == b else a + " | " + b.split("+")[0], folds.mean(), d.mean(), int(np.sum(d > 0)),
                np.mean(per_game), dm.mean(), " ".join("{:.3f}".format(v) for v in per_game), key, *recall))
    with open(os.path.join(improve.OUT, "metrica_mix.json"), "w") as handle:
        json.dump(summary, handle, indent=1)


def average(out, arms):
    """Seeds averaged in probability, as v5 averages its encoders, on IDSSE and on Metrica."""
    mean = lambda lps: np.log(np.clip(np.mean([np.exp(x) for x in lps], axis=0), 1e-4, 1.0)).astype(np.float32)
    folder = os.path.join(improve.OUT, "oof", out)
    os.makedirs(folder, exist_ok=True)
    for m in idsse.MATCHES:
        np.savez(os.path.join(folder, m + ".npz"),
                 **{"lp_" + g: mean([vs.load_oof(a, m, g)[0] for a in arms]) for g in improve.GAPS})
    for game in metrica_exp.ALL_GAMES:
        for r in range(metrica_exp.REPEATS):
            for g in improve.GAPS:
                parts = [np.load(lp_path(a, game, r, g)) for a in arms]
                np.savez(lp_path(out, game, r, g), lp=mean([p["lp"] for p in parts]).astype(np.float64),
                         y=parts[0]["y"], period=parts[0]["period"])


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train_all()
    elif sys.argv[1] == "average":
        average(sys.argv[2], sys.argv[3:])
    else:
        score(sys.argv[2:])
