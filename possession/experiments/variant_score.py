"""Encoder variants against the base encoder of the same seed, both mixed with v5's HGB. LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/variant_score.py encoder_w1_s2027 ...

Mixture weight, Viterbi costs and, when the variant has a change logit, its gain in the switch
cost (-switch + gain * logit, never above 0) are picked for a match on the other six; Metrica
uses their medians. Control labels only. Also prints recall by true spell length at the costs
picked, to see whether the short possessions are the ones that move.
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

HGB, BASE = "ctrl_mixed_all3", "encoder"
WEIGHTS = (0.5, 0.75, 1.0)
SWITCH = (14.0, 20.0, 28.0, 40.0)
DEAD = (20.0, 40.0)
GAINS = (0.0, 2.0, 4.0)


def paths(lp_hgb, lp_enc, logit, period):
    """{key: decoded path} over the grid."""
    out = {}
    for w in WEIGHTS:
        lp = np.log(np.clip((1 - w) * np.exp(lp_hgb) + w * np.exp(lp_enc), 1e-4, 1.0))
        for g in (GAINS if logit is not None else (0.0,)):
            for s in SWITCH:
                for d in DEAD:
                    out["{:g}_{:g}_{:g}_{:g}".format(w, s, d, g)] = improve.decode(lp, logit, period, s, g, d)
    return out


def load_oof(arm, m, gap):
    with np.load(os.path.join(improve.OUT, "oof", arm, m + ".npz")) as f:
        logit = f["logit_" + gap].astype(np.float64) if "logit_" + gap in f.files else None
        return f["lp_" + gap].astype(np.float64), logit


def idsse_job(args):
    arm, m, gap = args
    with np.load(improve.cache_path(m, "uniform")) as f:
        y, period = f["y_ctrl"], f["period"]
    lp_hgb = load_oof(HGB, m, gap)[0]
    lp_enc, logit = load_oof(arm, m, gap)
    found = paths(lp_hgb, lp_enc, logit, period)
    return arm, m, gap, {k: train.score(y, p, period)[0] for k, p in found.items()}, \
        {k: short_review.recall(y, p, np.ones((len(y), 3))) for k, p in found.items()}


def metrica_job(args):
    arm, game, repeat, gap = args
    def lp_of(a):
        with np.load(os.path.join(metrica_exp.OUT, "lp_{}_{}_{}_{}.npz".format(a, game, repeat, gap))) as f:
            return f["lp"].astype(np.float64), (f["logit"].astype(np.float64) if "logit" in f.files else None), f["y"], f["period"]
    lp_hgb, _, y, period = lp_of(HGB)
    lp_enc, logit, _, _ = lp_of(arm)
    y = live_review.metrica_control(y, *live_review.metrica_events(game))
    return arm, gap, {k: train.score(y, p, period)[0] for k, p in paths(lp_hgb, lp_enc, logit, period).items()}


def nested(by):
    folds, picked = [], []
    for held in idsse.MATCHES:
        keys = list(by[held])
        best = max(keys, key=lambda k: np.mean([by[m][k] for m in idsse.MATCHES if m != held]))
        folds.append(by[held][best])
        picked.append(best)
    parts = np.array([[float(x) for x in k.split("_")] for k in picked])
    grids = (WEIGHTS, SWITCH, DEAD, GAINS)
    key = "_".join("{:g}".format(min(g, key=lambda v: abs(v - x))) for g, x in zip(grids, np.median(parts, axis=0)))
    return np.array(folds), picked, key


def main():
    arms = (BASE,) + tuple(sys.argv[1:])
    with ProcessPoolExecutor(6) as pool:
        id_done = list(pool.map(idsse_job, [(a, m, g) for a in arms for m in idsse.MATCHES for g in improve.GAPS]))
        me_done = list(pool.map(metrica_job, [(a, game, r, g) for a in arms for game in metrica_exp.GAMES
                                              for r in range(metrica_exp.REPEATS) for g in improve.GAPS]))
    summary = {}
    for gap in improve.GAPS:
        res = {}
        for arm in arms:
            by = {m: t for a, m, g, t, _ in id_done if a == arm and g == gap}
            rec = {m: r for a, m, g, _, r in id_done if a == arm and g == gap}
            folds, picked, key = nested(by)
            me = [t for a, g, t in me_done if a == arm and g == gap]
            rows = [row for m, k in zip(idsse.MATCHES, picked) for row in rec[m][k]]
            rows = np.array(rows, dtype=float)
            res[arm] = dict(folds=folds, picked=picked, metrica=float(np.mean([t[key] for t in me])), key=key,
                            recall=[float(rows[rows[:, 0] == b, 1].mean()) for b in range(5)])
        print("\nhuecos {}  (recall por duracion: {})".format(gap, ", ".join(short_review.NAMES)))
        base = res[BASE]
        for arm in arms:
            r = res[arm]
            diff = r["folds"] - base["folds"]
            summary["{}|{}".format(arm, gap)] = dict(score=float(r["folds"].mean()), diff=float(diff.mean()),
                                                     up=int(np.sum(diff > 0)), metrica=r["metrica"],
                                                     metrica_diff=r["metrica"] - base["metrica"], key=r["key"],
                                                     picked=r["picked"], recall=r["recall"])
            print("  {:<18} IDSSE {:.3f} {:+.4f} ({}/7)  Metrica {:.3f} {:+.4f}  [{}]  recall {}".format(
                arm, r["folds"].mean(), diff.mean(), int(np.sum(diff > 0)), r["metrica"],
                r["metrica"] - base["metrica"], r["key"], " ".join("{:.0%}".format(v) for v in r["recall"])))
    with open(os.path.join(improve.OUT, "variants.json"), "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
