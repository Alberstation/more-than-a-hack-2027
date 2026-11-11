"""Score screen.py arms against `ref`, v6 constants fixed (0.75, 14, 20), control labels. LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/screen_score.py <arm> [<arm> ...]

Per fold: IDSSE held match (both gap patterns), Metrica held game (3 repeats x 2 patterns), mixed
with v5's HGB out of fold (IDSSE) or fitted on all IDSSE (Metrica). Rule (plan 3j): out if the mean
paired difference is <= 0 or the arm wins <= 2 of 4 folds; up with >= +0.01 and >= 3 of 4.
The `rand` column is the S6 bench (Metrica held games under random degradation), reported apart.
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
from possession import train  # noqa: E402
import improve  # noqa: E402
import metrica_mix  # noqa: E402

HGB = "ctrl_mixed_all3"
FOLDS = (("idsse", "J03WMX"), ("idsse", "J03WN1"), ("metrica", 1), ("metrica", 2))
if os.environ.get("SCREEN_FOLDS") == "all":   # confirmation: every held match / game
    FOLDS = tuple(("idsse", m) for m in ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"))         + tuple(("metrica", g) for g in (1, 2, 3))
W, SWITCH, DEAD = 0.75, 14.0, 20.0
SCREEN = os.path.join(improve.OUT, "screen")


def spell(lp_h, lp_e, y, period):
    lp = np.log(np.clip((1 - W) * np.exp(lp_h) + W * np.exp(lp_e), 1e-4, 1.0))
    return train.score(y, improve.decode(lp, None, period, SWITCH, 0.0, DEAD), period)[0]


def job(args):
    arm, kind, held = args
    if kind == "idsse":
        with np.load(improve.cache_path(held, "uniform")) as f:
            y, period = f["y_ctrl"], f["period"]
        oof = np.load(os.path.join(improve.OUT, "oof", HGB, held + ".npz"))
        enc = np.load(os.path.join(SCREEN, arm, "idsse_{}.npz".format(held)))
        return arm, kind, held, {"main": float(np.mean([spell(oof["lp_" + g].astype(np.float64), enc["lp_" + g].astype(np.float64), y, period)
                                                         for g in improve.GAPS]))}
    sets = os.path.join(improve.OUT, "sets")
    out = {}
    for gaps, key in ((improve.GAPS, "main"), (("rand",), "rand")):
        vals = []
        for r in range(3):
            for g in gaps:
                with np.load(metrica_mix.lp_path(HGB, held, r, g)) as f:
                    lp_h, period = f["lp"].astype(np.float64), f["period"]
                with np.load(os.path.join(sets, "metrica_g{}_{}_{}.npz".format(held, r, g))) as f:
                    y = f["y_ctrl"]
                lp_e = np.load(os.path.join(SCREEN, arm, "metrica_{}_{}_{}.npz".format(held, r, g)))["lp"].astype(np.float64)
                vals.append(spell(lp_h, lp_e, y, period))
        out[key] = float(np.mean(vals))
    return arm, kind, held, out


def main(arms):
    arms = ["ref"] + [a for a in arms if a != "ref"]
    jobs = [(a, k, h) for a in arms for k, h in FOLDS]
    with ProcessPoolExecutor(8) as pool:
        done = {(a, h): v for a, k, h, v in pool.map(job, jobs)}
    ref = np.array([done[("ref", h)]["main"] for _, h in FOLDS])
    ref_rand = np.mean([done[("ref", h)]["rand"] for k, h in FOLDS if k == "metrica"])
    print("ref   " + " ".join("{:.3f}".format(v) for v in ref) + "   media {:.3f}   rand {:.3f}".format(ref.mean(), ref_rand))
    summary = {}
    for a in arms[1:]:
        v = np.array([done[(a, h)]["main"] for _, h in FOLDS])
        d = v - ref
        rand = np.mean([done[(a, h)]["rand"] for k, h in FOLDS if k == "metrica"]) - ref_rand
        up = int(np.sum(d > 0))
        n = len(FOLDS)
        verdict = "fuera" if d.mean() <= 0 or up <= n // 2 else ("sube" if d.mean() >= 0.01 and up > n // 2 else "dudosa")
        kinds = np.array([k for k, _ in FOLDS])
        by = "   idsse {:+.4f}  metrica {:+.4f}".format(d[kinds == "idsse"].mean(), d[kinds == "metrica"].mean())
        summary[a] = dict(folds=d.tolist(), mean=float(d.mean()), up=up, rand=float(rand), verdict=verdict)
        print("{:<9} ".format(a) + " ".join("{:+.3f}".format(x) for x in d)
              + "   media {:+.4f} ({}/{})   rand {:+.4f}   {}".format(d.mean(), up, n, rand, verdict) + by)
    with open(os.path.join(SCREEN, "summary{}.json".format("_all" if len(FOLDS) > 4 else "")), "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
