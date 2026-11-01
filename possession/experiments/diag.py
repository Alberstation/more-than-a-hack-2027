"""Where does spellF1 go? Oracle swaps on held-out IDSSE predictions (legal: other games)."""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from possession import train  # noqa: E402

folds = [dict(np.load(p)) for p in sorted(glob.glob(os.path.join(train.DATA_DIR, "experiments", "oof_base", "*.npz")))]
GRID = [(s, d) for s in train.SWITCH_COSTS for d in train.DEAD_COSTS]


def mean_score(fn):
    return np.mean([train.score(f["y"], fn(f), f["period"]) for f in folds], axis=0)


def spells(codes, period):
    out = []
    for p in np.unique(period):
        c = codes[period == p]
        cut = np.r_[0, np.flatnonzero(np.diff(c)) + 1, len(c)]
        out += [(int(c[a]), b - a) for a, b in zip(cut[:-1], cut[1:])]
    return out


sym = lambda f: (f["lp"] + f["lp_swap"]) / 2
print("costs grid, spellF1 (rows switch, cols dead)")
tab = {}
for name, lp_fn in (("plain", lambda f: f["lp"]), ("sym", sym)):
    for f in folds:
        f["_" + name] = {g: train.decode_periods(lp_fn(f), f["period"], *g) for g in GRID}
    tab[name] = {g: mean_score(lambda f: f["_" + name][g]) for g in GRID}
    print(name)
    for s in train.SWITCH_COSTS:
        print("  {:>4g} ".format(s) + " ".join("{:.4f}".format(tab[name][(s, d)][0])
                                                for d in train.DEAD_COSTS))
    best = max(GRID, key=lambda g: tab[name][g][0])
    print("  best", best, tab[name][best])

name = "sym" if max(v[0] for v in tab["sym"].values()) >= max(
    v[0] for v in tab["plain"].values()) else "plain"
best = max(GRID, key=lambda g: tab[name][g][0])
pred = lambda f: f["_" + name][best]
print("\nusing", name, best)


def hybrid_dead(f):
    """True dead ball, predicted team."""
    p = pred(f).copy()
    y = f["y"]
    p[y == 2] = 2
    alive = (y != 2)
    wrong = alive & (p == 2)
    if wrong.any():
        idx = np.flatnonzero(~wrong)
        p[wrong] = p[idx[np.searchsorted(idx, np.flatnonzero(wrong)).clip(0, len(idx) - 1)]]
        p[(y != 2) & (p == 2)] = 0
    return p


def hybrid_team(f):
    """Predicted dead ball, true team where the frame is truly alive, else carried."""
    p = pred(f).copy()
    y = f["y"]
    keep = (p != 2) & (y != 2)
    p[keep] = y[keep]
    return p


print("prediction          ", mean_score(pred))
print("oracle D, our team  ", mean_score(hybrid_dead))
print("our D, oracle team  ", mean_score(hybrid_team))

print("\nperiod-level counts (mean per match)")
gt = [spells(f["y"], f["period"]) for f in folds]
pr = [spells(pred(f), f["period"]) for f in folds]
for label, seq in (("truth", gt), ("pred", pr)):
    poss = [[n for c, n in s if c < 2] for s in seq]
    dead = [[n for c, n in s if c == 2] for s in seq]
    print("  {:<6} poss n={:.0f} median {:.1f}s mean {:.1f}s p10 {:.1f}s | dead n={:.0f} median {:.1f}s".format(
        label, np.mean([len(p) for p in poss]), np.median(np.concatenate(poss)) / 25,
        np.mean(np.concatenate(poss)) / 25, np.percentile(np.concatenate(poss), 10) / 25,
        np.mean([len(d) for d in dead]), np.median(np.concatenate(dead)) / 25))
print("  D share truth {:.3f} pred {:.3f}".format(
    np.mean(np.concatenate([f["y"] for f in folds]) == 2),
    np.mean(np.concatenate([pred(f) for f in folds]) == 2)))
shortest = np.concatenate([[n for c, n in s if c < 2] for s in gt])
print("  truth poss share <2s {:.3f}, <5s {:.3f}".format(np.mean(shortest < 50), np.mean(shortest < 125)))
