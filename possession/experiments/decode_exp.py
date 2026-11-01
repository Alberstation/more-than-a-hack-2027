"""Viterbi with a per-frame H<->A cost: -switch + gain * logit(p_change), capped at 0."""
import glob
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from possession import train  # noqa: E402


def viterbi(em, ha, dead):
    n = len(em)
    score = em[0].copy()
    back = np.empty((n, 3), dtype=np.int8)
    back[0] = 0, 1, 2
    t = np.array([[0, 0, -dead], [0, 0, -dead], [-dead, -dead, 0.0]])
    for i in range(1, n):
        t[0, 1] = t[1, 0] = ha[i]
        c = score[:, None] + t
        back[i] = np.argmax(c, axis=0)
        score = c[back[i], [0, 1, 2]] + em[i]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def job(args):
    path, s, g, d = args
    f = np.load(path)
    lp, logit, y, period = f["lp"], f["logit"], f["y"], f["period"]
    ha = np.minimum(-s + g * logit, 0.0)
    out = np.empty(len(y), dtype=np.int8)
    for p in np.unique(period):
        r = np.flatnonzero(period == p)
        out[r] = viterbi(lp[r], ha[r], d)
    n = sum(1 for c in np.split(out, np.flatnonzero(np.diff(out)) + 1) if c[0] < 2)
    return (s, g, d), train.score(y, out, period), n


if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(train.DATA_DIR, "experiments", sys.argv[1], "*.npz")))
    grid = list(itertools.product((2.5, 5.0, 10.0), (0.0, 2.0, 4.0, 8.0), (40.0, 80.0)))
    with ProcessPoolExecutor(12) as pool:
        res = list(pool.map(job, [(p, *g) for g in grid for p in paths]))
    table = {}
    for g, sc, n in res:
        table.setdefault(g, []).append((*sc, n))
    mean = {g: np.mean(v, axis=0) for g, v in table.items()}
    print(sys.argv[1], "top (switch, gain, dead): spell macro n_poss")
    for g in sorted(mean, key=lambda g: -mean[g][0])[:6]:
        print(" ", g, np.round(mean[g], 4), [round(float(v[0]), 3) for v in table[g]])
    g0 = max((g for g in mean if g[1] == 0), key=lambda g: mean[g][0])
    print("  best without detector", g0, np.round(mean[g0], 4))
