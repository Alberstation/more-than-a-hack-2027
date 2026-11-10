"""A minimum length for every dead-ball stretch, inside the Viterbi decoder. LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/decode_min.py [hgb] [encoder]

The dead state becomes a chain of k sub-states: entering D starts at the first, each frame moves
one step on, and play can only resume from the last one, so no stoppage is shorter than k frames.
The start of a half may begin in the last sub-state, so a half can open with a short D. k = 0 is
today's decoder. Mixture weight, costs and k are nested over the other six IDSSE matches (control
labels, both gap patterns), Metrica games 1-3 decoded at the medians. The alternative k values
are fixed here, before any score: 0, 0.5, 1 and 2 s.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import idsse, train  # noqa: E402
import improve  # noqa: E402
import live_review  # noqa: E402
import metrica_exp  # noqa: E402
import metrica_mix  # noqa: E402
import variant_score as vs  # noqa: E402

MIN_FRAMES = (0, 12, 25, 50)
WEIGHTS = vs.WEIGHTS
SWITCH = vs.SWITCH
DEAD = vs.DEAD


@numba.njit(cache=True)
def viterbi_min(em, switch, dead, k):
    """States: 0 H, 1 A, 2 .. 2+k-1 the dead chain (k >= 1)."""
    n = em.shape[0]
    s = 2 + k
    neg = -1e18
    score = np.full(s, neg)
    score[0], score[1], score[s - 1] = em[0, 0], em[0, 1], em[0, 2]
    back = np.empty((n, s), dtype=np.int16)
    new = np.empty(s)
    for i in range(1, n):
        # H and A: stay, switch, or come back from the end of the dead chain.
        for t in range(2):
            best, arg = score[t], t
            c = score[1 - t] - switch
            if c > best:
                best, arg = c, 1 - t
            c = score[s - 1] - dead
            if c > best:
                best, arg = c, s - 1
            new[t] = best + em[i, t]
            back[i, t] = arg
        # First dead sub-state: entered from H or A.
        if score[0] >= score[1]:
            new[2], back[i, 2] = score[0] - dead + em[i, 2], 0
        else:
            new[2], back[i, 2] = score[1] - dead + em[i, 2], 1
        if k == 1:
            if score[2] + em[i, 2] > new[2]:
                new[2], back[i, 2] = score[2] + em[i, 2], 2
        for j in range(3, s):
            best, arg = score[j - 1], j - 1
            if j == s - 1 and score[j] > best:
                best, arg = score[j], j
            new[j] = best + em[i, 2]
            back[i, j] = arg
        score[:] = new
    path = np.empty(n, dtype=np.int8)
    state = int(np.argmax(score))
    for i in range(n - 1, -1, -1):
        path[i] = min(state, 2)
        if i > 0:
            state = back[i, state]
    return path


def decode(lp, period, switch, dead, k):
    if k == 0:
        return improve.decode(lp, None, period, switch, 0.0, dead)
    out = np.empty(len(lp), dtype=np.int8)
    for p in np.unique(period):
        r = np.flatnonzero(period == p)
        out[r] = viterbi_min(lp[r].astype(np.float64), float(switch), float(dead), int(k))
    return out


def grid(lp_h, lp_e, period):
    out = {}
    for w in WEIGHTS:
        lp = np.log(np.clip((1 - w) * np.exp(lp_h) + w * np.exp(lp_e), 1e-4, 1.0))
        for s in SWITCH:
            for d in DEAD:
                for k in MIN_FRAMES:
                    out[(w, s, d, k)] = decode(lp, period, s, d, k)
    return out


def idsse_job(args):
    hgb, enc, m, gap = args
    with np.load(improve.cache_path(m, "uniform")) as f:
        y, period = f["y_ctrl"], f["period"]
    paths = grid(vs.load_oof(hgb, m, gap)[0], vs.load_oof(enc, m, gap)[0], period)
    return m, gap, {k: train.score(y, p, period)[0] for k, p in paths.items()}


def metrica_job(args):
    hgb, enc, game, repeat, gap = args
    def load(a):
        with np.load(metrica_mix.lp_path(a, game, repeat, gap)) as f:
            return f["lp"].astype(np.float64), f["y"], f["period"]
    lp_h, y, period = load(hgb)
    y = live_review.metrica_control(y, *live_review.metrica_events(game))
    paths = grid(lp_h, load(enc)[0], period)
    return game, gap, {k: train.score(y, p, period)[0] for k, p in paths.items()}


def nested(by, allowed):
    folds, picked = [], []
    for held in idsse.MATCHES:
        keys = [k for k in by[held] if k[3] in allowed]
        best = max(keys, key=lambda k: np.mean([by[m][k] for m in idsse.MATCHES if m != held]))
        folds.append(by[held][best])
        picked.append(best)
    med = np.median(np.array(picked, dtype=float), axis=0)
    snap = lambda g, x: min(g, key=lambda v: abs(v - x))
    key = (snap(WEIGHTS, med[0]), snap(SWITCH, med[1]), snap(DEAD, med[2]), snap(allowed, med[3]))
    return np.array(folds), picked, key


def main():
    hgb = sys.argv[1] if len(sys.argv) > 1 else "ctrl_mixed_all3"
    enc = sys.argv[2] if len(sys.argv) > 2 else "encoder"
    with ProcessPoolExecutor(6) as pool:
        id_done = list(pool.map(idsse_job, [(hgb, enc, m, g) for m in idsse.MATCHES for g in improve.GAPS]))
        me_done = list(pool.map(metrica_job, [(hgb, enc, x, r, g) for x in metrica_exp.ALL_GAMES
                                              for r in range(metrica_exp.REPEATS) for g in improve.GAPS]))
    summary = {}
    for gap in improve.GAPS:
        by = {m: t for m, g, t in id_done if g == gap}
        print("\nhuecos {}  ({} + {})   IDSSE anidado | Metrica 1-3".format(gap, hgb, enc))
        ref = None
        for name, allowed in (("hoy (k = 0)", (0,)), ("k libre", MIN_FRAMES)) + tuple(
                ("k = {:.1f} s fijo".format(k / 25), (k,)) for k in MIN_FRAMES[1:]):
            folds, picked, key = nested(by, allowed)
            me = float(np.mean([t[key] for x, g, t in me_done if g == gap]))
            if ref is None:
                ref = (folds, me)
            d = folds - ref[0]
            summary["{}|{}".format(name, gap)] = dict(idsse=float(folds.mean()), diff=float(d.mean()), up=int(np.sum(d > 0)),
                                                      metrica=me, metrica_diff=me - ref[1], key=[float(v) for v in key])
            print("  {:<18} {:.3f} {:+.4f} ({}/7) | {:.3f} {:+.4f}  [w {:g}, s {:g}, d {:g}, k {:g}]".format(
                name, folds.mean(), d.mean(), int(np.sum(d > 0)), me, me - ref[1], *key))
    with open(os.path.join(improve.OUT, "decode_min.json"), "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
