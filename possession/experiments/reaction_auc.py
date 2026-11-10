"""Plan 3j: does the delayed team reaction tell a real change of possession? Clean IDSSE (best case).

    LIVE_DEAD=1 python possession/experiments/reaction_auc.py

At each change of `ctrl` possession between teams (t), the reaction is how much the gaining team
speeds up towards the opponent goal and the losing team towards its own, mean x-velocity over
[t + 0.5, t + 3] s minus over [t - 2.5, t] s. Negatives: live frames at least 3 s from any change.
AUC of the reaction for all changes and for those starting a possession shorter than 2 s. Out if
AUC < 0.6. Clean tracking, so this is an upper bound for the degraded tracker.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import idsse  # noqa: E402
import improve  # noqa: E402

FPS = 25


def auc(pos, neg):
    allv = np.r_[pos, neg]
    ranks = np.argsort(np.argsort(allv)) + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def match(m, rng):
    data = idsse.load(m)
    P, team, period = data["players"][..., :2].astype(np.float64), data["team"], data["period"]
    with np.load(improve.cache_path(m, "uniform")) as f:
        y = f["y_ctrl"]
    n = len(y)
    vx = np.zeros((n, P.shape[1]))
    vx[1:] = (P[1:, :, 0] - P[:-1, :, 0]) * FPS
    # attacking direction per team and half: away from the side where the team's mean x lies
    sign = np.zeros((n, 2))
    for p in np.unique(period):
        r = period == p
        for s in (0, 1):
            sign[r, s] = -np.sign(np.nanmean(P[r][:, team == s, 0]) - np.nanmean(P[r][:, :, 0]))
    team_v = np.stack([np.nanmean(vx[:, team == s], axis=1) for s in (0, 1)], axis=1) * sign   # +: towards goal
    team_v = np.nan_to_num(team_v)
    c = np.cumsum(np.r_[np.zeros((1, 2)), team_v], axis=0)
    win = lambda a, b, t: (c[np.clip(t + b, 0, n)] - c[np.clip(t + a, 0, n)]) / (b - a)

    def reaction(t, gain):
        d = win(int(0.5 * FPS), 3 * FPS, t) - win(-int(2.5 * FPS), 0, t)
        return d[np.arange(len(t)), gain] - d[np.arange(len(t)), 1 - gain]

    live = y < 2
    change = np.flatnonzero((y[1:] < 2) & (y[:-1] < 2) & (y[1:] != y[:-1])) + 1
    ends = np.r_[change[1:], n]
    short = (ends - change) < 2 * FPS
    near = np.zeros(n, dtype=bool)
    for t in change:
        near[max(0, t - 3 * FPS):t + 3 * FPS] = True
    cand = np.flatnonzero(live & ~near)
    cand = cand[(cand > 3 * FPS) & (cand < n - 3 * FPS)]
    neg_t = rng.choice(cand, min(len(cand), 20 * len(change)), replace=False)
    neg = np.r_[reaction(neg_t, np.zeros(len(neg_t), int)), reaction(neg_t, np.ones(len(neg_t), int))]
    pos = reaction(change, y[change].astype(int))
    # v6's own path (out of fold, uniform gaps, constants 0.75 / 14 / 20): changes it misses within +-1 s
    load = lambda a: np.load(os.path.join(improve.OUT, "oof", a, m + ".npz"))["lp_uniform"].astype(np.float64)
    lp = np.log(np.clip(0.25 * np.exp(load("ctrl_mixed_all3")) + 0.75 * np.exp(load("encoder_metrica3")), 1e-4, 1))
    path = improve.decode(lp, None, period, 14.0, 0.0, 20.0)
    sw = np.zeros(n + 1, dtype=np.int64)
    sw[1:] = np.cumsum(np.r_[False, (path[1:] != path[:-1]) & (path[1:] < 2) & (path[:-1] < 2)])
    found = sw[np.clip(change + FPS, 0, n)] - sw[np.clip(change - FPS, 0, n)] > 0
    return pos, pos[short], neg, pos[~found], float(found.mean())


def main():
    rng = np.random.default_rng(2027)
    out = [match(m, rng) for m in idsse.MATCHES]
    pos, short, neg, missed = (np.concatenate([o[i] for o in out]) for i in range(4))
    print("v6 detecta a +-1 s el {:.0%} de los cambios; AUC en los que no detecta ({}): {:.3f}".format(
        np.mean([o[4] for o in out]), len(missed), auc(missed, neg)))
    print("cambios {} (cortos {}), negativos {}".format(len(pos), len(short), len(neg)))
    print("AUC reaccion diferida: todos {:.3f}  inicio de posesion < 2 s {:.3f}".format(auc(pos, neg), auc(short, neg)))
    per = [auc(o[0], o[2]) for o in out]
    print("por partido: " + " ".join("{:.2f}".format(v) for v in per))


if __name__ == "__main__":
    main()
