"""Two-stage HGB on v4: a second model reads minutes of v4's probabilities. Run with LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/stack_exp.py l1    # nested stage-1 fits, 42 of them
    LIVE_DEAD=1 python possession/experiments/stack_exp.py run   # stage 2, scored against ctrl

Stage 1 is mixed_all3. For held match h and inner match m, stage 1 is fitted on the other five, so
the probabilities stage 2 learns from never saw m or h; on h, stage 2 reads the usual out-of-fold
probabilities. Stage 2 learns the control labels from context windows of those probabilities and
from the run they fall in. Viterbi costs for a match are picked on the other six; Metrica uses a
stage 2 fitted on all seven out-of-fold sets and the median costs. Paired with v4 at (40, 40).
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
FOLDER = os.path.join(improve.OUT, "stack")
V4 = (40.0, 40.0)
CENTRED_S = (1, 5, 15, 60)
SIDED_S = (5, 15)
SWITCH = (5.0, 10.0, 20.0, 40.0, 80.0)
DEAD = (20.0, 40.0, 80.0)


# -- stage 1 -----------------------------------------------------------------------------------

def l1_path(held, inner):
    return os.path.join(FOLDER, "l1", "{}_{}.npz".format(held, inner))


def l1_job(args):
    held, inner = args
    path = l1_path(held, inner)
    if os.path.isfile(path):
        return path
    cfg = improve.ARMS[ARM]
    rest = [m for m in idsse.MATCHES if m not in (held, inner)]
    pool = [improve.load_match(m, g, cfg["blocks"], d) for d in range(cfg["draws"]) for g in improve.GAPS for m in rest]
    model = improve.Fitted(dict(cfg, _extra={}), pool, "y_raw", train.SEED)
    del pool
    out = {}
    for g in improve.GAPS:
        out["lp_" + g] = improve.log_p(model.proba(improve.load_match(inner, g, cfg["blocks"])["X"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    print("l1", held, inner, flush=True)
    return path


# -- stage 2 -----------------------------------------------------------------------------------

def window_mean(p, period, before, after):
    out = np.empty_like(p)
    for q in np.unique(period):
        r = np.flatnonzero(period == q)
        c = np.vstack([np.zeros((1, p.shape[1])), np.cumsum(p[r], axis=0)])
        i = np.arange(len(r))
        lo, hi = np.maximum(i - before, 0), np.minimum(i + after + 1, len(r))
        empty = hi <= lo    # a one-sided window at the edge of a period: use the frame itself
        lo, hi = np.where(empty, i, lo), np.where(empty, i + 1, hi)
        out[r] = (c[hi] - c[lo]) / (hi - lo)[:, None]
    return out


def context(lp, period):
    p = np.exp(lp.astype(np.float64))
    cols = [lp.astype(np.float64)]
    for s in CENTRED_S:
        cols.append(window_mean(p, period, s * improve.FPS, s * improve.FPS))
    for s in SIDED_S:
        cols.append(window_mean(p, period, s * improve.FPS, -1))
        cols.append(window_mean(p, period, -1, s * improve.FPS))
    path = improve.decode(lp, None, period, V4[0], 0.0, V4[1])
    run = np.empty((len(p), 7))
    for s, e, v in zip(*improve.runs(path)):
        n = e - s
        run[s:e, 0] = v
        run[s:e, 1] = np.log(n / improve.FPS)
        run[s:e, 2] = np.minimum(np.arange(n), 1500) / improve.FPS
        run[s:e, 3] = np.minimum(np.arange(n)[::-1], 1500) / improve.FPS
        run[s:e, 4:7] = p[s:e].mean(axis=0)
    cols.append(run)
    return np.hstack(cols).astype(np.float32)


def swapped(X):
    """Same features with home and away exchanged: every H, A column pair flips."""
    out = X.copy()
    for start in range(0, X.shape[1] - 7, 3):
        out[:, [start, start + 1]] = X[:, [start + 1, start]]
    run = X.shape[1] - 7
    out[:, run] = np.where(X[:, run] == 2, 2, 1 - X[:, run])
    out[:, run + 4], out[:, run + 5] = X[:, run + 5], X[:, run + 4]
    return out


def fit_stage2(sets):
    X, y = [], []
    s = train.TRAIN_STRIDE
    for XX, yy in sets:
        X += [XX[::s], swapped(XX)[::s]]
        y += [yy[::s], improve.swap_y(yy)[::s]]
    return improve.hgb(train.SEED).fit(np.concatenate(X), np.concatenate(y))


def truth(m):
    with np.load(improve.cache_path(m, "uniform")) as f:
        return f["y_ctrl"], f["period"]


def oof_lp(m, gap):
    with np.load(os.path.join(improve.OUT, "oof", ARM, m + ".npz")) as f:
        return f["lp_" + gap].astype(np.float64)


def stage2_job(held):
    sets = []
    for m in idsse.MATCHES:
        if m == held:
            continue
        y, period = truth(m)
        with np.load(l1_path(held, m)) as f:
            for g in improve.GAPS:
                sets.append((context(f["lp_" + g].astype(np.float64), period), y))
    model = fit_stage2(sets)
    y, period = truth(held)
    rows = {}
    for g in improve.GAPS:
        lp = oof_lp(held, g)
        lp2 = improve.log_p(model.predict_proba(context(lp, period))).astype(np.float64)
        table = {"v4": train.score(y, improve.decode(lp, None, period, V4[0], 0.0, V4[1]), period)[0]}
        for s in SWITCH:
            for d in DEAD:
                table["{:g}_{:g}".format(s, d)] = train.score(y, improve.decode(lp2, None, period, s, 0.0, d), period)[0]
        rows[g] = table
    print("stage2", held, flush=True)
    return held, rows


def metrica_job(costs):
    sets = []
    for m in idsse.MATCHES:
        y, period = truth(m)
        for g in improve.GAPS:
            sets.append((context(oof_lp(m, g), period), y))
    model = fit_stage2(sets)
    rows = {g: [] for g in improve.GAPS}
    for game in metrica_exp.GAMES:
        events = live_review.metrica_events(game)
        for repeat in range(metrica_exp.REPEATS):
            for g in improve.GAPS:
                with np.load(os.path.join(metrica_exp.OUT, "lp_{}_{}_{}_{}.npz".format(ARM, game, repeat, g))) as f:
                    lp, y, period = f["lp"].astype(np.float64), f["y"], f["period"]
                y = live_review.metrica_control(y, *events)
                lp2 = improve.log_p(model.predict_proba(context(lp, period))).astype(np.float64)
                s, d = costs[g]
                rows[g].append((train.score(y, improve.decode(lp, None, period, V4[0], 0.0, V4[1]), period)[0],
                                train.score(y, improve.decode(lp2, None, period, s, 0.0, d), period)[0]))
    return rows


def run():
    with ProcessPoolExecutor(4) as pool:
        done = dict(pool.map(stage2_job, idsse.MATCHES))
    summary, costs = {}, {}
    for g in improve.GAPS:
        keys = [k for k in done[idsse.MATCHES[0]][g] if k != "v4"]
        diffs, picked = [], []
        for held in idsse.MATCHES:
            others = [m for m in idsse.MATCHES if m != held]
            best = max(keys, key=lambda k: np.mean([done[m][g][k] for m in others]))
            diffs.append(done[held][g][best] - done[held][g]["v4"])
            picked.append(best)
        s = float(np.median([float(k.split("_")[0]) for k in picked]))
        d = float(np.median([float(k.split("_")[1]) for k in picked]))
        costs[g] = (min(SWITCH, key=lambda v: abs(v - s)), min(DEAD, key=lambda v: abs(v - d)))
        summary[g] = dict(v4=float(np.mean([done[m][g]["v4"] for m in idsse.MATCHES])), diff=float(np.mean(diffs)),
                          up=int(np.sum(np.array(diffs) > 0)), folds=[round(float(v), 4) for v in diffs],
                          picked=picked, metrica_costs=costs[g])
    me = metrica_job(costs)
    for g in improve.GAPS:
        base, new = np.array(me[g]).T
        summary[g].update(metrica_v4=float(base.mean()), metrica_diff=float((new - base).mean()))
    with open(os.path.join(FOLDER, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    for g, v in summary.items():
        print("{:<8} IDSSE v4 {:.3f} {:+.4f} ({}/7) {}  Metrica v4 {:.3f} {:+.4f}  costes {}".format(
            g, v["v4"], v["diff"], v["up"], v["folds"], v["metrica_v4"], v["metrica_diff"], v["metrica_costs"]))


if __name__ == "__main__":
    if sys.argv[1] == "l1":
        jobs = [(h, m) for h in idsse.MATCHES for m in idsse.MATCHES if m != h]
        with ProcessPoolExecutor(3) as pool:
            list(pool.map(l1_job, jobs))
    elif sys.argv[1] == "run":
        run()
