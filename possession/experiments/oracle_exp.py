"""Where v4 loses: boundaries or labels. Out-of-fold mixed_all3, costs (40, 40). Run with LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/oracle_exp.py

v4_cuts_true_labels: each v4 run takes the majority true label (labels perfect, v4 boundaries).
true_cuts_v4_vote / true_cuts_v4_mean: each true run takes v4's majority frame, or the class with
the highest mean probability (boundaries perfect, v4 labels; the second is the run-rescore ceiling).
true_dead: Viterbi forced to the true dead frames. Diagnostic only; decides which arms exist.
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

ARM = "mixed_all3"
SWITCH, DEAD = 40.0, 40.0


def relabel(cuts, pick):
    out = np.empty(len(cuts), dtype=np.int8)
    for s, e, _ in zip(*improve.runs(cuts)):
        out[s:e] = pick(s, e)
    return out


def majority(a):
    return int(np.argmax(np.bincount(a, minlength=3)))


def job(args):
    held, gap = args
    with np.load(os.path.join(improve.OUT, "oof", ARM, held + ".npz")) as f:
        lp = f["lp_" + gap].astype(np.float64)
    with np.load(improve.cache_path(held, "uniform")) as f:
        truth = {k: f["y_" + k] for k in improve.LABELS}
        period = f["period"]
    v4 = improve.decode(lp, None, period, SWITCH, 0.0, DEAD)
    p = np.exp(lp)
    rows = {}
    for k, y in truth.items():
        forced = lp.copy()
        forced[y == 2, :2] = -1e9
        forced[y != 2, 2] = -1e9
        paths = dict(
            v4=v4,
            v4_cuts_true_labels=relabel(v4, lambda s, e: majority(y[s:e])),
            true_cuts_v4_vote=relabel(y, lambda s, e: majority(v4[s:e])),
            true_cuts_v4_mean=relabel(y, lambda s, e: int(np.argmax(p[s:e].mean(axis=0)))),
            true_dead=improve.decode(forced, None, period, SWITCH, 0.0, DEAD),
        )
        rows[k] = {name: train.score(y, path, period)[0] for name, path in paths.items()}
    return held, gap, rows


def main():
    jobs = [(m, g) for m in idsse.MATCHES for g in improve.GAPS]
    with ProcessPoolExecutor(6) as pool:
        done = list(pool.map(job, jobs))
    out = {}
    for held, gap, rows in done:
        for k, by_name in rows.items():
            for name, v in by_name.items():
                out.setdefault("{}|{}".format(gap, k), {}).setdefault(name, []).append(round(float(v), 4))
    with open(os.path.join(improve.OUT, "oracle.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    for key, by_name in out.items():
        print("\n" + key)
        for name, folds in by_name.items():
            print("  {:<22} {:.3f}  ({:.3f}-{:.3f})".format(name, np.mean(folds), min(folds), max(folds)))


if __name__ == "__main__":
    main()
