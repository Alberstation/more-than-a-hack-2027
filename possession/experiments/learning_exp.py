"""Learning curve: does v4 gain from more matches? Run with LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/learning_exp.py

mixed_all3 fitted on 2, 4 and 6 IDSSE matches (three subsets each, fixed seed) and on all 7,
scored on Metrica's moving-ball tests at costs (40, 40) with raw and control labels. A flat curve
lowers the priority of new data (SkillCorner, PFF); a rising one justifies it. Diagnostic only.
"""
import json
import os
import pickle
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
SIZES = (2, 4, 6)
SUBSETS = 3
SWITCH, DEAD = 40.0, 40.0
FOLDER = os.path.join(improve.OUT, "learning")


def subsets():
    rng = np.random.default_rng(train.SEED)
    out = [tuple(idsse.MATCHES)]
    for k in SIZES:
        for _ in range(SUBSETS):
            out.append(tuple(sorted(rng.choice(idsse.MATCHES, k, replace=False))))
    return out


def fit(chosen):
    cfg = improve.ARMS[ARM]
    path = os.path.join(FOLDER, "_".join(chosen) + ".pkl")
    if len(chosen) == len(idsse.MATCHES):
        path = os.path.join(metrica_exp.OUT, ARM + ".pkl")
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)[0]
    pool = [improve.load_match(m, g, cfg["blocks"], d) for d in range(cfg["draws"])
            for g in improve.GAPS for m in chosen]
    model = improve.Fitted(dict(cfg, _extra={}), pool, "y_raw", train.SEED)
    with open(path, "wb") as handle:
        pickle.dump((model, None), handle)
    return model


def job(chosen):
    model = fit(chosen)
    rows = []
    for game in metrica_exp.GAMES:
        y_events = live_review.metrica_events(game)
        for repeat in range(metrica_exp.REPEATS):
            for gap in improve.GAPS:
                with np.load(os.path.join(metrica_exp.OUT, "g{}_{}_{}.npz".format(game, repeat, gap))) as f:
                    X = np.hstack([f[b] for b in improve.ARMS[ARM]["blocks"]])
                    y, period = f["y"], f["period"]
                path = improve.decode(improve.log_p(model.proba(X)).astype(np.float64), None, period, SWITCH, 0.0, DEAD)
                y_ctrl = live_review.metrica_control(y, *y_events)
                rows.append(dict(game=game, repeat=repeat, gap=gap, raw=train.score(y, path, period)[0],
                                 ctrl=train.score(y_ctrl, path, period)[0]))
    return dict(matches=list(chosen), rows=rows)


def main():
    os.makedirs(FOLDER, exist_ok=True)
    with ProcessPoolExecutor(3) as pool:
        done = list(pool.map(job, subsets()))
    with open(os.path.join(FOLDER, "curve.json"), "w") as handle:
        json.dump(done, handle, indent=1)
    for gap in improve.GAPS:
        for label in ("ctrl", "raw"):
            print("\ngaps={} labels={}".format(gap, label))
            for k in SIZES + (len(idsse.MATCHES),):
                means = [np.mean([r[label] for r in d["rows"] if r["gap"] == gap])
                         for d in done if len(d["matches"]) == k]
                print("  {} partidos: {:.3f}  ({})".format(k, np.mean(means), ", ".join("{:.3f}".format(v) for v in means)))


if __name__ == "__main__":
    main()
