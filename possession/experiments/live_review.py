"""Paired costs table on tests where the dead ball moves. Run with LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/live_review.py <out.json> arm ...

Arms fitted on still balls keep their name with frozen_; IDSSE out-of-fold on raw and control
labels, then the same model family fitted on all IDSSE and tested on Metrica with both labels.
"""
import csv
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
import metrica_exp  # noqa: E402

SWITCH = (5.0, 10.0, 14.0, 20.0, 28.0, 40.0)
DEAD = (10.0, 20.0, 40.0, 80.0)


def idsse_job(args):
    arm, held, gap = args
    with np.load(os.path.join(improve.OUT, "oof", arm, held + ".npz")) as f:
        lp = f["lp_" + gap].astype(np.float64)
    with np.load(improve.cache_path(held, "uniform")) as f:
        y_raw, y_ctrl, period = f["y_raw"], f["y_ctrl"], f["period"]
    rows = {}
    for s in SWITCH:
        for d in DEAD:
            path = improve.decode(lp, None, period, s, 0.0, d)
            rows["{:g}_{:g}".format(s, d)] = dict(raw=train.score(y_raw, path, period)[0],
                                                  ctrl=train.score(y_ctrl, path, period)[0],
                                                  dead_recall=float(np.mean(path[y_raw == 2] == 2)))
    return dict(arm=arm, held=held, gap=gap, rows=rows)


def metrica_events(game):
    frames, teams = [], []
    with open(os.path.join(metrica_exp.FOLDER, "Sample_Game_{}_RawEventsData.csv".format(game))) as handle:
        for row in csv.DictReader(handle):
            if row["Type"] in ("PASS", "SHOT", "SET PIECE") and row["Team"] in ("Home", "Away"):
                frames.append(int(row["Start Frame"]) - 1)
                teams.append(0 if row["Team"] == "Home" else 1)
    return np.array(frames), np.array(teams)


def metrica_control(y, frames, owners):
    window = int(improve.CONTROL_WINDOW_S * improve.FPS)
    out = y.copy()
    for _ in range(3):
        start, end, value = improve.runs(out)
        for i in range(1, len(start)):
            if value[i] < 2 and value[i - 1] < 2:
                near = (frames >= start[i] - window) & (frames < end[i] + window)
                if not np.any(owners[near] == value[i]):
                    out[start[i]:end[i]] = out[start[i] - 1]
    return out


def metrica_job(args):
    arm, game, repeat, gap, lp_path = args
    with np.load(lp_path) as f:
        lp, y, period = f["lp"], f["y"], f["period"]
    y_ctrl = metrica_control(y, *metrica_events(game))
    rows = {}
    for s in SWITCH:
        for d in DEAD:
            path = improve.decode(lp, None, period, s, 0.0, d)
            rows["{:g}_{:g}".format(s, d)] = dict(raw=train.score(y, path, period)[0],
                                                  ctrl=train.score(y_ctrl, path, period)[0],
                                                  dead_recall=float(np.mean(path[y == 2] == 2)))
    return dict(arm=arm, held="g{}_{}".format(game, repeat), gap=gap, rows=rows)


def metrica_probabilities(arm, cfg_blocks):
    """Log-probabilities on Metrica's moving-ball tests, from a model fitted on all IDSSE."""
    path = os.path.join(metrica_exp.OUT, arm + ".pkl")
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            model, _ = pickle.load(handle)
    else:
        cfg = improve.ARMS[arm]
        root = improve.FROZEN_OUT if cfg.get("frozen") else improve.SHARP_OUT if cfg.get("sharp") else None
        pool = [improve.load_match(m, g, cfg_blocks, d, root) for d in range(cfg.get("draws", 1))
                for g in improve.GAPS for m in idsse.MATCHES]
        model = improve.Fitted(dict(cfg, _extra={}), pool, "y_" + cfg.get("labels", "raw"), train.SEED)
        del pool
        with open(path, "wb") as handle:
            pickle.dump((model, None), handle)
    jobs = []
    for game in metrica_exp.GAMES:
        for repeat in range(metrica_exp.REPEATS):
            for gap in improve.GAPS:
                out = os.path.join(metrica_exp.OUT, "lp_{}_{}_{}_{}.npz".format(arm, game, repeat, gap))
                if not os.path.isfile(out):
                    with np.load(os.path.join(metrica_exp.OUT, "g{}_{}_{}.npz".format(game, repeat, gap))) as f:
                        X, y, period = np.hstack([f[b] for b in cfg_blocks]), f["y"], f["period"]
                    np.savez(out, lp=improve.log_p(model.proba(X)).astype(np.float64), y=y, period=period)
                jobs.append((arm, game, repeat, gap, out))
    return jobs


def main():
    target, arms = sys.argv[1], sys.argv[2:]
    jobs = [(a, m, g) for a in arms for m in idsse.MATCHES for g in improve.GAPS]
    with ProcessPoolExecutor(6) as pool:
        id_rows = list(pool.map(idsse_job, jobs))
    m_jobs = [j for a in arms for j in metrica_probabilities(a, improve.ARMS[a]["blocks"])]
    with ProcessPoolExecutor(6) as pool:
        me_rows = list(pool.map(metrica_job, m_jobs))
    with open(target, "w") as handle:
        json.dump(dict(idsse=id_rows, metrica=me_rows), handle)


if __name__ == "__main__":
    main()
