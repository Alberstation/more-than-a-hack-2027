"""Fit the v3 model on IDSSE: the arm mixed_all3 of experiments/improve.py, on all 7 matches.

    python possession/train_v3.py

Three degradations of every match under each of two ball-gap patterns, because the challenge's
pattern is not known for sure: gaps at random, as v2 trains, or following the stoppages. Costs are
not searched here. They come from improve.py, chosen on held-out matches.
"""

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

from possession import control, features_v3, idsse, train  # noqa: E402

MODEL_PATH = os.path.join(DATA_DIR, "models", "hgb_v3.pkl")

# Params
DRAWS = 3
GAPS = ("uniform", "dead")
SWITCH_COST = 5.0
DEAD_COST = 40.0
WORKERS = 4


def swap_y(y):
    return np.where(y == 2, 2, 1 - y).astype(np.int8)


def degraded_rows(args):
    """Every TRAIN_STRIDE-th frame of one degraded match, seen from both sides."""
    index, match_id, gaps, draw, lengths, share, moving, labels = args
    data = idsse.load(match_id)
    y = np.array([train.CODE[v] for v in data["labels"]], dtype=np.int8)
    period, pitch, team = data["period"], tuple(data["pitch"]), data["team"]
    seed = [train.SEED, index, draw] if draw else [train.SEED, index]
    streams = np.random.default_rng(seed).spawn(3)
    if moving:
        data = dict(data, ball=train.moving_dead_ball(data["players"], data["ball"], y, streams[2]))
    players, ball = train.degrade(data, streams[0], lengths, 0.0)
    if gaps == "dead":
        hidden = train.dead_mask(y, streams[1], lengths, train.GAP_GIVEN_ALIVE)
    else:
        hidden = train.ball_mask(len(y), streams[1], lengths, share)
    ball[hidden] = np.nan
    s = train.TRAIN_STRIDE
    X = features_v3.build(players, ball, team, period, pitch)[::s]
    X_swap = features_v3.build(players, ball, 1 - team, period, pitch)[::s]
    if labels == "ctrl":
        y = control.control_labels(match_id, y, period)
    return np.concatenate([X, X_swap]), np.concatenate([y[::s], swap_y(y)[::s]])


def main(moving=False, model_path=MODEL_PATH, switch_cost=SWITCH_COST, dead_cost=DEAD_COST, labels="raw"):
    lengths, share = train.challenge_gap_profile()
    jobs = [(i, m, g, d, lengths, float(share), moving, labels) for d in range(DRAWS) for g in GAPS
            for i, m in enumerate(idsse.MATCHES)]
    with ProcessPoolExecutor(WORKERS) as pool:
        parts = list(pool.map(degraded_rows, jobs))
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    print("fitting on", X.shape, flush=True)
    # Unit weights change sklearn's code path and so the trees. improve.py passes them, so this is
    # the model that was measured.
    weight = np.ones(len(y), dtype=np.float32)
    model = HistGradientBoostingClassifier(**dict(train.HGB, early_stopping=False)).fit(X, y, sample_weight=weight)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as handle:
        pickle.dump({"model": model, "switch_cost": switch_cost, "dead_cost": dead_cost}, handle)
    print("->", os.path.relpath(model_path, ROOT))


if __name__ == "__main__":
    main()
