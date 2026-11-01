"""Learned pairwise term: a change detector makes a team flip cheap where it looks like one.

Trains with K degraded draws of each training match, tests on one fixed draw. Caches per-fold
predictions; decoder grid runs in decode_exp.py.
"""
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import idsse, train  # noqa: E402
from team_exp import touch_features, gap_lengths, missing_share  # noqa: E402

K = int(sys.argv[1])
OUT = os.path.join(train.DATA_DIR, "experiments", "oof_change_k{}".format(K))
os.makedirs(OUT, exist_ok=True)
WINDOW = 6


def flip_target(y):
    """1 within WINDOW frames of an H<->A flip, D stretches bridged."""
    alive = np.flatnonzero(y != 2)
    flips = alive[1:][y[alive[1:]] != y[alive[:-1]]]
    out = np.zeros(len(y), dtype=np.int8)
    for f in flips:
        out[max(0, f - WINDOW):f + WINDOW + 1] = 1
    return out


def draw(match_id, rng):
    m = train.match_arrays(match_id, rng, gap_lengths, missing_share)
    a, _ = touch_features(m["players"], m["ball"], m["team"], m["period"])
    b, _ = touch_features(m["players"], m["ball"], 1 - m["team"], m["period"])
    return dict(X=np.hstack([m["X"], a]).astype(np.float32),
                X_swap=np.hstack([m["X_swap"], b]).astype(np.float32),
                y=m["y"], y_swap=m["y_swap"], period=m["period"], flip=flip_target(m["y"]))


rng = np.random.default_rng(train.SEED)
test_draw = {m: draw(m, rng) for m in idsse.MATCHES}
extra = {m: [draw(m, rng) for _ in range(K - 1)] for m in idsse.MATCHES}
print("built", flush=True)

for held in idsse.MATCHES:
    t = time.time()
    pool = [d for m in idsse.MATCHES if m != held for d in [test_draw[m]] + extra[m]]
    model = train.fit(pool)
    Xc = np.concatenate([d["X"][::train.TRAIN_STRIDE] for d in pool])
    yc = np.concatenate([d["flip"][::train.TRAIN_STRIDE] for d in pool])
    detector = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=63,
                                              min_samples_leaf=200, l2_regularization=1.0,
                                              random_state=train.SEED).fit(Xc, yc)
    test = test_draw[held]
    lp = train.log_proba(model, test["X"])
    p = np.clip(detector.predict_proba(test["X"])[:, 1], 1e-4, 1 - 1e-4)
    np.savez_compressed(os.path.join(OUT, held + ".npz"), lp=lp, logit=np.log(p / (1 - p)),
                        y=test["y"], period=test["period"])
    print("fold", held, round(time.time() - t), "flip share", round(float(yc.mean()), 3), flush=True)
print("done", flush=True)
