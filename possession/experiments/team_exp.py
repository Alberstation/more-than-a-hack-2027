"""What limits the team decision? Compare three variants with the same CV:
  base      current features on degraded matches
  clean     current features, no degradation (upper bound: how much does losing the ball cost?)
  touch     degraded, plus touch/possessor features
Scored with the official functions on held-out matches; the degraded test is what matters.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from possession import features, idsse, train  # noqa: E402
from possession.features_v3 import touch_features  # noqa: E402


def run(label, make):
    t = time.time()
    rng = np.random.default_rng(train.SEED)
    matches = {m: make(m, rng) for m in idsse.MATCHES}
    scores = []
    for held in idsse.MATCHES:
        test = matches[held]
        model = train.fit([m for k, m in matches.items() if k != held])
        lp = train.log_proba(model, test["X"])
        scores.append({g: train.score(test["y"], train.decode_periods(lp, test["period"], *g),
                                      test["period"]) for g in GRID})
    mean = {g: np.mean([s[g] for s in scores], axis=0) for g in GRID}
    best = max(GRID, key=lambda g: mean[g][0])
    per = [round(float(s[best][0]), 3) for s in scores]
    print("{:<8} best {} spell {:.4f} macro {:.4f} folds {} [{:.0f}s]".format(
        label, best, mean[best][0], mean[best][1], per, time.time() - t), flush=True)


GRID = [(s, d) for s in (2.5, 5.0, 10.0, 20.0) for d in (10.0, 20.0, 40.0, 80.0)]
gap_lengths, missing_share = train.challenge_gap_profile()


def base(match_id, rng):
    return train.match_arrays(match_id, rng, gap_lengths, missing_share)


def clean(match_id, rng):
    data = idsse.load(match_id)
    y = np.array([train.CODE[v] for v in data["labels"]], dtype=np.int8)
    pitch = tuple(data["pitch"])
    X, names = features.build(data["players"], data["ball"], data["team"], data["period"], pitch)
    X_swap, _ = features.build(data["players"], data["ball"], 1 - data["team"], data["period"], pitch)
    return dict(X=X, y=y, X_swap=X_swap, y_swap=np.where(y == 2, 2, 1 - y).astype(np.int8),
                period=data["period"], names=names)


def touch(match_id, rng):
    m = base(match_id, rng)
    a, _ = touch_features(m["players"], m["ball"], m["team"], m["period"])
    b, _ = touch_features(m["players"], m["ball"], 1 - m["team"], m["period"])
    m["X"] = np.hstack([m["X"], a])
    m["X_swap"] = np.hstack([m["X_swap"], b])
    return m


if __name__ == "__main__":
    run("base", base)
    run("clean", clean)
    run("touch", touch)
