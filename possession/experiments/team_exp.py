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
from scipy.ndimage import uniform_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from possession import features, idsse, train  # noqa: E402


def touch_features(players, ball, team, period, fps=25):
    """Who is nearest, when that changes, and whether the ball just got kicked."""
    players = players.astype(np.float64)
    ball = ball.astype(np.float64)
    n = len(ball)
    d_home, d_away = features.team_distances(players, ball, team)
    nearest = np.where(np.isnan(d_home) | np.isnan(d_away), np.nan,
                       (d_away < d_home).astype(np.float64))
    close = np.fmin(d_home, d_away) < 2.5
    owner = np.where(close, nearest, np.nan)          # 0 home / 1 away, only when really close
    held = features.forward_fill(owner)
    f = {}
    f["owner_last"] = held
    f["owner_age"] = np.fmin(features.frames_since_valid(owner), 500)
    change = np.r_[0.0, np.abs(np.diff(held))]
    change = np.nan_to_num(change)
    for w in (12, 50):
        f["owner_changes_c{}".format(w)] = uniform_filter1d(change, 2 * w + 1) * (2 * w + 1)
    # Frames to the next and from the previous confirmed change of owner.
    idx = np.flatnonzero(change > 0.5)
    pos = np.arange(n)
    if len(idx):
        after = np.searchsorted(idx, pos)
        nxt = idx[np.clip(after, 0, len(idx) - 1)] - pos
        prv = pos - idx[np.clip(after - 1, 0, len(idx) - 1)]
        f["to_next_change"] = np.clip(nxt, -1, 250).astype(np.float64)
        f["from_prev_change"] = np.clip(prv, -1, 250).astype(np.float64)
    else:
        f["to_next_change"] = f["from_prev_change"] = np.full(n, 250.0)
    # Share of close frames owned by home in centred windows; the next owner seen after this frame.
    h = np.where(np.isnan(owner), np.nan, 1 - owner)
    for w in (12, 25, 50, 125):
        f["home_share_c{}".format(w)] = features.centred_mean(h, w)
    f["home_share_past25"] = features.window_mean(h, 25, 0)
    f["home_share_next25"] = features.window_mean(h, 0, 25)
    f["next_owner"] = features.forward_fill(owner[::-1])[::-1]
    # Ball acceleration: a kick is a jump in velocity.
    vel = np.full((n, 2), np.nan)
    vel[1:] = np.diff(ball[:, :2], axis=0) * fps
    acc = np.full(n, np.nan)
    acc[3:] = np.hypot(*(vel[3:] - vel[:-3]).T)
    f["ball_acc"] = np.clip(acc, 0, 60)
    f["ball_acc_max_c6"] = features.centred_mean(f["ball_acc"], 6)
    names = list(f)
    return np.column_stack([f[k] for k in names]).astype(np.float32), names


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
