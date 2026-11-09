"""Who strikes the ball: touch evidence that survives a video tracker's smoothed kicks.

The short control possessions start with a touch by their own team as often as the long ones do,
while the opponent stands just as close (label_audit.py). The challenge ball changes velocity about
three times more gently than IDSSE's, so a kick is looked for over +-3, +-6 and +-12 frames and by
its change of direction, not by a one-frame jump. Every side-specific value is from the point of
view of the team marked 0 in `team`, so passing 1 - team gives the swapped view.

Thresholds are fixed here from the label audit and the calibration, before any score was seen.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from possession import features

FPS = 25
SPANS = (3, 6, 12)
KICK_DV_MS = 3.0       # velocity change over +-6 frames that counts as a strike, on a blurred ball
PEAK_HALF = 6          # a strike is the largest change within +-6 frames
NEAR_M = 3.0           # nearest player this close to the ball around the strike: his team strikes
MOVING_MS = 1.0        # below this, the direction of the ball is not defined
SHARE_WINDOWS = (25, 50, 125)


def kick_features(players, ball, team):
    """(n_frames, n_features), names."""
    B = ball[:, :2].astype(np.float64)
    n = len(B)
    v = np.full((n, 2), np.nan)
    v[1:-1] = (B[2:] - B[:-2]) * FPS / 2
    f = {}
    dv = {}
    for k in SPANS:
        d = np.full(n, np.nan)
        d[k:-k] = np.hypot(*(v[2 * k:] - v[:-2 * k]).T)
        dv[k] = d
        f["kick_dv{}".format(k)] = np.clip(d, 0, 60)
    k = PEAK_HALF
    before, after = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    before[k:], after[:-k] = v[:-k], v[k:]
    norm = np.hypot(*before.T) * np.hypot(*after.T)
    moving = (np.hypot(*before.T) > MOVING_MS) & (np.hypot(*after.T) > MOVING_MS)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["kick_cos6"] = np.where(moving, np.sum(before * after, axis=1) / norm, np.nan)
        f["kick_speed_ratio6"] = np.where(moving, np.log(np.hypot(*after.T) / np.hypot(*before.T)), np.nan)

    # Strikes: local peaks of the +-6 frame change, with the nearest team around them.
    d6 = np.nan_to_num(dv[6], nan=0.0)
    peak = (d6 >= KICK_DV_MS) & (d6 >= maximum_filter1d(d6, 2 * PEAK_HALF + 1, mode="nearest"))
    d_home, d_away = features.team_distances(players.astype(np.float64), B, team)
    big = 1e3
    near_home = -maximum_filter1d(-np.nan_to_num(d_home, nan=big), 2 * PEAK_HALF + 1, mode="nearest")
    near_away = -maximum_filter1d(-np.nan_to_num(d_away, nan=big), 2 * PEAK_HALF + 1, mode="nearest")
    near = np.fmin(near_home, near_away)
    known = peak & (near <= NEAR_M) & (near_home < big) & (near_away < big)
    side = np.full(n, np.nan)
    side[known] = (near_away[known] < near_home[known]).astype(np.float64)   # 0 home, 1 away
    margin = np.full(n, np.nan)
    margin[known] = np.clip(near_away[known] - near_home[known], -10, 10)    # > 0: home nearer
    strike = np.where(peak, 1.0, np.nan)

    f["kick_side_last"] = features.forward_fill(side)
    f["kick_side_next"] = features.forward_fill(side[::-1])[::-1]
    f["kick_margin_last"] = features.forward_fill(margin)
    f["kick_margin_next"] = features.forward_fill(margin[::-1])[::-1]
    f["kick_age"] = np.fmin(features.frames_since_valid(side), 500)
    f["kick_to_next"] = np.fmin(features.frames_since_valid(side[::-1])[::-1], 250)
    f["strike_age"] = np.fmin(features.frames_since_valid(strike), 500)
    # Did the last two strikes come from the same side: kept, or won back?
    sides = side[known]
    same = np.full(n, np.nan)
    if len(sides) > 1:
        same[np.flatnonzero(known)[1:]] = (sides[1:] == sides[:-1]).astype(np.float64)
    f["kick_same_last"] = features.forward_fill(same)
    home = np.where(np.isnan(side), np.nan, 1 - side)
    for w in SHARE_WINDOWS:
        f["kick_home_share_c{}".format(w)] = features.centred_mean(home, w)
        f["kick_count_c{}".format(w)] = uniform_filter1d(known.astype(np.float64), 2 * w + 1) * (2 * w + 1)
    f["kick_home_share_past50"] = features.window_mean(home, 50, 0)
    f["kick_home_share_next50"] = features.window_mean(home, 0, 50)
    names = list(f)
    return np.column_stack([f[k] for k in names]).astype(np.float32), names
