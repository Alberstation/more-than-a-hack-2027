"""The evidence added on top of features.build for v3. Still no player identities.

touch: who is nearest and when that changes. shape: identity-free team shape, which is what tells
the teams apart inside a long ball gap. control: retention and pass chains.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

from possession import features

FPS = 25


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


def shape_features(players, team, period, fps=FPS):
    """Identity-free team shape, home side's point of view."""
    players = players.astype(np.float64)
    right = features.home_attacks_right(players, team, period)
    length = float(np.nanmax(players[..., 0]))
    px = np.where(right[:, None], players[:, :, 0], length - players[:, :, 0])
    py = players[:, :, 1]
    f = {}
    q = {}
    for name, cols in (("home", team == 0), ("away", team == 1)):
        x, y_ = px[:, cols], py[:, cols]
        seen = np.sum(~np.isnan(x), axis=1)
        filled = np.where(np.isnan(x), np.inf, x)
        order = np.sort(filled, axis=1)
        pick = lambda share: order[np.arange(len(x)), np.clip((seen * share).astype(int), 0, x.shape[1] - 1)]
        q[name] = [pick(s) for s in (0.1, 0.5, 0.9)]
        for s, v in zip(("q10", "q50", "q90"), q[name]):
            f["{}_{}".format(s, name)] = np.where(seen > 3, v, np.nan)
        spread_x = np.nanmax(x, axis=1, initial=-1e9) - np.nanmin(x, axis=1, initial=1e9)
        spread_y = np.nanmax(y_, axis=1, initial=-1e9) - np.nanmin(y_, axis=1, initial=1e9)
        f["box_" + name] = np.where(seen > 3, spread_x * spread_y, np.nan)
        centre = np.where(seen > 0, np.sum(np.where(np.isnan(x), 0, x), axis=1) / np.maximum(seen, 1), np.nan)
        vel = np.full(len(x), np.nan)
        vel[12:-12] = (centre[24:] - centre[:-24]) * fps / 24
        f["advance_" + name] = features.centred_mean(vel, 12)
    f["home_final_third"] = np.nansum(px[:, team == 0] > 2 * length / 3, axis=1).astype(np.float64)
    f["away_final_third"] = np.nansum(px[:, team == 1] < length / 3, axis=1).astype(np.float64)
    f["line_gap"] = f["q90_home"] - f["q10_away"]
    for name in ("home_final_third", "away_final_third", "line_gap"):
        f[name + "_c50"] = features.centred_mean(f[name], 50)
    return np.column_stack(list(f.values())).astype(np.float32)


def control_features(players, ball, team):
    """Retention and pass chains, with `unknown` kept apart from `nothing happened`."""
    dx = players[:, :, 0] - ball[:, None, 0]
    dy = players[:, :, 1] - ball[:, None, 1]
    dist = np.sqrt(dx * dx + dy * dy).astype(np.float64)
    known = ~np.isnan(ball[:, 0])
    filled = np.where(np.isnan(dist), np.inf, dist)
    slot = np.argmin(filled, axis=1)
    near = filled[np.arange(len(ball)), slot] < 2.5
    held = np.where(known & near, slot, -1)
    side = np.where(held >= 0, team[np.maximum(held, 0)], -1).astype(np.float64)
    side[side < 0] = np.nan
    side_last = features.forward_fill(side)
    pass_ = {0: np.zeros(len(ball)), 1: np.zeros(len(ball))}
    frames = np.flatnonzero(held >= 0)
    if len(frames) > 1:
        a, b = held[frames[:-1]], held[frames[1:]]
        same = (a != b) & (team[a] == team[b]) & (frames[1:] - frames[:-1] < 75)
        for t in (0, 1):
            pass_[t][frames[1:][same & (team[b] == t)]] = 1
    f = {}
    for w in (50, 125):
        for t, name in ((0, "home"), (1, "away")):
            f["chain_{}_c{}".format(name, w)] = uniform_filter1d(pass_[t], 2 * w + 1) * (2 * w + 1)
        f["chain_diff_c{}".format(w)] = f["chain_home_c{}".format(w)] - f["chain_away_c{}".format(w)]
    change = np.r_[False, np.nan_to_num(np.diff(side_last)) != 0]
    since = features.frames_since_valid(np.where(change, 1.0, np.nan))
    f["retained"] = np.where(side_last == 0, 1.0, -1.0) * np.fmin(np.nan_to_num(since, nan=500.0), 500)
    f["retained"][np.isnan(side_last)] = np.nan
    f["seen_c50"] = uniform_filter1d(known.astype(np.float64), 101)
    return np.column_stack(list(f.values())).astype(np.float32)


def build(players, ball, team, period, pitch):
    """(n_frames, n_features) in the order the v3 model was fitted on."""
    return np.hstack([features.build(players, ball, team, period, pitch)[0],
                      touch_features(players, ball, team, period)[0],
                      shape_features(players, team, period),
                      control_features(players, ball, team)])
