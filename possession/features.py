"""Per-frame evidence from the tracking, all vectorised over the whole match.

The same code runs on the challenge match (atd.py) and on the external IDSSE matches (idsse.py),
so a model fitted on one sees exactly the features it will be applied to on the other. Nothing
here uses player identities -- only positions, speeds and team membership -- because the
challenge tracking has none.

Everything is expressed from the HOME side's point of view, with x measured in the direction home
attacks. Swapping the team array (1 - team) therefore gives the away side's point of view of the
same frames, which is how the model is trained symmetric in the two teams.
"""

import warnings

import numpy as np

HOME, AWAY = 0, 1


# -- small array helpers -----------------------------------------------------------------------

def window_mean(values, before, after):
    """NaN-aware mean over [t - before, t + after], for every t."""
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(filled)))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    low = np.clip(index - before, 0, len(values))
    high = np.clip(index + after + 1, 0, len(values))
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cumulative[high] - cumulative[low]) / np.maximum(n, 1), np.nan)


def centred_mean(values, half_width):
    return window_mean(values, half_width, half_width)


def forward_fill(values):
    """Carry the last non-NaN value forward. Leading NaNs stay NaN."""
    valid = ~np.isnan(values)
    index = np.where(valid, np.arange(len(values)), 0)
    np.maximum.accumulate(index, out=index)
    out = values[index]
    out[~np.maximum.accumulate(valid)] = np.nan
    return out


def frames_since_valid(values):
    """Frames since the last non-NaN value (0 on a valid frame, NaN before the first one)."""
    valid = ~np.isnan(values)
    index = np.arange(len(values))
    last = np.maximum.accumulate(np.where(valid, index, -1))
    return np.where(last >= 0, index - last, np.nan).astype(np.float64)


def speeds_from_positions(xy, period, fps):
    """Speed in m/s by central difference over +/- 2 frames, within each period.

    Recomputed rather than read from the provider, so both data sources go through one formula.
    """
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float64)
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        block = xy[rows].astype(np.float64)
        if len(rows) > 4:
            out[rows[2:-2]] = np.linalg.norm(block[4:] - block[:-4], axis=-1) * fps / 4
    return out


# -- the evidence the hand-written v1 pipeline uses --------------------------------------------

def team_distances(players, ball, team):
    """(d_home, d_away): metres from the ball to the nearest player of each side, NaN if unknown."""
    dx = players[:, :, 0] - ball[:, None, 0]
    dy = players[:, :, 1] - ball[:, None, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    out = []
    for side in (HOME, AWAY):
        side_dist = dist[:, team == side]
        any_seen = ~np.all(np.isnan(side_dist), axis=1)
        nearest = np.full(len(dist), np.nan, dtype=np.float64)
        nearest[any_seen] = np.nanmin(side_dist[any_seen], axis=1)
        out.append(nearest)
    return out[0], out[1]


def collective_speed(players, half_width):
    """Mean speed of every tracked player, smoothed. Stoppages are walking pace, play is not."""
    speed = players[:, :, 2]
    seen = np.sum(~np.isnan(speed), axis=1)
    per_frame = np.where(seen > 0, np.nansum(speed, axis=1) / np.maximum(seen, 1), np.nan)
    return centred_mean(per_frame.astype(np.float64), half_width)


def ball_outside(ball, pitch, margin):
    """1 where the last seen ball position is beyond a line by more than `margin` metres."""
    return outside_distance(ball, pitch) > margin


def outside_distance(ball, pitch):
    """Metres beyond the nearest line of the last SEEN ball position, held through a gap: a ball
    that leaves the pitch and then vanishes is how most stoppages look in automatic tracking."""
    length, width = pitch
    x = forward_fill(ball[:, 0].astype(np.float64))
    y = forward_fill(ball[:, 1].astype(np.float64))
    with np.errstate(invalid="ignore"):
        return np.nanmax(np.stack([-x, x - length, -y, y - width, np.zeros_like(x)]), axis=0)


# -- the feature matrix for the learned model --------------------------------------------------

def home_attacks_right(players, team, period):
    """Per frame: True where home attacks towards x = length in that period.

    Read off the data -- in each period the side whose players stand further left on average is
    the one defending the left goal -- so no orientation metadata is needed from either provider.
    """
    out = np.zeros(len(players), dtype=bool)
    x = players[:, :, 0]
    for p in np.unique(period):
        rows = period == p
        home_x = np.nanmean(x[rows][:, team == HOME])
        away_x = np.nanmean(x[rows][:, team == AWAY])
        out[rows] = home_x < away_x
    return out


def build(players, ball, team, period, pitch, fps=25):
    """(feature matrix, names) for every frame, from the home side's point of view."""
    length, width = pitch
    players = players.astype(np.float64)
    ball = ball.astype(np.float64)
    right = home_attacks_right(players, team, period)

    px = np.where(right[:, None], players[:, :, 0], length - players[:, :, 0])
    py = players[:, :, 1]
    bx = np.where(right, ball[:, 0], length - ball[:, 0])
    by = ball[:, 1]
    pspeed = speeds_from_positions(players[:, :, :2], period, fps)
    bspeed = speeds_from_positions(ball[:, :2], period, fps)
    home, away = team == HOME, team == AWAY

    f = {}
    # Ball and its nearest players.
    d_home, d_away = team_distances(players, ball, team)
    f["ball_seen"] = (~np.isnan(ball[:, 0])).astype(np.float64)
    f["d_home"], f["d_away"] = d_home, d_away
    f["d_diff"] = np.clip(d_away - d_home, -10, 10)
    dist = np.sqrt((players[:, :, 0] - ball[:, None, 0]) ** 2
                   + (players[:, :, 1] - ball[:, None, 1]) ** 2)
    for r in (2.0, 5.0):
        with np.errstate(invalid="ignore"):
            near = dist < r
        f["n_home_{:g}m".format(r)] = np.where(np.isnan(ball[:, 0]), np.nan,
                                              np.sum(near[:, home], axis=1))
        f["n_away_{:g}m".format(r)] = np.where(np.isnan(ball[:, 0]), np.nan,
                                              np.sum(near[:, away], axis=1))
    f["ball_speed"] = bspeed
    f["ball_x"], f["ball_y_edge"] = bx, np.fmin(by, width - by)
    f["ball_moved_1s"] = np.r_[np.full(fps, np.nan),
                               np.hypot(*(ball[fps:, :2] - ball[:-fps, :2]).T)]

    # The last thing known about the ball, held through tracking gaps.
    f["since_ball"] = np.fmin(frames_since_valid(ball[:, 0]), 3000)
    f["last_d_diff"] = forward_fill(f["d_diff"])
    f["last_ball_x"] = forward_fill(bx)
    f["outside_m"] = np.clip(outside_distance(ball, pitch), 0, 10)
    for w in (12, 50):
        f["d_diff_c{}".format(w)] = centred_mean(f["d_diff"], w)
    f["d_diff_past50"] = window_mean(f["d_diff"], 50, 0)
    f["d_diff_next50"] = window_mean(f["d_diff"], 0, 50)
    f["d_diff_past250"] = window_mean(f["d_diff"], 250, 0)

    # Collective shape and pace (works with no ball at all).
    seen = np.sum(~np.isnan(px), axis=1)
    f["n_players"] = seen.astype(np.float64)
    all_speed = np.where(seen > 0, np.nansum(pspeed, axis=1) / np.maximum(seen, 1), np.nan)
    for name, cols in (("home", home), ("away", away)):
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            f["cx_" + name] = np.nanmean(px[:, cols], axis=1)
            f["sx_" + name] = np.nanstd(px[:, cols], axis=1)
            f["sy_" + name] = np.nanstd(py[:, cols], axis=1)
            f["v_" + name] = centred_mean(np.nanmean(pspeed[:, cols], axis=1), 12)
    f["cx_diff"] = f["cx_home"] - f["cx_away"]
    for w in (12, 50, 125):
        f["v_all_c{}".format(w)] = centred_mean(all_speed, w)
    for w in (50, 250):
        f["v_all_past{}".format(w)] = window_mean(all_speed, w, 0)
        f["v_all_next{}".format(w)] = window_mean(all_speed, 0, w)
    f["cx_diff_c50"] = centred_mean(f["cx_diff"], 50)

    # Restart configurations (Vidal-Codina): somebody beyond a line, everybody in their own half.
    with np.errstate(invalid="ignore"):
        beyond_side = (py < -0.5) | (py > width + 0.5)
        beyond_goal = (px < -0.5) | (px > length + 0.5)
        f["n_beyond_side"] = np.sum(beyond_side, axis=1).astype(np.float64)
        f["n_beyond_goal"] = np.sum(beyond_goal, axis=1).astype(np.float64)
        own_half = np.where(home[None, :], px < length / 2 + 1, px > length / 2 - 1)
        f["frac_own_half"] = np.sum(own_half & ~np.isnan(px), axis=1) / np.maximum(seen, 1)
        centre = np.hypot(px - length / 2, py - width / 2) < 2.0
        f["n_centre"] = np.sum(centre, axis=1).astype(np.float64)
    for name in ("n_beyond_side", "n_beyond_goal", "frac_own_half"):
        f[name + "_c25"] = centred_mean(f[name], 25)

    # Speeds relative to the match's own median. Providers differ in how much they smooth, so
    # absolute m/s do not transfer from IDSSE to the challenge tracking; "slower than this match
    # usually is" does.
    in_play = period > 0
    for name in [k for k in f if k.startswith("v_") or k in ("ball_speed", "ball_moved_1s")]:
        f[name] = f[name] / np.nanmedian(f[name][in_play])

    names = list(f)
    return np.column_stack([f[k] for k in names]).astype(np.float32), names
