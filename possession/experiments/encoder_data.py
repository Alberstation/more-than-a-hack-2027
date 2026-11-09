"""Per-player tensors for the set encoder, from the same degraded tracking as the v4 caches. LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/encoder_data.py

One file per IDSSE match, gap pattern and draw, and per Metrica game, repeat and gap pattern, in
improve_live/sets/. Steps are every 5th frame of each period. Both views are stored: view 1 swaps
the teams, so the encoder is trained symmetric as v4 is. Per player, in the home attacking frame:
x, y, vx, vy, offset to the ball, distance to the ball, seen. Ball: x, y, vx, vy, seen, time since seen.
Kick (features_kick.py, per view, over the 5 frames of each step): largest velocity change over
+-3/6/12 frames, a strike in the step, last and next striking side (+1 home, -1 away, 0 unknown),
margin of the last strike, time since and to a strike, home share of strikes, change of direction.
Read only by the encoder's `kick` variant.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import features, features_kick, idsse, train  # noqa: E402
import dose_exp  # noqa: E402
import improve  # noqa: E402
import metrica_exp  # noqa: E402

FOLDER = os.path.join(improve.OUT, "sets")
STEP = 5
SLOTS = 20                  # per team; IDSSE lists 20 with substitutes, Metrica 14


def velocity(xy, period, fps=25):
    out = np.full(xy.shape, np.nan)
    for p in np.unique(period):
        r = np.flatnonzero(period == p)
        if len(r) > 4:
            out[r[2:-2]] = (xy[r[4:]] - xy[r[:-4]]) * fps / 4
    return out


def tensors(players, ball, team, period, pitch, steps):
    length, width = pitch
    P = players[:, :, :2].astype(np.float64)
    B = ball[:, :2].astype(np.float64)
    vP, vB = velocity(P, period), velocity(B, period)
    since = np.fmin(features.frames_since_valid(B[:, 0]), 250) / 250
    out_p, out_b = [], []
    for side in (team, 1 - team):
        right = features.home_attacks_right(players, side, period)[steps]
        sign = np.where(right, 1.0, -1.0)
        px = np.where(right[:, None], P[steps, :, 0], length - P[steps, :, 0])
        py = P[steps, :, 1]
        bx = np.where(right, B[steps, 0], length - B[steps, 0])
        by = B[steps, 1]
        dx, dy = px - bx[:, None], py - by[:, None]
        f = np.stack([px / length, py / width - 0.5, vP[steps, :, 0] * sign[:, None] / 10, vP[steps, :, 1] / 10,
                      dx / 20, dy / 20, np.hypot(dx, dy) / 20], axis=-1)
        seen = ~np.isnan(px)
        f = np.concatenate([np.nan_to_num(f), seen[..., None]], axis=-1)
        grid = np.zeros((len(steps), 2, SLOTS, f.shape[-1]))
        for t in (0, 1):
            cols = np.flatnonzero(side == t)[:SLOTS]
            grid[:, t, :len(cols)] = f[:, cols]
        ball_seen = ~np.isnan(bx)
        b = np.stack([bx / length, by / width - 0.5, vB[steps, 0] * sign / 10, vB[steps, 1] / 10], axis=-1)
        b = np.concatenate([np.nan_to_num(b), ball_seen[:, None], since[steps, None]], axis=-1)
        out_p.append(grid)
        out_b.append(b)
    return np.stack(out_p).astype(np.float16), np.stack(out_b).astype(np.float16)


def kick_steps(players, ball, side, steps):
    """(n_steps, 13) kick channels for one view, NaN-free."""
    X, names = features_kick.kick_features(players, ball, side)
    c = {k: X[:, i].astype(np.float64) for i, k in enumerate(names)}
    window = lambda v, fill: np.max(np.stack([np.nan_to_num(v, nan=fill)[np.clip(steps + o, 0, len(v) - 1)]
                                              for o in range(-2, 3)]), axis=0)
    signed = lambda v: np.where(np.isnan(v), 0.0, 1.0 - 2.0 * v)       # 0 home -> +1, 1 away -> -1
    cos = c["kick_cos6"][steps]
    out = [window(c["kick_dv3"], 0) / 10, window(c["kick_dv6"], 0) / 10, window(c["kick_dv12"], 0) / 10,
           window(np.where(c["strike_age"] == 0, 1.0, 0.0), 0),
           signed(c["kick_side_last"][steps]), signed(c["kick_side_next"][steps]),
           np.nan_to_num(c["kick_margin_last"][steps]) / 10,
           np.nan_to_num(c["kick_age"][steps], nan=500) / 500, np.nan_to_num(c["kick_to_next"][steps], nan=250) / 250,
           np.nan_to_num(c["kick_home_share_c25"][steps] - 0.5) * 2, np.nan_to_num(c["kick_home_share_c50"][steps] - 0.5) * 2,
           np.nan_to_num(cos), np.isfinite(cos).astype(np.float64)]
    return np.stack(out, axis=-1)


def step_frames(period):
    return np.concatenate([np.flatnonzero(period == p)[::STEP] for p in np.unique(period)])


def idsse_job(args):
    index, match_id, gaps, lengths, share, draw = args
    path = os.path.join(FOLDER, "{}_{}_d{}.npz".format(match_id, gaps, draw))
    if os.path.isfile(path):
        return path
    data = idsse.load(match_id)
    y = np.array([train.CODE[v] for v in data["labels"]], dtype=np.int8)
    period, pitch, team = data["period"], tuple(data["pitch"]), data["team"]
    seed = [train.SEED, index, draw] if draw else [train.SEED, index]
    streams = np.random.default_rng(seed).spawn(3)
    if improve.LIVE_DEAD:
        data = dict(data, ball=train.moving_dead_ball(data["players"], data["ball"], y, streams[2]))
    players, ball = train.degrade(data, streams[0], lengths, 0.0, ball_smooth_frames=improve.BLUR_FRAMES)
    if gaps == "dead":
        hidden = train.dead_mask(y, streams[1], lengths, train.GAP_GIVEN_ALIVE)
    else:
        hidden = train.ball_mask(len(y), streams[1], lengths, share)
    ball[hidden] = np.nan
    steps = step_frames(period)
    p, b = tensors(players, ball, team, period, pitch, steps)
    kick = np.stack([kick_steps(players, ball, side, steps) for side in (team, 1 - team)]).astype(np.float16)
    with np.load(improve.cache_path(match_id, gaps, draw)) as f:
        y_ctrl = f["y_ctrl"]
    np.savez(path, players=p, ball=b, kick=kick, steps=steps, period=period, y_raw=y, y_ctrl=y_ctrl)
    return path


def metrica_job(args):
    game, repeat, gaps, lengths, share = args
    path = os.path.join(FOLDER, "metrica_g{}_{}_{}.npz".format(game, repeat, gaps))
    if os.path.isfile(path):
        return path
    data = metrica_exp.load_game(game)
    y, period, team = data["y"], data["period"], data["team"]
    if improve.LIVE_DEAD:
        moved = np.random.default_rng([train.SEED, game, repeat, 1])
        data = dict(data, ball=train.moving_dead_ball(data["players"], data["ball"], y, moved))
    streams = np.random.default_rng([train.SEED, game, repeat]).spawn(2)
    players, ball = train.degrade(data, streams[0], lengths, 0.0, ball_smooth_frames=improve.BLUR_FRAMES)
    if gaps == "dead":
        hidden = train.dead_mask(y, streams[1], lengths, train.GAP_GIVEN_ALIVE)
    else:
        hidden = train.ball_mask(len(y), streams[1], lengths, share)
    ball[hidden] = np.nan
    steps = step_frames(period)
    p, b = tensors(players, ball, team, period, tuple(data["pitch"]), steps)
    kick = np.stack([kick_steps(players, ball, side, steps) for side in (team, 1 - team)]).astype(np.float16)
    np.savez(path, players=p, ball=b, kick=kick, steps=steps, period=period, y_raw=y)
    return path


if __name__ == "__main__":
    os.makedirs(FOLDER, exist_ok=True)
    lengths, share, real = dose_exp.gap_profile()
    assert real
    jobs = [(i, m, g, lengths, share, d) for d in range(3) for i, m in enumerate(idsse.MATCHES) for g in improve.GAPS]
    with ProcessPoolExecutor(3) as pool:
        for path in pool.map(idsse_job, jobs):
            print("ready", os.path.basename(path), flush=True)
    jobs = [(g, r, gaps, lengths, share) for g in metrica_exp.GAMES for r in range(metrica_exp.REPEATS)
            for gaps in improve.GAPS]
    with ProcessPoolExecutor(3) as pool:
        for path in pool.map(metrica_job, jobs):
            print("ready", os.path.basename(path), flush=True)
