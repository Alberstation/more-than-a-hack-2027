"""Inputs for the plan 3j screen arms, next to sets/ (improve_live). LIVE_DEAD=1. CPU, run once.

    LIVE_DEAD=1 python possession/experiments/screen_data.py

sets_aux/<set>.npz, one per existing set file (same steps), from the CLEAN tracking:
  poss   (2, T) int16  slot in that view's (2 x SLOTS) grid of the player of the possessing team
                       (control labels) nearest the clean ball, if within 3 m; -1 otherwise
  state  (T,)   int8   0 control (one team within 1.5 m of the ball), 1 contest (both), 2 loose /
                       in flight (nobody), 3 dead
sets_clean/<match>.npz and metrica_g<game>.npz: tensors of the clean tracking (moving dead ball
kept, since it is a simulation of the tracker's dead ball, not an error), same steps.
sets_rand/: the IDSSE and Metrica set files again under a random degradation per file
(stress_random.RANGES, seeded by file), for arm R. Targets only from external data.
"""
import os
import zlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import idsse, train  # noqa: E402
import dose_exp  # noqa: E402
import encoder_data  # noqa: E402
import improve  # noqa: E402
import live_review  # noqa: E402
import metrica_exp  # noqa: E402
import stress_random  # noqa: E402

SETS = encoder_data.FOLDER
AUX, CLEAN, RAND = (os.path.join(improve.OUT, f) for f in ("sets_aux", "sets_clean", "sets_rand"))
SLOTS = encoder_data.SLOTS


def clean_idsse(m):
    data = idsse.load(m)
    y = np.array([train.CODE[v] for v in data["labels"]], dtype=np.int8)
    index = idsse.MATCHES.index(m)
    streams = np.random.default_rng([train.SEED, index]).spawn(3)
    data = dict(data, ball=train.moving_dead_ball(data["players"], data["ball"], y, streams[2]))
    return data, tuple(data["pitch"])


def clean_metrica(game):
    data = metrica_exp.load_game(game)
    moved = np.random.default_rng([train.SEED, game, 0, 1])
    data = dict(data, ball=train.moving_dead_ball(data["players"], data["ball"], data["y"], moved))
    return data, tuple(data["pitch"])


def targets(players, ball, team, steps, y_ctrl):
    P = players[steps, :, :2].astype(np.float64)
    B = ball[steps, :2].astype(np.float64)
    d = np.hypot(P[..., 0] - B[:, None, 0], P[..., 1] - B[:, None, 1])
    y = y_ctrl[steps]
    poss = np.full((2, len(steps)), -1, dtype=np.int16)
    for view, side in enumerate((team, 1 - team)):
        for t in (0, 1):
            cols = np.flatnonzero(side == t)[:SLOTS]
            own = (y == t) if view == 0 else (np.where(y == 2, 2, 1 - y) == t)
            dd = np.where(np.isnan(d[:, cols]), np.inf, d[:, cols])
            k = np.argmin(dd, axis=1)
            ok = own & (dd[np.arange(len(k)), k] <= 3.0)
            poss[view, ok] = t * SLOTS + k[ok]
    near = [np.nanmin(np.where(np.isnan(d[:, team == t]), np.inf, d[:, team == t]), axis=1) <= 1.5 for t in (0, 1)]
    state = np.where(near[0] & near[1], 1, np.where(near[0] | near[1], 0, 2)).astype(np.int8)
    state[y == 2] = 3
    return poss, state


def aux_job(name):
    out = os.path.join(AUX, name)
    if os.path.isfile(out):
        return name
    with np.load(os.path.join(SETS, name)) as f:
        steps, y_ctrl = f["steps"], f["y_ctrl"]
    if name.startswith("metrica"):
        data, pitch = clean_metrica(int(name.split("_")[1][1:]))
    else:
        data, pitch = clean_idsse(name.split("_")[0])
    poss, state = targets(data["players"], data["ball"], data["team"], steps, y_ctrl)
    np.savez(out, poss=poss, state=state)
    return name


def clean_job(key):
    out = os.path.join(CLEAN, "{}.npz".format(key))
    if os.path.isfile(out):
        return key
    data, pitch = clean_metrica(int(key[9:])) if key.startswith("metrica") else clean_idsse(key)
    steps = encoder_data.step_frames(data["period"])
    p, b = encoder_data.tensors(data["players"], data["ball"], data["team"], data["period"], pitch, steps)
    np.savez(out, players=p, ball=b, steps=steps)
    return key


def rand_job(args):
    kind, key, gaps, draw, lengths, share = args
    name = ("metrica_g{}_{}_{}.npz".format(key, draw, gaps) if kind == "metrica"
            else "{}_{}_d{}.npz".format(key, gaps, draw))
    out = os.path.join(RAND, name)
    if os.path.isfile(out):
        return name
    rng = np.random.default_rng([train.SEED, 91, zlib.crc32(name.encode())])
    p = {k: float(rng.uniform(*v)) for k, v in stress_random.RANGES.items()}
    train.GAP_GIVEN_DEAD, train.PLAYER_DROP_SHARE, train.POSITION_NOISE_M = p["dead"], p["drop"], p["noise"]
    lengths = np.maximum(1, np.round(np.asarray(lengths) * p["length"])).astype(int)
    if kind == "metrica":
        data, pitch = clean_metrica(key)
        y = data["y"]
        y_ctrl = live_review.metrica_control(y, *live_review.metrica_events(key))
    else:
        data, pitch = clean_idsse(key)
        y = np.array([train.CODE[v] for v in data["labels"]], dtype=np.int8)
        with np.load(improve.cache_path(key, gaps, draw)) as f:
            y_ctrl = f["y_ctrl"]
    streams = rng.spawn(2)
    players, ball = train.degrade(data, streams[0], lengths, 0.0, ball_smooth_frames=p["blur"])
    if gaps == "dead":
        hidden = train.dead_mask(y, streams[1], lengths, p["alive"])
    else:
        hidden = train.ball_mask(len(y), streams[1], lengths, share * p["dead"] / 0.75)
    ball[hidden] = np.nan
    steps = encoder_data.step_frames(data["period"])
    pp, b = encoder_data.tensors(players, ball, data["team"], data["period"], pitch, steps)
    np.savez(out, players=pp, ball=b, steps=steps, period=data["period"], y_raw=y, y_ctrl=y_ctrl)
    return name


def main():
    for folder in (AUX, CLEAN, RAND):
        os.makedirs(folder, exist_ok=True)
    names = sorted(n for n in os.listdir(SETS) if n.endswith(".npz") and ("uniform" in n or "dead" in n))
    lengths, share, real = dose_exp.gap_profile()
    assert real
    with ProcessPoolExecutor(4) as pool:
        for k in pool.map(clean_job, list(idsse.MATCHES) + ["metrica_g{}".format(g) for g in metrica_exp.ALL_GAMES]):
            print("clean", k, flush=True)
        for n in pool.map(aux_job, names):
            pass
        print("aux", len(names), flush=True)
        jobs = [("idsse", m, g, d, lengths, share) for m in idsse.MATCHES for g in improve.GAPS for d in range(3)]
        jobs += [("metrica", x, g, r, lengths, share) for x in metrica_exp.ALL_GAMES for g in improve.GAPS
                 for r in range(metrica_exp.REPEATS)]
        for n in pool.map(rand_job, jobs):
            print("rand", n, flush=True)


if __name__ == "__main__":
    main()
