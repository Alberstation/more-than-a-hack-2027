"""Plan 3j idea 4: SkillCorner Open Data (MIT, 20 A-League matches, 10 fps) as encoder training sets.

    LIVE_DEAD=1 python possession/experiments/skillcorner_data.py

Broadcast tracking: whole stretches without any player (replays) are cut out, and each continuous
stretch of at least MIN_SEGMENT_S becomes its own period. Upsampled to 25 fps (linear positions).
Labels: team of the last `player_possession` event inside a phase of play (else the phase's team),
dead outside every phase. `ctrl` rule as for Metrica, with the possession starts as own events.
Writes sets/skillcorner_<match>_0_<gap>.npz like encoder_data.metrica_job, one repeat per match.
External data only.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import train  # noqa: E402
import dose_exp  # noqa: E402
import encoder_data  # noqa: E402
import improve  # noqa: E402
import live_review  # noqa: E402

DATA = os.path.join(ROOT, "data", "external", "skillcorner", "data", "matches")
MATCHES = sorted(os.listdir(DATA))
PITCH = (105.0, 68.0)
SRC_FPS, UP = 10, 2.5
MIN_SEGMENT_S = 30


def read(match):
    folder = os.path.join(DATA, match)
    meta = json.load(open(os.path.join(folder, match + "_match.json"), encoding="utf-8"))
    home = meta["home_team"]["id"]
    sx, sy = PITCH[0] / meta["pitch_length"], PITCH[1] / meta["pitch_width"]
    ids = [p["id"] for p in meta["players"]]
    col = {i: k for k, i in enumerate(ids)}
    team = np.array([0 if p["team_id"] == home else 1 for p in meta["players"]], dtype=np.int8)
    frames, period, ball, players = [], [], [], []
    with open(os.path.join(folder, match + "_tracking_extrapolated.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            d = json.loads(line)
            if not d["player_data"] or d["period"] is None or d["ball_data"]["x"] is None:
                continue
            row = np.full((len(ids), 2), np.nan, dtype=np.float32)
            for p in d["player_data"]:
                if p["player_id"] in col:
                    row[col[p["player_id"]]] = (p["x"] * sx + PITCH[0] / 2, p["y"] * sy + PITCH[1] / 2)
            frames.append(d["frame"])
            period.append(d["period"])
            ball.append((d["ball_data"]["x"] * sx + PITCH[0] / 2, d["ball_data"]["y"] * sy + PITCH[1] / 2))
            players.append(row)
    frames, period = np.array(frames), np.array(period)
    ball, players = np.array(ball, dtype=np.float32), np.stack(players)
    # labels on the source frames
    phases = pd.read_csv(os.path.join(folder, match + "_phases_of_play.csv"))
    events = pd.read_csv(os.path.join(folder, match + "_dynamic_events.csv"), low_memory=False)
    poss = events[events.event_type == "player_possession"].sort_values("frame_start")
    last = int(frames.max()) + 1
    y_all = np.full(last + 1, 2, dtype=np.int8)
    for row in phases.itertuples():
        y_all[row.frame_start:row.frame_end + 1] = 0 if row.team_in_possession_id == home else 1
    owner = np.full(last + 1, -1, dtype=np.int8)
    starts = poss.frame_start.to_numpy().clip(0, last)
    owner[starts] = np.where(poss.team_id.to_numpy() == home, 0, 1)
    known = np.maximum.accumulate(np.where(owner >= 0, np.arange(last + 1), 0))
    filled = owner[known]
    inside = (y_all < 2) & (filled >= 0)
    # the last possession only counts inside the same phase
    phase_id = np.full(last + 1, -1)
    for k, row in enumerate(phases.itertuples()):
        phase_id[row.frame_start:row.frame_end + 1] = k
    inside &= phase_id[known] == phase_id
    y_all[inside] = filled[inside]
    owners = np.where(poss.team_id.to_numpy() == home, 0, 1)
    return dict(frames=frames, period=period, ball=ball, players=players, team=team, y=y_all[frames],
                event_frames=starts, owners=owners)


def upsample(src):
    """Continuous stretches -> 25 fps arrays, one period per stretch."""
    f = src["frames"]
    cut = np.flatnonzero((np.diff(f) != 1) | (np.diff(src["period"]) != 0)) + 1
    bounds = np.split(np.arange(len(f)), cut)
    out = dict(players=[], ball=[], y=[], period=[], src=[])
    pid = 0
    for idx in bounds:
        if len(idx) < MIN_SEGMENT_S * SRC_FPS:
            continue
        pid += 1
        t = np.arange(0, len(idx) - 1 + 1e-9, 1 / UP)
        lo = np.floor(t).astype(int).clip(0, len(idx) - 1)
        hi = np.minimum(lo + 1, len(idx) - 1)
        w = (t - lo)[:, None]
        P, B = src["players"][idx], src["ball"][idx]
        out["players"].append((P[lo] * (1 - w[..., None]) + P[hi] * w[..., None]).astype(np.float32))
        out["ball"].append((B[lo] * (1 - w) + B[hi] * w).astype(np.float32))
        out["y"].append(src["y"][idx][np.rint(t).astype(int).clip(0, len(idx) - 1)])
        out["period"].append(np.full(len(t), pid, dtype=np.int64))
        out["src"].append(f[idx][np.rint(t).astype(int).clip(0, len(idx) - 1)])
    return {k: np.concatenate(v) for k, v in out.items()}


def job(args):
    match, gaps, lengths, share = args
    path = os.path.join(encoder_data.FOLDER, "skillcorner_{}_0_{}.npz".format(match, gaps))
    if os.path.isfile(path):
        return path, None
    src = read(match)
    up = upsample(src)
    y, period, team = up["y"].astype(np.int8), up["period"], src["team"]
    # ctrl rule on the 25 fps frames: events mapped to the nearest upsampled frame
    pos = np.searchsorted(up["src"], src["event_frames"]).clip(0, len(y) - 1)
    ok = np.abs(up["src"][pos] - src["event_frames"]) <= 1
    y_ctrl = live_review.metrica_control(y, pos[ok], src["owners"][ok])
    index = MATCHES.index(match)
    data = dict(players=up["players"], ball=up["ball"], team=team, period=period, pitch=np.array(PITCH), y=y)
    if improve.LIVE_DEAD:
        data = dict(data, ball=train.moving_dead_ball(data["players"], data["ball"], y,
                                                      np.random.default_rng([train.SEED, 300 + index, 0, 1])))
    streams = np.random.default_rng([train.SEED, 300 + index, 0]).spawn(2)
    players, ball = train.degrade(data, streams[0], lengths, 0.0, ball_smooth_frames=improve.BLUR_FRAMES)
    if gaps == "dead":
        hidden = train.dead_mask(y, streams[1], lengths, train.GAP_GIVEN_ALIVE)
    else:
        hidden = train.ball_mask(len(y), streams[1], lengths, share)
    ball[hidden] = np.nan
    steps = encoder_data.step_frames(period)
    p, b = encoder_data.tensors(players, ball, team, period, PITCH, steps)
    kick = np.stack([encoder_data.kick_steps(players, ball, s, steps) for s in (team, 1 - team)]).astype(np.float16)
    np.savez(path, players=p, ball=b, kick=kick, steps=steps, period=period, y_raw=y, y_ctrl=y_ctrl)
    stats = dict(minutes=len(y) / 25 / 60, periods=int(period.max()), dead=float((y == 2).mean()),
                 home=float((y == 0).mean()), ctrl_changed=float((y_ctrl != y).mean()))
    return path, stats


def main():
    os.makedirs(encoder_data.FOLDER, exist_ok=True)
    lengths, share, real = dose_exp.gap_profile()
    assert real
    jobs = [(m, g, lengths, share) for m in MATCHES for g in improve.GAPS]
    with ProcessPoolExecutor(4) as pool:
        for path, stats in pool.map(job, jobs):
            print(os.path.basename(path), "" if stats is None else
                  " ".join("{} {:.2f}".format(k, v) for k, v in stats.items()), flush=True)


if __name__ == "__main__":
    main()
