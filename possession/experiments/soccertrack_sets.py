"""SoccerTrack v2 as encoder training sets, in skillcorner_data.py's format.

    LIVE_DEAD=1 python possession/experiments/soccertrack_sets.py

Two kinds per match and gap pattern, fixed before any training score was seen:
  soccertrack_<m>_0_<gap>.npz   ball hidden on every frame (arm ST): the reconstructed ball, built
                                from the events that also give the labels, never reaches the model
  soccertrackb_<m>_0_<gap>.npz  the reconstructed ball, moving dead ball, the challenge's gaps (STb)
Players as in soccertrack_data (smoothed off the 1 m grid), then degraded like every other set.
Labels: y_raw from the provider's ballStatus, y_ctrl with the control rule. External data only.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import train  # noqa: E402
import dose_exp  # noqa: E402
import encoder_data  # noqa: E402
import improve  # noqa: E402
import soccertrack_data as st  # noqa: E402

KINDS = ("soccertrack", "soccertrackb")


def job(args):
    match, kind, gaps, lengths, share = args
    path = os.path.join(encoder_data.FOLDER, "{}_{}_0_{}.npz".format(kind, match, gaps))
    if os.path.isfile(path):
        return path, None
    d = st.load(match)
    y, y_ctrl, period, team = d["y"].astype(np.int8), d["y_ctrl"].astype(np.int8), d["period"].astype(np.int64), d["team"]
    index = st.MATCHES.index(match)
    data = dict(players=d["players"], ball=d["ball"].copy(), team=team, period=period, pitch=np.array(st.PITCH), y=y)
    if improve.LIVE_DEAD:
        data["ball"] = train.moving_dead_ball(data["players"], data["ball"], y,
                                              np.random.default_rng([train.SEED, 800 + index, 0, 1]))
    streams = np.random.default_rng([train.SEED, 800 + index, 0]).spawn(2)
    players, ball = train.degrade(data, streams[0], lengths, 0.0, ball_smooth_frames=improve.BLUR_FRAMES)
    if kind == "soccertrack":
        ball[:] = np.nan
    elif gaps == "dead":
        ball[train.dead_mask(y, streams[1], lengths, train.GAP_GIVEN_ALIVE)] = np.nan
    else:
        ball[train.ball_mask(len(y), streams[1], lengths, share)] = np.nan
    steps = encoder_data.step_frames(period)
    p, b = encoder_data.tensors(players, ball, team, period, st.PITCH, steps)
    kick = np.stack([encoder_data.kick_steps(players, ball, s, steps) for s in (team, 1 - team)]).astype(np.float16)
    np.savez(path, players=p, ball=b, kick=kick, steps=steps, period=period, y_raw=y, y_ctrl=y_ctrl)
    return path, dict(minutes=len(y) / 25 / 60, dead=float((y == 2).mean()), ball_seen=float(np.isfinite(ball[:, 0]).mean()))


def main():
    os.makedirs(encoder_data.FOLDER, exist_ok=True)
    lengths, share, real = dose_exp.gap_profile()
    assert real
    jobs = [(m, k, g, lengths, share) for k in KINDS for m in st.MATCHES for g in improve.GAPS]
    with ProcessPoolExecutor(4) as pool:
        for path, stats in pool.map(job, jobs):
            print(os.path.basename(path), "" if stats is None else
                  " ".join("{} {:.2f}".format(k, v) for k, v in stats.items()), flush=True)


if __name__ == "__main__":
    main()
