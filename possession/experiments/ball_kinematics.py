"""How the ball moves in the challenge against degraded IDSSE, and which ball blur matches it. ~5 min.

    python possession/experiments/ball_kinematics.py          # coarse sweep
    python possession/experiments/ball_kinematics.py fine     # smoothing only, around the coarse best

Live, visible ball only. Per frame: speed (central difference), change of velocity over +-3 frames
(as label_audit.py's touch detector), distance to the nearest player. Challenge live frames are
the ones v5 does not call dead: a prediction, not a label, used as the gap profile uses v2's.
IDSSE is degraded exactly as improve.py's caches (draw 0, uniform gaps, moving dead ball), then
train.blur_ball is swept. Input statistics only: no label of the challenge and no public window
enter, so the chosen blur is fitted to the tracker, never to the score.
"""
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import atd, idsse, train  # noqa: E402
import dose_exp  # noqa: E402
import improve  # noqa: E402

FPS = 25
KICK_HALF = 3
CLOSE_M = 1.5
SUB = 5                      # every 5th frame kept for the pooled quantiles
SMOOTH = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
NOISE = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
NOISE_FRAMES = (5, 25)
if os.environ.get("BALL_SWEEP") == "fine":       # read again by every worker process
    SMOOTH, NOISE = (5.0, 5.5, 6.0, 6.5, 7.0, 7.5), (0.0,)
LIVE_CACHE = os.path.join(improve.OUT, "challenge_live_v5.npy")
OUT = os.path.join(improve.OUT, "ball_kinematics{}.json".format(
    "_fine" if os.environ.get("BALL_SWEEP") == "fine" else ""))


def per_frame(players, ball, team):
    """(speed, dv, nearest distance), NaN where not measurable."""
    P = players[:, :, :2].astype(np.float64)
    B = ball[:, :2].astype(np.float64)
    d = np.hypot(P[:, :, 0] - B[:, None, 0], P[:, :, 1] - B[:, None, 1])
    close = np.where(np.all(np.isnan(d), axis=1), np.nan, np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1))
    v = np.full(B.shape, np.nan)
    v[1:-1] = (B[2:] - B[:-2]) * FPS / 2
    speed = np.hypot(*v.T)
    k = KICK_HALF
    dv = np.full(len(B), np.nan)
    dv[k:-k] = np.hypot(*(v[2 * k:] - v[:-2 * k]).T)
    return speed, dv, close


def sample(players, ball, team, live):
    speed, dv, close = per_frame(players, ball, team)
    keep = live & np.isfinite(speed) & np.isfinite(dv) & np.isfinite(close)
    idx = np.flatnonzero(keep)[::SUB]
    return np.stack([speed[idx], dv[idx], close[idx]]).astype(np.float32)


def summary(rows):
    s = np.concatenate(rows, axis=1)
    return dict(speed50=float(np.percentile(s[0], 50)), speed90=float(np.percentile(s[0], 90)),
                dv50=float(np.percentile(s[1], 50)), dv90=float(np.percentile(s[1], 90)),
                dv99=float(np.percentile(s[1], 99)), close=float(np.mean(s[2] <= CLOSE_M)), n=int(s.shape[1]))


def challenge():
    players, ball, team = atd.load()
    if not os.path.isfile(LIVE_CACHE):
        from possession import pipeline
        labels = pipeline.predict_v5()
        np.save(LIVE_CACHE, np.isin(labels, ("H", "A")))
    live = np.load(LIVE_CACHE)
    return sample(players, ball, team, live)


def configs():
    out = [(0.0, 0.0, 25)]
    for s, m, f in itertools.product(SMOOTH, NOISE, NOISE_FRAMES):
        if (s, m) != (0.0, 0.0) and not (m == 0.0 and f != 25):
            out.append((s, m, f))
    return out


def idsse_job(args):
    index, match_id, lengths, share = args
    data = idsse.load(match_id)
    y = np.array([train.CODE[v] for v in data["labels"]], dtype=np.int8)
    team = data["team"]
    live = y < 2
    streams = np.random.default_rng([train.SEED, index]).spawn(3)
    clean = sample(data["players"], data["ball"], team, live)
    data = dict(data, ball=train.moving_dead_ball(data["players"], data["ball"], y, streams[2]))
    players, ball = train.degrade(data, streams[0], lengths, 0.0)
    hidden = train.ball_mask(len(y), streams[1], lengths, share)
    out = {"clean": clean}
    for c in configs():
        blurred = train.blur_ball(ball, np.random.default_rng([train.SEED, index, 99]), *c)
        blurred[hidden] = np.nan
        out[c] = sample(players, blurred, team, live)
    return out


def loss(stats, target):
    keys = ("speed50", "speed90", "dv50", "dv90", "dv99", "close")
    return float(np.mean([np.log(stats[k] / target[k]) ** 2 for k in keys]))


def main():
    lengths, share, real = dose_exp.gap_profile()
    assert real, "needs the challenge gap profile"
    target = summary([challenge()])
    with ProcessPoolExecutor(7) as pool:
        done = list(pool.map(idsse_job, [(i, m, lengths, share) for i, m in enumerate(idsse.MATCHES)]))
    rows = {"clean": summary([d["clean"] for d in done])}
    for c in configs():
        rows[c] = summary([d[c] for d in done])
        rows[c]["loss"] = loss(rows[c], target)
    fmt = "   {:<26} {:6.1f} {:6.1f}   {:5.1f} {:6.1f} {:6.1f}   {:5.0%}   {}"
    print("   balon vivo y visible       vel p50   p90    dv p50   p90    p99   <=1,5 m   error")
    print(fmt.format("reto", target["speed50"], target["speed90"], target["dv50"], target["dv90"],
                     target["dv99"], target["close"], ""))
    for name, key in (("IDSSE limpio", "clean"), ("IDSSE degradado (hoy)", (0.0, 0.0, 25))):
        r = rows[key]
        print(fmt.format(name, r["speed50"], r["speed90"], r["dv50"], r["dv90"], r["dv99"], r["close"],
                         "{:.3f}".format(r["loss"]) if "loss" in r else ""))
    print("\n   mejores 12 (sigma frames, ruido m, correlacion frames)")
    ranked = sorted((c for c in configs()), key=lambda c: rows[c]["loss"])
    for c in ranked[:12]:
        r = rows[c]
        print(fmt.format("s {:.1f}  m {:.2f}  f {}".format(*c), r["speed50"], r["speed90"], r["dv50"], r["dv90"],
                         r["dv99"], r["close"], "{:.3f}".format(r["loss"])))
    with open(OUT, "w") as handle:
        json.dump(dict(target=target, rows={str(k): v for k, v in rows.items()}), handle, indent=1)
    print("\n  ", OUT)


if __name__ == "__main__":
    if sys.argv[1:] == ["fine"] and os.environ.get("BALL_SWEEP") != "fine":
        import subprocess
        sys.exit(subprocess.call([sys.executable, __file__], env=dict(os.environ, BALL_SWEEP="fine")))
    main()
