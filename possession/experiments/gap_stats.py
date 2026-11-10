"""Where the ball goes missing: challenge tracker against the degraded IDSSE. LIVE_DEAD=1. ~3 min.

    LIVE_DEAD=1 python possession/experiments/gap_stats.py

Input statistics only, no label of the challenge: live there is what v5 does not call dead (as in
ball_kinematics.py). At every live frame with the ball seen, the hazard that the ball is gone on
the next frame, split by the nearest player's distance (occlusion) and by the distance to the
nearest pitch line (ball leaving the camera's best view); the speed of the ball when it is lost;
and the share of seen balls outside the lines. IDSSE as improve.py's caches, draw 0, both gap
patterns. Nothing is fitted here; it says whether the gaps start where the challenge's start.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAMES, half_video_blocks  # noqa: E402
from possession import atd, idsse, train  # noqa: E402
import ball_kinematics  # noqa: E402
import dose_exp  # noqa: E402
import improve  # noqa: E402

FPS = 25
NEAR_EDGES = (0, 1, 2, 4, 1e9)
LINE_EDGES = (-1e9, 0, 3, 10, 1e9)
SPEED_EDGES = (0, 2, 5, 10, 1e9)


def stats(players, ball, live, pitch):
    P, B = players[:, :, :2].astype(np.float64), ball[:, :2].astype(np.float64)
    seen = np.isfinite(B[:, 0])
    d = np.hypot(P[:, :, 0] - B[:, None, 0], P[:, :, 1] - B[:, None, 1])
    near = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    line = np.minimum.reduce([B[:, 0], pitch[0] - B[:, 0], B[:, 1], pitch[1] - B[:, 1]])
    v = np.full(len(B), np.nan)
    v[1:] = np.hypot(*(B[1:] - B[:-1]).T) * FPS
    at = np.flatnonzero(seen[:-1] & live[:-1] & live[1:] & np.isfinite(near[:-1]))
    lost = ~seen[at + 1]
    out = dict(missing_live=float(np.mean(~seen[live])), lost_per_min=float(lost.sum() / (live.sum() / FPS / 60)),
               outside=float(np.mean(line[seen & live] < 0)))
    for name, x, edges in (("near", near[at], NEAR_EDGES), ("line", line[at], LINE_EDGES),
                           ("speed", np.nan_to_num(v[at], nan=0.0), SPEED_EDGES)):
        out[name] = [float(np.mean(lost[(x >= a) & (x < b)])) * 1000 for a, b in zip(edges[:-1], edges[1:])]
        out[name + "_share"] = [float(np.mean((x >= a) & (x < b))) for a, b in zip(edges[:-1], edges[1:])]
    out["speed_lost"] = float(np.nanmedian(v[at[lost]]))
    out["speed_kept"] = float(np.nanmedian(v[at[~lost]]))
    out["near_lost"] = float(np.median(near[at[lost]]))
    return out


def challenge():
    players, ball, _ = atd.load()
    live = np.load(ball_kinematics.LIVE_CACHE)
    graded = np.zeros(len(ball), dtype=bool)
    for _, first, last in half_video_blocks("BAR-ATH"):
        graded[first:last + 1] = True
    return stats(players, ball, live & graded, np.array(GAMES["BAR-ATH"]["pitch"]))


def idsse_job(args):
    index, match_id, gaps, lengths, share = args
    data = idsse.load(match_id)
    y = np.array([train.CODE[v] for v in data["labels"]], dtype=np.int8)
    streams = np.random.default_rng([train.SEED, index]).spawn(3)
    data = dict(data, ball=train.moving_dead_ball(data["players"], data["ball"], y, streams[2]))
    players, ball = train.degrade(data, streams[0], lengths, 0.0)
    if gaps == "dead":
        hidden = train.dead_mask(y, streams[1], lengths, train.GAP_GIVEN_ALIVE)
    else:
        hidden = train.ball_mask(len(y), streams[1], lengths, share)
    ball[hidden] = np.nan
    return gaps, stats(players, ball, y < 2, np.array(data["pitch"]))


def main():
    lengths, share, real = dose_exp.gap_profile()
    assert real
    rows = {"reto": challenge()}
    with ProcessPoolExecutor(4) as pool:
        done = list(pool.map(idsse_job, [(i, m, g, lengths, share) for i, m in enumerate(idsse.MATCHES)
                                         for g in improve.GAPS]))
    for g in improve.GAPS:
        part = [s for gg, s in done if gg == g]
        rows["IDSSE " + g] = {k: (np.mean([p[k] for p in part], axis=0).tolist() if isinstance(part[0][k], list)
                                  else float(np.mean([p[k] for p in part]))) for k in part[0]}
    print("pérdida del balón por 1000 frames vivos con balón, según el jugador más cercano (0-1, 1-2, 2-4, > 4 m)")
    print("según la distancia a la línea (fuera, 0-3, 3-10, > 10 m) y la velocidad del balón (0-2, 2-5, 5-10, > 10 m/s)")
    for name, r in rows.items():
        print("  {:<14} falta vivo {:.1%}  pérdidas/min {:.1f}  fuera {:.1%}  cercano al perder {:.2f} m  "
              "vel perder/seguir {:.1f}/{:.1f}".format(name, r["missing_live"], r["lost_per_min"], r["outside"],
                                                      r["near_lost"], r["speed_lost"], r["speed_kept"]))
        print("     cercano: " + "  ".join("{:5.1f} ({:.0%})".format(h, s) for h, s in zip(r["near"], r["near_share"])))
        print("     linea:   " + "  ".join("{:5.1f} ({:.0%})".format(h, s) for h, s in zip(r["line"], r["line_share"])))
        print("     veloc.:  " + "  ".join("{:5.1f} ({:.0%})".format(h, s) for h, s in zip(r["speed"], r["speed_share"])))


if __name__ == "__main__":
    main()
