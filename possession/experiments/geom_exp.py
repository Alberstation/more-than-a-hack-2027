"""H-A geometry batch on top of v4, judged as one batch. Run with LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/geom_exp.py prep    # add the block to every cache
    LIVE_DEAD=1 python possession/experiments/geom_exp.py train   # out-of-fold geom3
    LIVE_DEAD=1 python possession/experiments/geom_exp.py score   # paired with mixed_all3, ctrl only

geom: soft minimum of each side's distances to the ball, projected arrival, pressure on the
nearest player, Voronoi share (whole pitch, thirds, near the ball) and a 6 x 4 occupancy grid.
The caches are extended by regenerating the same degraded tracking from the same seeds; the base
block is recomputed and must match the cached one bit for bit, or prep stops.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import features, idsse, train  # noqa: E402
import dose_exp  # noqa: E402
import improve  # noqa: E402
import live_review  # noqa: E402
import metrica_exp  # noqa: E402

ARM = "ctrl_geom3"
REFS = ("ctrl_mixed_all3", "mixed_all3")   # same labels without the batch; v4 as shipped
improve.ARMS[ARM] = dict(labels="ctrl", gaps="mixed", blocks=improve.BLOCKS + ("geom",), draws=3)
SOFT_TAU_M = (1.0, 3.0)
PROJECT_S = (0.5, 1.0)
GRID = (12, 8)              # Voronoi sample points along x, y
ZONES = (6, 4)              # occupancy grid
NEAR_BALL_M = 20.0
CHUNK = 20000
SWITCH = (14.0, 20.0, 28.0, 40.0, 56.0)
DEAD = (20.0, 40.0, 80.0)


# -- features ----------------------------------------------------------------------------------

def velocity(xy, period, fps=25):
    out = np.full(xy.shape, np.nan)
    for p in np.unique(period):
        r = np.flatnonzero(period == p)
        if len(r) > 4:
            out[r[2:-2]] = (xy[r[4:]] - xy[r[:-4]]) * fps / 4
    return out


def soft_min(d, tau):
    low = np.fmin.reduce(d, axis=1)
    e = np.where(np.isnan(d), 0.0, np.exp(-(d - low[:, None]) / tau))
    s = e.sum(axis=1)
    return np.where(s > 0, low - tau * np.log(np.maximum(s, 1e-12)), np.nan)


def nearest(d):
    filled = np.where(np.isnan(d), np.inf, d)
    low = filled.min(axis=1)
    return np.where(np.isinf(low), np.nan, low)


def geom_features(players, ball, team, period, pitch):
    """(n_frames, n_features), home side's point of view; no identities."""
    length, width = pitch
    P = players[:, :, :2].astype(np.float64)
    B = ball[:, :2].astype(np.float64)
    right = features.home_attacks_right(players, team, period)
    px = np.where(right[:, None], P[:, :, 0], length - P[:, :, 0])
    py = P[:, :, 1]
    bx = np.where(right, B[:, 0], length - B[:, 0])
    home, away = team == 0, team == 1
    n = len(B)
    d = np.hypot(P[:, :, 0] - B[:, None, 0], P[:, :, 1] - B[:, None, 1])
    f = {}
    for tau in SOFT_TAU_M:
        h, a = soft_min(d[:, home], tau), soft_min(d[:, away], tau)
        f["soft_home_{:g}".format(tau)], f["soft_away_{:g}".format(tau)] = h, a
        f["soft_diff_{:g}".format(tau)] = np.clip(a - h, -10, 10)
    v, bv = velocity(P, period), velocity(B, period)
    for t in PROJECT_S:
        Pp, Bp = P + v * t, B + bv * t
        dp = np.hypot(Pp[:, :, 0] - Bp[:, None, 0], Pp[:, :, 1] - Bp[:, None, 1])
        h, a = nearest(dp[:, home]), nearest(dp[:, away])
        f["proj_diff_{:g}".format(t)] = np.clip(a - h, -10, 10)
    # Pressure: how close the nearest opponent stands to each side's player nearest the ball.
    for name, own, other in (("home", home, away), ("away", away, home)):
        filled = np.where(np.isnan(d[:, own]), np.inf, d[:, own])
        pick = np.argmin(filled, axis=1)
        carrier = P[:, own][np.arange(n), pick]
        dc = np.hypot(P[:, other, 0] - carrier[:, None, 0], P[:, other, 1] - carrier[:, None, 1])
        f["pressure_" + name] = np.where(np.isinf(filled.min(axis=1)), np.nan, nearest(dc))
    f["pressure_diff"] = f["pressure_home"] - f["pressure_away"]
    # Voronoi share of the home side, from sample points in the home attacking frame.
    gx = (np.arange(GRID[0]) + 0.5) * length / GRID[0]
    gy = (np.arange(GRID[1]) + 0.5) * width / GRID[1]
    G = np.array([(x, y) for x in gx for y in gy])
    third = np.minimum((G[:, 0] / length * 3).astype(int), 2)
    share = np.full((n, 5), np.nan)
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        dg = np.hypot(px[s:e, None, :] - G[None, :, 0, None], py[s:e, None, :] - G[None, :, 1, None])
        dg = np.where(np.isnan(dg), np.inf, dg)
        owner_home = home[np.argmin(dg, axis=2)].astype(np.float64)
        owner_home[np.isinf(dg.min(axis=2))] = np.nan
        share[s:e, 0] = np.nanmean(owner_home, axis=1)
        for k in range(3):
            share[s:e, 1 + k] = np.nanmean(owner_home[:, third == k], axis=1)
        near = np.hypot(G[None, :, 0] - bx[s:e, None], G[None, :, 1] - B[s:e, 1, None]) < NEAR_BALL_M
        w = np.where(np.isnan(owner_home), 0.0, near)
        with np.errstate(invalid="ignore"):
            share[s:e, 4] = np.nansum(owner_home * w, axis=1) / w.sum(axis=1)
    for k, name in enumerate(("vor_all", "vor_own", "vor_mid", "vor_att", "vor_ball")):
        f[name] = share[:, k]
    # Occupancy, home minus away, in 6 x 4 zones of the home attacking frame.
    zx = np.clip((px / length * ZONES[0]).astype(np.float64), 0, ZONES[0] - 1)
    zy = np.clip((py / width * ZONES[1]).astype(np.float64), 0, ZONES[1] - 1)
    zone = np.where(np.isnan(zx) | np.isnan(zy), -1, np.floor(zx) * ZONES[1] + np.floor(zy)).astype(int)
    sign = np.where(home, 1.0, -1.0)
    for z in range(ZONES[0] * ZONES[1]):
        f["zone_{}".format(z)] = np.sum((zone == z) * sign[None, :], axis=1).astype(np.float64)
    with np.errstate(invalid="ignore"):
        cy_h, cy_a = np.nanmean(py[:, home], axis=1), np.nanmean(py[:, away], axis=1)
    f["ball_y_side"] = np.abs(B[:, 1] - cy_a) - np.abs(B[:, 1] - cy_h)
    for name in ("soft_diff_1", "proj_diff_0.5", "vor_ball"):
        f[name + "_c25"] = features.centred_mean(f[name], 25)
    return np.column_stack(list(f.values())).astype(np.float32)


# -- caches ------------------------------------------------------------------------------------

def extend(path, add):
    with np.load(path) as f:
        out = {k: f[k] for k in f.files}
    out.update(add)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)


def prep_idsse(args):
    index, match_id, gaps, lengths, share, draw = args
    path = improve.cache_path(match_id, gaps, draw)
    with np.load(path) as f:
        if "geom" in f.files:
            return path
        cached = f["base"]
    data = idsse.load(match_id)
    y = np.array([train.CODE[v] for v in data["labels"]], dtype=np.int8)
    period, pitch, team = data["period"], tuple(data["pitch"]), data["team"]
    seed = [train.SEED, index, draw] if draw else [train.SEED, index]
    streams = np.random.default_rng(seed).spawn(3)
    if improve.LIVE_DEAD:
        data = dict(data, ball=train.moving_dead_ball(data["players"], data["ball"], y, streams[2]))
    players, ball = train.degrade(data, streams[0], lengths, 0.0)
    if gaps == "dead":
        hidden = train.dead_mask(y, streams[1], lengths, train.GAP_GIVEN_ALIVE)
    else:
        hidden = train.ball_mask(len(y), streams[1], lengths, share)
    ball[hidden] = np.nan
    assert np.array_equal(features.build(players, ball, team, period, pitch)[0], cached, equal_nan=True), path
    extend(path, dict(geom=geom_features(players, ball, team, period, pitch),
                      geom_swap=geom_features(players, ball, 1 - team, period, pitch)))
    return path


def prep_metrica(args):
    game, repeat, gaps, lengths, share = args
    path = os.path.join(metrica_exp.OUT, "g{}_{}_{}.npz".format(game, repeat, gaps))
    with np.load(path) as f:
        if "geom" in f.files:
            return path
        cached = f["base"]
    data = metrica_exp.load_game(game)
    y, period, team = data["y"], data["period"], data["team"]
    if improve.LIVE_DEAD:
        moved = np.random.default_rng([train.SEED, game, repeat, 1])
        data = dict(data, ball=train.moving_dead_ball(data["players"], data["ball"], y, moved))
    streams = np.random.default_rng([train.SEED, game, repeat]).spawn(2)
    players, ball = train.degrade(data, streams[0], lengths, 0.0)
    if gaps == "dead":
        hidden = train.dead_mask(y, streams[1], lengths, train.GAP_GIVEN_ALIVE)
    else:
        hidden = train.ball_mask(len(y), streams[1], lengths, share)
    ball[hidden] = np.nan
    pitch = tuple(data["pitch"])
    assert np.array_equal(features.build(players, ball, team, period, pitch)[0], cached, equal_nan=True), path
    extend(path, dict(geom=geom_features(players, ball, team, period, pitch)))
    return path


# -- scoring -----------------------------------------------------------------------------------

def table(lp, y, period):
    return {"{:g}_{:g}".format(s, d): train.score(y, improve.decode(lp, None, period, s, 0.0, d), period)[0]
            for s in SWITCH for d in DEAD}


def idsse_job(args):
    arm, held, gap = args
    with np.load(os.path.join(improve.OUT, "oof", arm, held + ".npz")) as f:
        lp = f["lp_" + gap].astype(np.float64)
    with np.load(improve.cache_path(held, "uniform")) as f:
        y, period = f["y_ctrl"], f["period"]
    return arm, held, gap, table(lp, y, period)


def metrica_job(args):
    arm, game, repeat, gap = args
    with np.load(os.path.join(metrica_exp.OUT, "lp_{}_{}_{}_{}.npz".format(arm, game, repeat, gap))) as f:
        lp, y, period = f["lp"].astype(np.float64), f["y"], f["period"]
    y = live_review.metrica_control(y, *live_review.metrica_events(game))
    return arm, gap, table(lp, y, period)


def score():
    arms = REFS + (ARM,)
    for arm in arms:
        live_review.metrica_probabilities(arm, improve.ARMS[arm]["blocks"])
    with ProcessPoolExecutor(6) as pool:
        id_done = list(pool.map(idsse_job, [(a, m, g) for a in arms for m in idsse.MATCHES for g in improve.GAPS]))
        me_done = list(pool.map(metrica_job, [(a, game, r, g) for a in arms for game in metrica_exp.GAMES
                                              for r in range(metrica_exp.REPEATS) for g in improve.GAPS]))
    summary = {}
    for gap in improve.GAPS:
        result = {}
        for arm in arms:
            by = {h: t for a, h, g, t in id_done if a == arm and g == gap}
            folds, picked = [], []
            for held in idsse.MATCHES:
                best = max(by[held], key=lambda k: np.mean([by[m][k] for m in idsse.MATCHES if m != held]))
                folds.append(by[held][best])
                picked.append(best)
            s = float(np.median([float(k.split("_")[0]) for k in picked]))
            d = float(np.median([float(k.split("_")[1]) for k in picked]))
            key = "{:g}_{:g}".format(min(SWITCH, key=lambda v: abs(v - s)), min(DEAD, key=lambda v: abs(v - d)))
            fixed = [by[h]["40_40"] for h in idsse.MATCHES]
            me = [t for a, g, t in me_done if a == arm and g == gap]
            result[arm] = dict(folds=np.array(folds), fixed=np.array(fixed), picked=picked, key=key,
                               metrica=float(np.mean([t[key] for t in me])),
                               metrica_fixed=float(np.mean([t["40_40"] for t in me])))
        a = result[ARM]
        for ref in REFS:
            r = result[ref]
            diff, diff_fixed = a["folds"] - r["folds"], a["fixed"] - r["fixed"]
            summary["{}|vs_{}".format(gap, ref)] = dict(
                ref=float(r["folds"].mean()), arm=float(a["folds"].mean()), diff=float(diff.mean()),
                up=int(np.sum(diff > 0)), diff_fixed=float(diff_fixed.mean()), up_fixed=int(np.sum(diff_fixed > 0)),
                metrica_ref=r["metrica"], metrica_arm=a["metrica"],
                metrica_ref_fixed=r["metrica_fixed"], metrica_arm_fixed=a["metrica_fixed"],
                picked=a["picked"], ref_picked=r["picked"], folds=[round(float(v), 4) for v in diff])
    with open(os.path.join(improve.OUT, "geom.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    for g, v in summary.items():
        print("{:<30} IDSSE {:.3f} -> {:.3f} {:+.4f} ({}/7) | a (40,40) {:+.4f} ({}/7) | Metrica {:.3f} -> {:.3f}, "
              "a (40,40) {:.3f} -> {:.3f} | costes {}".format(
                  g, v["ref"], v["arm"], v["diff"], v["up"], v["diff_fixed"], v["up_fixed"], v["metrica_ref"],
                  v["metrica_arm"], v["metrica_ref_fixed"], v["metrica_arm_fixed"], sorted(set(v["picked"]))))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "prep":
        lengths, share, real = dose_exp.gap_profile()
        assert real
        jobs = [(i, m, g, lengths, share, d) for d in range(3) for i, m in enumerate(idsse.MATCHES) for g in improve.GAPS]
        with ProcessPoolExecutor(3) as pool:
            for path in pool.map(prep_idsse, jobs):
                print("ready", os.path.basename(path), flush=True)
        jobs = [(g, r, gaps, lengths, share) for g in metrica_exp.GAMES for r in range(metrica_exp.REPEATS)
                for gaps in improve.GAPS]
        with ProcessPoolExecutor(3) as pool:
            for path in pool.map(prep_metrica, jobs):
                print("ready", os.path.basename(path), flush=True)
    elif stage == "train":
        improve.train_arm(ARM)
    elif stage == "score":
        score()
