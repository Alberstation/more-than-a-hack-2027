"""Ball gaps that start where the challenge loses the ball: slow balls and near the lines. LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/speed_gaps.py prep     # Metrica tests + hazard check
    <torch-python> encoder_train.py <improve_live> predict <arm> speed   (each encoder seed)
    LIVE_DEAD=1 python possession/experiments/speed_gaps.py score

gap_stats.py: per 1000 live frames with the ball seen, the challenge loses it 2.6 / 1.7 / 0.6 / 0.1
times at 0-2 / 2-5 / 5-10 / > 10 m/s, and 11.7 / 2.3 / 1.4 / 0.6 times outside / 0-3 / 3-10 / > 10 m
from a line; our gaps start anywhere. The `speed` pattern hides dead play as `dead` does and places
each live gap's start with odds proportional to those two hazards (input statistics only, no label).
Shortcut, no training: v5's HGB and encoders fitted on all IDSSE have never seen Metrica, so their
score on Metrica under `uniform`, `dead` and `speed` says whether v5 suffers from where gaps start.
Retrain only if it clearly does. v5's own constants (weight 0.75, costs 20 / 20).
"""
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import features, train  # noqa: E402
import dose_exp  # noqa: E402
import encoder_data  # noqa: E402
import gap_stats  # noqa: E402
import improve  # noqa: E402
import live_review  # noqa: E402
import metrica_exp  # noqa: E402
import metrica_mix  # noqa: E402

FPS = 25
SPEED_HAZARD = np.array([2.6, 1.7, 0.6, 0.1])      # gap_stats.SPEED_EDGES bins, challenge
LINE_HAZARD = np.array([11.7, 2.3, 1.4, 0.6])      # gap_stats.LINE_EDGES bins, challenge
HGB, ENCODERS = "ctrl_mixed_all3", ("encoder", "encoder_s2028", "encoder_s2029")
WEIGHT, SWITCH, DEAD = 0.75, 20.0, 20.0


def start_odds(ball, pitch):
    """Relative odds that a live gap starts at each frame, from the ball's speed and line distance."""
    B = ball[:, :2].astype(np.float64)
    v = np.zeros(len(B))
    v[1:] = np.nan_to_num(np.hypot(*(B[1:] - B[:-1]).T) * FPS)
    line = np.nan_to_num(np.minimum.reduce([B[:, 0], pitch[0] - B[:, 0], B[:, 1], pitch[1] - B[:, 1]]), nan=99.0)
    s = SPEED_HAZARD[np.clip(np.searchsorted(gap_stats.SPEED_EDGES, v, side="right") - 1, 0, 3)]
    li = LINE_HAZARD[np.clip(np.searchsorted(gap_stats.LINE_EDGES, line, side="right") - 1, 0, 3)]
    return s * li


def live_hazard(odds, dead, hidden, lengths, rng, base):
    """Walk the frames: at a live frame with the ball seen, a gap starts with probability base * odds."""
    out = hidden.copy()
    n = len(odds)
    u = rng.random(n)
    draws = rng.choice(lengths, size=n)
    i = 0
    while i < n:
        if not dead[i] and not out[i] and u[i] < base * odds[i]:
            stop = min(n, i + 1 + int(draws[i]))
            seg = np.arange(i + 1, stop)
            out[seg[~dead[seg]]] = True
            i = stop
        else:
            i += 1
    return out


def speed_mask(y, ball, pitch, rng, lengths, alive_share=train.GAP_GIVEN_ALIVE):
    """As train.dead_mask for dead play (without its jitter); in live play a gap starts at a seen frame with probability
    proportional to the challenge's hazard for that speed and line distance, the base set (by
    bisection, same random draws) so that the live share hidden is unchanged."""
    n = len(y)
    dead = y == 2
    hidden = np.zeros(n, dtype=bool)
    while hidden[dead].mean() < train.GAP_GIVEN_DEAD:
        length = int(rng.choice(lengths))
        start = int(rng.integers(0, max(1, n - length)))
        frames = np.arange(start, min(n, start + length))
        # No jitter here: dead_mask's jitter hides live frames next to every stoppage at whatever
        # speed the ball has, which flattens the speed hazard; the line odds below cover them.
        hidden[frames[dead[frames]]] = True
    odds = start_odds(ball, pitch)
    seed = int(rng.integers(0, 2**31))
    lo, hi = 0.0, 1.0 / odds.max()
    for _ in range(30):
        mid = (lo + hi) / 2
        share = live_hazard(odds, dead, hidden, lengths, np.random.default_rng(seed), mid)[~dead].mean()
        lo, hi = (mid, hi) if share < alive_share else (lo, mid)
    return live_hazard(odds, dead, hidden, lengths, np.random.default_rng(seed), hi)


def prep_job(args):
    game, repeat, lengths = args
    data = metrica_exp.load_game(game)
    y, period, team, pitch = data["y"], data["period"], data["team"], np.array(data["pitch"])
    moved = np.random.default_rng([train.SEED, game, repeat, 1])
    data = dict(data, ball=train.moving_dead_ball(data["players"], data["ball"], y, moved))
    streams = np.random.default_rng([train.SEED, game, repeat]).spawn(2)
    players, ball = train.degrade(data, streams[0], lengths, 0.0)
    hidden = speed_mask(y, ball, pitch, streams[1], lengths)
    check = gap_stats.stats(players, np.where(hidden[:, None], np.nan, ball), y < 2, pitch)
    ball[hidden] = np.nan
    with open(os.path.join(metrica_exp.OUT, HGB + ".pkl"), "rb") as handle:
        model = pickle.load(handle)[0]
    blocks = improve.ARMS[HGB]["blocks"]
    X = np.hstack([features.build(players, ball, team, period, tuple(pitch))[0],
                   improve.touch_features(players, ball, team, period)[0],
                   improve.shape_features(players, team, period),
                   improve.control_features(players, ball, team)])
    assert blocks == ("base", "touch", "shape", "control")
    np.savez(metrica_mix.lp_path(HGB, game, repeat, "speed"), lp=improve.log_p(model.proba(X)).astype(np.float64),
             y=y, period=period)
    steps = encoder_data.step_frames(period)
    p, b = encoder_data.tensors(players, ball, team, period, tuple(pitch), steps)
    kick = np.stack([encoder_data.kick_steps(players, ball, s, steps) for s in (team, 1 - team)]).astype(np.float16)
    y_ctrl = live_review.metrica_control(y, *live_review.metrica_events(game))
    np.savez(os.path.join(encoder_data.FOLDER, "metrica_g{}_{}_speed.npz".format(game, repeat)), players=p, ball=b,
             kick=kick, steps=steps, period=period, y_raw=y, y_ctrl=y_ctrl)
    return check


def prep():
    lengths, share, real = dose_exp.gap_profile()
    assert real
    with ProcessPoolExecutor(3) as pool:
        checks = list(pool.map(prep_job, [(g, r, lengths) for g in metrica_exp.ALL_GAMES
                                          for r in range(metrica_exp.REPEATS)]))
    mean = lambda k: np.mean([c[k] for c in checks], axis=0)
    print("Metrica con huecos `speed`: falta vivo {:.1%}, velocidad perder/seguir {:.1f}/{:.1f} m/s".format(
        mean("missing_live"), mean("speed_lost"), mean("speed_kept")))
    print("  riesgo por velocidad 0-2/2-5/5-10/>10 m/s: " + " ".join("{:.1f}".format(v) for v in mean("speed"))
          + "   (reto 2.6 1.7 0.6 0.1)")
    print("  riesgo por linea fuera/0-3/3-10/>10 m:     " + " ".join("{:.1f}".format(v) for v in mean("line"))
          + "   (reto 11.7 2.3 1.4 0.6)")


def score_job(args):
    game, repeat, gap = args
    def load(arm):
        with np.load(metrica_mix.lp_path(arm, game, repeat, gap)) as f:
            return f["lp"].astype(np.float64), f["y"], f["period"]
    lp_h, y, period = load(HGB)
    p_e = np.mean([np.exp(load(a)[0]) for a in ENCODERS], axis=0)
    lp = np.log(np.clip((1 - WEIGHT) * np.exp(lp_h) + WEIGHT * p_e, 1e-4, 1.0))
    y = live_review.metrica_control(y, *live_review.metrica_events(game))
    path = improve.decode(lp, None, period, SWITCH, 0.0, DEAD)
    hgb_only = improve.decode(lp_h, None, period, 40.0, 0.0, 40.0)
    return gap, game, train.score(y, path, period)[0], train.score(y, hgb_only, period)[0], \
        float(np.mean(path[y == 2] == 2)), float(np.mean(np.isnan(np.exp(lp_h[:, 0]))))


def score():
    jobs = [(g, r, gap) for gap in ("uniform", "dead", "speed") for g in metrica_exp.ALL_GAMES
            for r in range(metrica_exp.REPEATS)]
    with ProcessPoolExecutor(6) as pool:
        done = list(pool.map(score_job, jobs))
    print("v5 de todo IDSSE sobre Metrica 1-3 (sin entrenar), control labels, constantes de v5")
    print("  huecos     spellF1 v5   por partido 1/2/3        HGB solo   recall muerto")
    for gap in ("uniform", "dead", "speed"):
        r = [d for d in done if d[0] == gap]
        per = [np.mean([d[2] for d in r if d[1] == g]) for g in metrica_exp.ALL_GAMES]
        print("  {:<9} {:.3f}        {}      {:.3f}      {:.2f}".format(
            gap, np.mean([d[2] for d in r]), " ".join("{:.3f}".format(v) for v in per),
            np.mean([d[3] for d in r]), np.mean([d[4] for d in r])))


if __name__ == "__main__":
    prep() if sys.argv[1] == "prep" else score()
