"""Plan 3j S6: is v5 hurt by a degradation unlike the one it trained on? No training. LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/stress_random.py prep
    <torch-python> encoder_train.py <improve_live> predict <arm> rand      (encoder, encoder_s2028, encoder_s2029)
    LIVE_DEAD=1 python possession/experiments/stress_random.py score

Each Metrica game x draw gets its own random degradation within plausible ranges around the
challenge profile (dead-ball and live gap shares, gap length scale, player drop share, position
noise, ball blur), gaps placed as the `dead` pattern. v5's models fitted on all IDSSE (never saw
Metrica) score it; compared with the same models under `dead`. Idea 6 (randomised degradation in
training) is out if the mean drop is < 0.01.
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
import improve  # noqa: E402
import live_review  # noqa: E402
import metrica_exp  # noqa: E402
import metrica_mix  # noqa: E402
import speed_gaps  # noqa: E402

HGB, ENCODERS = speed_gaps.HGB, speed_gaps.ENCODERS
RANGES = dict(dead=(0.55, 0.9), alive=(0.07, 0.2), length=(0.5, 2.0), drop=(0.03, 0.15), noise=(0.1, 0.5),
              blur=(0.0, 10.0))


def draw(game, repeat):
    rng = np.random.default_rng([train.SEED, game, repeat, 77])
    return {k: float(rng.uniform(*v)) for k, v in RANGES.items()}


def prep_job(args):
    game, repeat, lengths = args
    p = draw(game, repeat)
    train.GAP_GIVEN_DEAD, train.PLAYER_DROP_SHARE, train.POSITION_NOISE_M = p["dead"], p["drop"], p["noise"]
    lengths = np.maximum(1, np.round(np.asarray(lengths) * p["length"])).astype(int)
    data = metrica_exp.load_game(game)
    y, period, team, pitch = data["y"], data["period"], data["team"], np.array(data["pitch"])
    moved = np.random.default_rng([train.SEED, game, repeat, 1])
    data = dict(data, ball=train.moving_dead_ball(data["players"], data["ball"], y, moved))
    streams = np.random.default_rng([train.SEED, game, repeat, 78]).spawn(2)
    players, ball = train.degrade(data, streams[0], lengths, 0.0, ball_smooth_frames=p["blur"])
    hidden = train.dead_mask(y, streams[1], lengths, p["alive"])
    ball[hidden] = np.nan
    with open(os.path.join(metrica_exp.OUT, HGB + ".pkl"), "rb") as handle:
        model = pickle.load(handle)[0]
    X = np.hstack([features.build(players, ball, team, period, tuple(pitch))[0],
                   improve.touch_features(players, ball, team, period)[0],
                   improve.shape_features(players, team, period),
                   improve.control_features(players, ball, team)])
    np.savez(metrica_mix.lp_path(HGB, game, repeat, "rand"), lp=improve.log_p(model.proba(X)).astype(np.float64),
             y=y, period=period)
    steps = encoder_data.step_frames(period)
    pp, b = encoder_data.tensors(players, ball, team, period, tuple(pitch), steps)
    kick = np.stack([encoder_data.kick_steps(players, ball, s, steps) for s in (team, 1 - team)]).astype(np.float16)
    y_ctrl = live_review.metrica_control(y, *live_review.metrica_events(game))
    np.savez(os.path.join(encoder_data.FOLDER, "metrica_g{}_{}_rand.npz".format(game, repeat)), players=pp, ball=b,
             kick=kick, steps=steps, period=period, y_raw=y, y_ctrl=y_ctrl)
    return game, repeat, p, float(np.isnan(ball[:, 0]).mean())


def prep():
    lengths, share, real = dose_exp.gap_profile()
    assert real
    with ProcessPoolExecutor(3) as pool:
        for game, repeat, p, miss in pool.map(prep_job, [(g, r, lengths) for g in metrica_exp.ALL_GAMES
                                                           for r in range(metrica_exp.REPEATS)]):
            print("g{} r{} sin balon {:.0%}  ".format(game, repeat, miss)
                  + " ".join("{} {:.2f}".format(k, v) for k, v in p.items()), flush=True)


def score():
    jobs = [(g, r, gap) for gap in ("dead", "rand") for g in metrica_exp.ALL_GAMES for r in range(metrica_exp.REPEATS)]
    with ProcessPoolExecutor(6) as pool:
        done = list(pool.map(speed_gaps.score_job, jobs))
    by = {(d[0], d[1]): [] for d in done}
    for d in done:
        by[(d[0], d[1])].append(d[2])
    print("v5 de todo IDSSE sobre Metrica 1-3, constantes de v5")
    for gap in ("dead", "rand"):
        per = [np.mean(by[(gap, g)]) for g in metrica_exp.ALL_GAMES]
        print("  {:<5} {:.3f}   por partido {}".format(gap, np.mean(per), " ".join("{:.3f}".format(v) for v in per)))
    rand = [d[2] for d in done if d[0] == "rand"]
    dead = [d[2] for d in done if d[0] == "dead"]
    print("  caida media {:+.3f}, peor sorteo {:+.3f}".format(np.mean(rand) - np.mean(dead),
                                                           min(rand) - np.mean(dead)))


if __name__ == "__main__":
    prep() if sys.argv[1] == "prep" else score()
