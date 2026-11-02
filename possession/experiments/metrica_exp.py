"""Rehearsal on another provider: fit on all of IDSSE, score on Metrica's sample games 1 and 2.

    python possession/experiments/metrica_exp.py prep
    python possession/experiments/metrica_exp.py run <arm> ...

No Metrica frame is used for fitting or for choosing costs: those stay the median of what
improve.py picked on IDSSE. The labels are built from Metrica's events, not from a tracking field,
so they are a stand-in for the challenge's: dead is a missing ball, the owner is the team of the
last event that needs the ball. Challenges and fouls do not change the owner.

The corrected tracking has no ball exactly when play is dead, which would hand the model the
answer, so the ball is held through those frames before degrading, as in IDSSE.

Sample data: https://github.com/metrica-sports/sample-data (kept out of git, in data/external/metrica).
"""
import csv
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import features, idsse, train  # noqa: E402
from possession.features_v3 import touch_features  # noqa: E402
import decoder_exp  # noqa: E402
import dose_exp  # noqa: E402
import improve  # noqa: E402

FOLDER = os.path.join(train.DATA_DIR, "external", "metrica")
OUT = os.path.join(improve.OUT, "metrica")
PITCH = (105.0, 68.0)
GAMES = (1, 2)
REPEATS = 3
OWNER_EVENTS = ("PASS", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST")


def read_tracking(game, side):
    path = os.path.join(FOLDER, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(game, side))
    with open(path) as handle:
        handle.readline()
        handle.readline()
        names = handle.readline().strip().split(",")
    table = np.genfromtxt(path, delimiter=",", skip_header=3, dtype=np.float64)
    columns = [i for i, name in enumerate(names) if name.startswith("Player")]
    players = np.stack([table[:, [c, c + 1]] for c in columns], axis=1)
    ball_col = names.index("Ball")
    return table[:, 0].astype(np.int8), players, table[:, [ball_col, ball_col + 1]]


def owners(game, n):
    frames, teams = [], []
    with open(os.path.join(FOLDER, "Sample_Game_{}_RawEventsData.csv".format(game))) as handle:
        for row in csv.DictReader(handle):
            if row["Type"] in OWNER_EVENTS and row["Team"] in ("Home", "Away"):
                frames.append(int(row["Start Frame"]) - 1)
                teams.append(0 if row["Team"] == "Home" else 1)
    order = np.argsort(frames, kind="stable")
    frames, teams = np.array(frames)[order], np.array(teams)[order]
    owner = np.full(n, -1, dtype=np.int8)
    owner[np.clip(frames, 0, n - 1)] = teams
    known = np.maximum.accumulate(np.where(owner >= 0, np.arange(n), 0))
    return owner[known]


def load_game(game):
    period, home, ball = read_tracking(game, "Home")
    _, away, _ = read_tracking(game, "Away")
    n = min(len(home), len(away))
    period, home, away, ball = period[:n], home[:n], away[:n], ball[:n]
    players = np.concatenate([home, away], axis=1) * np.array(PITCH)
    ball = ball * np.array(PITCH)
    team = np.r_[np.zeros(home.shape[1], dtype=np.int8), np.ones(away.shape[1], dtype=np.int8)]
    keep = period > 0
    players, ball, period = players[keep], ball[keep], period[keep]
    n = len(ball)
    y = owners(game, n)[:n].copy()
    y[np.isnan(ball[:, 0])] = 2
    last = np.maximum.accumulate(np.where(~np.isnan(ball[:, 0]), np.arange(n), 0))
    held = ball[last]
    held[np.isnan(held)] = 0.0
    return dict(players=players.astype(np.float32), ball=held.astype(np.float32), team=team, period=period,
                pitch=np.array(PITCH), y=y.astype(np.int8))


def prepare(args):
    game, repeat, gaps, lengths, share = args
    path = os.path.join(OUT, "g{}_{}_{}.npz".format(game, repeat, gaps))
    if os.path.isfile(path):
        return path
    data = load_game(game)
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
    out = dict(y=y, period=period)
    out["base"] = features.build(players, ball, team, period, tuple(data["pitch"]))[0]
    out["touch"] = touch_features(players, ball, team, period)[0]
    out["shape"] = improve.shape_features(players, team, period)
    out["control"] = improve.control_features(players, ball, team)
    np.savez(path, **out)
    return path


def fit(arm):
    path = os.path.join(OUT, arm + ".pkl")
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    cfg = improve.ARMS[arm]
    key = "y_" + cfg.get("labels", "raw")
    gaps = cfg.get("gaps", "uniform")
    blocks = cfg.get("blocks", ("base",))
    train_gaps = improve.GAPS if gaps == "mixed" else (gaps,)
    pool = [improve.load_match(m, g, blocks, d) for d in range(cfg.get("draws", 1))
            for g in train_gaps for m in idsse.MATCHES]
    model = improve.Fitted(dict(cfg, _extra={}), pool, key, train.SEED)
    detector = improve.fit_flips(pool, key, train.SEED) if cfg.get("change") else None
    with open(path, "wb") as handle:
        pickle.dump((model, detector), handle)
    return model, detector


def run(arm):
    cfg = improve.ARMS[arm]
    blocks = cfg.get("blocks", ("base",))
    model, detector = fit(arm)
    with open(os.path.join(improve.OUT, "summary.json")) as handle:
        summary = json.load(handle)
    out = {}
    for gaps in improve.GAPS:
        for label in improve.LABELS:
            costs = np.median(np.array(summary["{}|{}|{}".format(arm, gaps, label)]["costs"]), axis=0)
            spell, macro, count, seconds = [], [], [], []
            for game in GAMES:
                for repeat in range(REPEATS):
                    with np.load(os.path.join(OUT, "g{}_{}_{}.npz".format(game, repeat, gaps))) as f:
                        X, y, period = np.hstack([f[b] for b in blocks]), f["y"], f["period"]
                    lp = improve.log_p(model.proba(X)).astype(np.float64)
                    logit = None
                    if detector is not None:
                        p = np.clip(detector.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
                        logit = np.log(p / (1 - p))
                    path = improve.decode(lp, logit, period, costs[0], costs[1], costs[2])
                    s, m = train.score(y, path, period)
                    found = decoder_exp.spells(path, period)
                    spell.append(s)
                    macro.append(m)
                    count.append(len(found))
                    seconds.append(np.mean(found[:, 1] - found[:, 0]) / improve.FPS)
            out["{}|{}".format(gaps, label)] = dict(spell=spell, macro=macro, count=float(np.mean(count)),
                                                    seconds=float(np.mean(seconds)), costs=costs.tolist())
            print("{:<18} gaps={:<7} costs_for={:<4} spell {:.4f} macro {:.4f} n {:.0f} mean {:.1f} s".format(
                arm, gaps, label, np.mean(spell), np.mean(macro), np.mean(count), np.mean(seconds)), flush=True)
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "prep":
        lengths, share, real = dose_exp.gap_profile()
        assert real
        jobs = [(g, r, gaps, lengths, share) for g in GAMES for r in range(REPEATS) for gaps in improve.GAPS]
        with ProcessPoolExecutor(4) as pool:
            for path in pool.map(prepare, jobs):
                print("ready", os.path.basename(path), flush=True)
        for g in GAMES:
            d = load_game(g)
            found = decoder_exp.spells(d["y"], d["period"])
            print("game", g, "frames", len(d["y"]), "spells", len(found),
                  "mean s", round(float(np.mean(found[:, 1] - found[:, 0])) / improve.FPS, 1),
                  "dead share", round(float(np.mean(d["y"] == 2)), 3))
    else:
        results = {arm: run(arm) for arm in sys.argv[2:]}
        path = os.path.join(OUT, "summary.json")
        old = json.load(open(path)) if os.path.isfile(path) else {}
        old.update(results)
        with open(path, "w") as handle:
            json.dump(old, handle, indent=1)
