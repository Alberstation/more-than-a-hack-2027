"""Tracking in, H / A / D per frame out: evidence per frame, then one Viterbi pass per half.

    python possession/pipeline.py                     # writes evaluation/results/possession_v5.csv
    python possession/pipeline.py --out my_team.csv

Three changes over starter/baseline.py, each one a line of the improvement plan (docs/plan):

  M1  Viterbi over H / A / D with a cost per change of state, instead of a majority vote.
  M2  No ball is not dead ball. A missing ball contributes no evidence, so the current possession
      continues through the gap instead of being cut in two.
  M3  Dead ball from its own evidence: collective walking pace, the last seen ball beyond a line,
      and the graded stretch before the second-half kickoff.

PARAMS are priors chosen from football reasoning, not fitted. They are deliberately NOT tuned
against the public five minutes -- that is against the rules and picks the wrong answer anyway.
They will be fitted on external matches (IDSSE/DFL, line M5 of the plan).
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from baseline import write_submission  # noqa: E402
from load_smart_tagging import load as load_smart_tagging  # noqa: E402

from possession import atd, decode, features  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

# Params
PARAMS = {
    # H vs A from the ball: evidence = gain * clip(d_away - d_home, +/- cap), in metres, faded out
    # when the ball is far from everybody (in flight, nobody controls it).
    "team_gain": 0.5,
    "team_cap_m": 3.0,
    "control_radius_m": 1.5,
    "flight_fade_m": 2.0,
    # Dead ball: + per frame when the players walk, + when the last seen ball is out.
    "walk_speed_ms": 1.1,
    "walk_gain": 1.0,
    "walk_cap": 3.0,
    "speed_window_frames": 25,
    "outside_margin_m": 1.0,
    "outside_gain": 3.0,
    # Cost of one change of state, in the same log units as the evidence.
    "switch_cost": 25.0,
    "dead_cost": 20.0,
}


def kickoff_frames(game):
    """Start frame of each 'Kick Off (Start)' Smart Tagging window, by half label."""
    out = {}
    for event in load_smart_tagging(game):
        if (event["code"].lower() == "set pieces"
                and event["tags"].get("Type", "").lower() == "kick off (start)"):
            out[event["tags"].get("Half")] = event["start_frame"]
    return out


def emissions(players, ball, team, pitch, params):
    """(n_frames, 3) per-frame log scores for H, A, D."""
    d_home, d_away = features.team_distances(players, ball, team)
    nearest = np.fmin(d_home, d_away)
    fade = np.exp(-np.maximum(nearest - params["control_radius_m"], 0) / params["flight_fade_m"])
    margin = np.clip(d_away - d_home, -params["team_cap_m"], params["team_cap_m"])
    team_evidence = np.nan_to_num(params["team_gain"] * margin * fade)

    speed = features.collective_speed(players, params["speed_window_frames"])
    walk = np.clip(params["walk_gain"] * (params["walk_speed_ms"] - speed),
                   -params["walk_cap"], params["walk_cap"])
    outside = features.ball_outside(ball, pitch, params["outside_margin_m"])
    dead = np.nan_to_num(walk) + params["outside_gain"] * outside

    out = np.zeros((len(ball), 3))
    out[:, 0] = team_evidence / 2
    out[:, 1] = -team_evidence / 2
    out[:, 2] = dead
    return out


def predict(game=GAME, params=None):
    params = dict(PARAMS, **(params or {}))
    players, ball, team = atd.load(game)
    scores = emissions(players, ball, team, GAMES[game]["pitch"], params)

    # Before a half's kickoff the ball cannot be in play.
    half_names = {1: "1st Half", 2: "2nd Half"}
    kickoffs = kickoff_frames(game)
    transitions = decode.transition_matrix(params["switch_cost"], params["dead_cost"])

    labels = np.full(len(ball), "X", dtype="<U1")
    for half, first, last in half_video_blocks(game):
        block = scores[first:last + 1].copy()
        kickoff = kickoffs.get(half_names[half])
        if kickoff is not None and kickoff > first:
            block[:kickoff - first, :2] = -1e6
        path = decode.viterbi(block, transitions)
        labels[first:last + 1] = decode.STATES[path]
    return labels


def predict_v2(game=GAME):
    """The learned model (train.py) and its CV-selected decoder costs, on the challenge match."""
    import pickle
    from possession import train

    with open(train.MODEL_PATH, "rb") as handle:
        bundle = pickle.load(handle)
    players, ball, team = atd.load(game)
    blocks = half_video_blocks(game)
    period = np.zeros(len(ball), dtype=np.int8)
    for half, first, last in blocks:
        period[first:last + 1] = half
    X, names = features.build(players, ball, team, period, GAMES[game]["pitch"])
    if names != bundle["names"]:
        raise ValueError("feature set changed since the model was trained; rerun train.py")
    scores = train.log_proba(bundle["model"], X)

    half_names = {1: "1st Half", 2: "2nd Half"}
    kickoffs = kickoff_frames(game)
    transitions = decode.transition_matrix(bundle["switch_cost"], bundle["dead_cost"])
    labels = np.full(len(ball), "X", dtype="<U1")
    for half, first, last in blocks:
        block = scores[first:last + 1].copy()
        kickoff = kickoffs.get(half_names[half])
        if kickoff is not None and kickoff > first:
            block[:kickoff - first, :2] = -1e6
        labels[first:last + 1] = decode.STATES[decode.viterbi(block, transitions)]
    return labels


def predict_v4(game=GAME):
    """The v3 pipeline on the model trained with a moving dead ball."""
    from possession import train_v4

    return predict_v3(game, train_v4.MODEL_PATH)


def predict_v3(game=GAME, model_path=None):
    """v2's decoder on the v3 model: more evidence, trained on both ball-gap patterns."""
    import pickle
    from possession import features_v3, train, train_v3

    with open(model_path or train_v3.MODEL_PATH, "rb") as handle:
        bundle = pickle.load(handle)
    players, ball, team = atd.load(game)
    blocks = half_video_blocks(game)
    period = np.zeros(len(ball), dtype=np.int8)
    for half, first, last in blocks:
        period[first:last + 1] = half
    X = features_v3.build(players, ball, team, period, GAMES[game]["pitch"])
    scores = train.log_proba(bundle["model"], X)

    half_names = {1: "1st Half", 2: "2nd Half"}
    kickoffs = kickoff_frames(game)
    transitions = decode.transition_matrix(bundle["switch_cost"], bundle["dead_cost"])
    labels = np.full(len(ball), "X", dtype="<U1")
    for half, first, last in blocks:
        block = scores[first:last + 1].copy()
        kickoff = kickoffs.get(half_names[half])
        if kickoff is not None and kickoff > first:
            block[:kickoff - first, :2] = -1e6
        labels[first:last + 1] = decode.STATES[decode.viterbi(block, transitions)]
    return labels


def proba_v5(game=GAME):
    """(p_hgb, p_encoder, blocks): v5's two sets of per-frame class probabilities."""
    import pickle
    from possession import encoder, features_v3, v5

    with open(v5.HGB_PATH, "rb") as handle:
        hgb = pickle.load(handle)["model"]
    players, ball, team = atd.load(game)
    blocks = half_video_blocks(game)
    period = np.zeros(len(ball), dtype=np.int8)
    for half, first, last in blocks:
        period[first:last + 1] = half
    pitch = GAMES[game]["pitch"]
    p_hgb = hgb.predict_proba(features_v3.build(players, ball, team, period, pitch))
    p_enc = encoder.proba([encoder.load(p) for p in v5.ENCODER_PATHS], players, ball, team, period, pitch)
    return p_hgb, p_enc, blocks


def predict_v5(game=GAME):
    """v4's HGB on control labels mixed with the set encoder (seeds averaged), one Viterbi per half."""
    from possession import v5

    p_hgb, p_enc, blocks = proba_v5(game)
    p = (1 - v5.ENCODER_WEIGHT) * p_hgb + v5.ENCODER_WEIGHT * p_enc
    scores = np.log(np.clip(p, 1e-4, 1.0))

    half_names = {1: "1st Half", 2: "2nd Half"}
    kickoffs = kickoff_frames(game)
    transitions = decode.transition_matrix(v5.SWITCH_COST, v5.DEAD_COST)
    labels = np.full(len(scores), "X", dtype="<U1")
    for half, first, last in blocks:
        block = scores[first:last + 1].copy()
        kickoff = kickoffs.get(half_names[half])
        if kickoff is not None and kickoff > first:
            block[:kickoff - first, :2] = -1e6
        labels[first:last + 1] = decode.STATES[decode.viterbi(block, transitions)]
    return labels


def set_piece_check(labels, game=GAME):
    """Share of Smart Tagging set pieces with a dead-to-alive restart inside their window.

    A plausibility check that needs no ground truth: every set piece is a restart, so a sound
    prediction should restart play somewhere inside almost every one of these windows.
    """
    hits = total = 0
    for event in load_smart_tagging(game):
        if event["code"].lower() != "set pieces":
            continue
        window = labels[event["start_frame"]:event["end_frame"]]
        total += 1
        hits += bool(np.any((window[:-1] == "D") & np.isin(window[1:], ["H", "A"])))
    return hits, total


def describe(labels, game=GAME):
    """Possession count and mean length on the graded frames -- a plausibility check that needs
    no ground truth."""
    spells = []
    for _, first, last in half_video_blocks(game):
        block = labels[first:last + 1]
        change = np.flatnonzero(block[1:] != block[:-1]) + 1
        starts = np.r_[0, change]
        ends = np.r_[change, len(block)]
        spells += [e - s for s, e in zip(starts, ends) if block[s] in ("H", "A")]
    graded = np.concatenate([labels[f:l + 1] for _, f, l in half_video_blocks(game)])
    share = {k: float(np.mean(graded == k)) for k in "HAD"}
    return len(spells), (np.mean(spells) / GAMES[game]["fps"] if spells else 0.0), share


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", choices=("v1", "v2", "v3", "v4", "v5"), default="v5",
                        help="v1: hand-set evidence; v2: model fitted on IDSSE (run train.py); "
                             "v3: frozen dead ball (run train_v3.py); v4: HGB with moving dead ball (run train_v4.py); "
                             "v5: best, HGB on control labels mixed with the set encoder (run v5.py)")
    parser.add_argument("--out", help="default evaluation/results/possession_<version>.csv")
    args = parser.parse_args()

    labels = {"v1": predict, "v2": predict_v2, "v3": predict_v3, "v4": predict_v4, "v5": predict_v5}[args.version]()
    out = args.out or os.path.join(RESULTS_DIR, "possession_{}.csv".format(args.version))
    path = write_submission(out, labels)
    n, mean_s, share = describe(labels)
    print("{} possessions, mean {:.1f} s   H {:.1%}  A {:.1%}  D {:.1%}".format(
        n, mean_s, share["H"], share["A"], share["D"]))
    print("set pieces with a restart inside their window: {} / {}".format(
        *set_piece_check(labels)))
    print("->", os.path.relpath(path, ROOT))
