"""Fit the per-frame H / A / D model on IDSSE and select decoder costs by leave-one-match-out CV.

    python possession/train.py            # CV report, then fits the final model on all 7 matches

Why this is the only place anything is fitted: the rules forbid hand labels on the challenge match
and forbid tuning on its public five minutes, but allow labelled data from other games. IDSSE has
H / A / D on every frame of seven matches (see idsse.py). Its tracking is clean, so every match is
first DEGRADED to look like the challenge's automatic tracking -- the ball removed in long
stretches drawn from the challenge's own gap lengths, players dropped in and out, position noise
-- so that the model learns to work in the conditions it will meet. Only input data of the
challenge match is used for that (gap lengths, players visible), never its labels.

Every score printed here is measured with evaluation/possession_metrics.py on held-out IDSSE
matches, so the baseline, v1 and the learned model are compared on equal, legal terms.
"""

import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import DATA_DIR  # noqa: E402
import possession_metrics as pm  # noqa: E402

from possession import atd, decode, features, idsse  # noqa: E402

MODEL_PATH = os.path.join(DATA_DIR, "models", "hgb_v2.pkl")
CODE = {"H": 0, "A": 1, "D": 2}

# Params
SEED = 2027
TRAIN_STRIDE = 5            # every 5th frame for fitting: neighbours are near-duplicates
PLAYER_DROP_SHARE = 0.07    # challenge tracking sees ~20.5 of 22 players
POSITION_NOISE_M = 0.25
NOISE_SMOOTHING_FRAMES = 25
HARD_SHARE = 0.7            # share of gaps placed on a label change, when events are given
GAP_GIVEN_DEAD = 0.75       # ball missing on frames v2 calls dead in the challenge match, no labels
GAP_GIVEN_ALIVE = 0.127     # same, on frames v2 calls alive
DEAD_HOLD_FRAMES = (25, 100)  # 1 to 4 s with one player, while play is stopped
DEAD_PASS_M_PER_FRAME = 0.4   # 10 m/s at 25 fps, a typical pass between two of them, not fitted
GAP_JITTER = 50             # frames; a real gap does not start exactly where the ball goes dead
SWITCH_COSTS = (5.0, 10.0, 20.0, 40.0, 80.0)
DEAD_COSTS = (2.5, 5.0, 10.0, 20.0, 40.0)
HGB = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
           l2_regularization=1.0, random_state=SEED)


# -- degradation -------------------------------------------------------------------------------

def challenge_gap_profile():
    """(gap lengths, missing share) of the ball in the challenge tracking's graded frames."""
    from games import half_video_blocks
    _, ball, _ = atd.load()
    missing = np.zeros(len(ball), dtype=np.int8)
    for _, first, last in half_video_blocks("BAR-ATH"):
        missing[first:last + 1] = np.isnan(ball[first:last + 1, 0])
    edges = np.diff(np.r_[0, missing, 0])
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    graded = sum(last - first + 1 for _, first, last in half_video_blocks("BAR-ATH"))
    return lengths, lengths.sum() / graded


def ball_mask(n, rng, lengths, share, events=None):
    hidden = np.zeros(n, dtype=bool)
    while hidden.mean() < share:
        length = int(rng.choice(lengths))
        if events is not None and rng.random() < HARD_SHARE:
            start = max(0, int(rng.choice(events)) - int(rng.integers(0, length + 1)))
        else:
            start = int(rng.integers(0, max(1, n - length)))
        hidden[start:start + length] = True
    return hidden


def dead_mask(y, rng, lengths, alive_share):
    """Gaps that follow the dead-ball stretches, up to GAP_JITTER frames out of step with them."""
    n = len(y)
    dead = y == 2
    hidden = np.zeros(n, dtype=bool)
    for state, share in ((dead, GAP_GIVEN_DEAD), (~dead, alive_share)):
        while hidden[state].mean() < share:
            length = int(rng.choice(lengths))
            start = int(rng.integers(0, max(1, n - length)))
            frames = np.arange(start, min(n, start + length))
            shift = int(rng.integers(-GAP_JITTER, GAP_JITTER + 1))
            hidden[frames[state[np.clip(frames + shift, 0, n - 1)]]] = True
    return hidden


def moving_dead_ball(players, ball, y, rng):
    """The ball handed between players through dead stretches, on its way to the restart.

    IDSSE and Metrica hold the ball still while play is stopped and the real tracker does not, so a
    still ball tells a model that play is dead in a way the challenge never will.
    """
    out = ball.copy()
    edges = np.diff(np.r_[0, (y == 2).astype(np.int8), 0])
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        first, last = ball[max(start - 1, 0), :2], ball[min(end, len(y) - 1), :2]
        size = end - start
        path = np.zeros((size, 2))
        at = 0
        while at < size:
            stop = min(size, at + int(rng.integers(*DEAD_HOLD_FRAMES)))
            target = first + (at + stop) / (2 * size) * (last - first)
            gap = np.hypot(*(players[start + at, :, :2] - target).T)
            path[at:stop] = target
            if not np.all(np.isnan(gap)):
                carried = players[start + at:start + stop, int(np.nanargmin(gap)), :2]
                path[at:stop] = np.where(np.isnan(carried), target, carried)
            at = stop
        now = first.astype(np.float64)
        for i in range(size):
            step = path[i] - now
            now = now + step * min(1.0, DEAD_PASS_M_PER_FRAME / max(np.hypot(*step), 1e-6))
            out[start + i, :2] = now
    return out


def degrade(data, rng, gap_lengths, missing_share):
    players = data["players"].copy()
    ball = data["ball"].copy()
    n = len(ball)
    # Ball: remove whole stretches until the challenge's missing share is reached.
    hidden = np.zeros(n, dtype=bool)
    while hidden.mean() < missing_share:
        length = int(rng.choice(gap_lengths))
        start = int(rng.integers(0, max(1, n - length)))
        hidden[start:start + length] = True
    ball[hidden] = np.nan
    # Players: drop each slot for random stretches, so tracklets come and go.
    slots = players.shape[1]
    for s in range(slots):
        dropped = 0
        while dropped < PLAYER_DROP_SHARE * n:
            length = int(rng.integers(25, 750))
            start = int(rng.integers(0, max(1, n - length)))
            players[start:start + length, s] = np.nan
            dropped += length
    # Position error that drifts slowly, as a tracker's does. White noise of the same size would
    # read as ~2 m/s of phantom speed once differentiated at 25 fps, and speed is the main
    # dead-ball signal.
    players[..., :2] += smooth_noise(rng, players[..., :2].shape, POSITION_NOISE_M)
    ball[:, :2] += smooth_noise(rng, ball[:, :2].shape, POSITION_NOISE_M / 2)
    return players, ball


def smooth_noise(rng, shape, std, frames=NOISE_SMOOTHING_FRAMES):
    from scipy.ndimage import uniform_filter1d
    noise = uniform_filter1d(rng.normal(0, 1, shape), size=frames, axis=0, mode="nearest")
    return (noise * std / noise.std()).astype(np.float32)


def match_arrays(match_id, rng, gap_lengths, missing_share):
    data = idsse.load(match_id)
    players, ball = degrade(data, rng, gap_lengths, missing_share)
    period = data["period"]
    pitch = tuple(data["pitch"])
    y = np.array([CODE[v] for v in data["labels"]], dtype=np.int8)
    X, names = features.build(players, ball, data["team"], period, pitch)
    # The away side's view of the same frames: swap the teams and the H/A labels.
    X_swap, _ = features.build(players, ball, 1 - data["team"], period, pitch)
    y_swap = np.where(y == 2, 2, 1 - y).astype(np.int8)
    return dict(X=X, y=y, X_swap=X_swap, y_swap=y_swap, period=period, names=names,
                players=players, ball=ball, team=data["team"], pitch=pitch)


# -- scoring on IDSSE --------------------------------------------------------------------------

def score(y, pred, period):
    """(spellF1, macroF1) with the official functions, one run per period."""
    gt = y.astype(np.int8).copy()
    pr = pred.astype(np.int8).copy()
    # A one-frame X between periods keeps the scorer from joining them, as halftime does.
    cut = np.flatnonzero(np.diff(period)) + 1
    gt = np.insert(gt, cut, pm.X)
    pr = np.insert(pr, cut, pm.X)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pr, runs)
    spell = pm.spell_scores(gt, filled, runs)["mean_f1"]
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pr)))
    return spell, macro


def decode_periods(log_proba, period, switch_cost, dead_cost):
    transitions = decode.transition_matrix(switch_cost, dead_cost)
    out = np.empty(len(log_proba), dtype=np.int8)
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        out[rows] = decode.viterbi(log_proba[rows], transitions)
    return out


def baseline_predictions(m):
    """starter/baseline.py's smooth50, on the same degraded arrays."""
    d_home, d_away = features.team_distances(m["players"], m["ball"], m["team"])
    codes = np.where(np.isnan(d_home) | np.isnan(d_away), -1,
                     np.where(d_home <= d_away, 0, 1)).astype(np.int8)
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    from baseline import smooth
    labels = smooth(codes, 50)
    return np.array([CODE[v] for v in labels], dtype=np.int8)


def v1_predictions(m):
    from possession import pipeline
    scores = pipeline.emissions(m["players"], m["ball"], m["team"], m["pitch"], pipeline.PARAMS)
    return decode_periods(scores, m["period"], pipeline.PARAMS["switch_cost"],
                          pipeline.PARAMS["dead_cost"])


def fit(matches):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.concatenate([np.concatenate([m["X"][::TRAIN_STRIDE], m["X_swap"][::TRAIN_STRIDE]])
                        for m in matches])
    y = np.concatenate([np.concatenate([m["y"][::TRAIN_STRIDE], m["y_swap"][::TRAIN_STRIDE]])
                        for m in matches])
    return HistGradientBoostingClassifier(**HGB).fit(X, y)


def log_proba(model, X):
    return np.log(np.clip(model.predict_proba(X), 1e-4, 1.0))


def main():
    gap_lengths, missing_share = challenge_gap_profile()
    print("challenge ball gaps: {} stretches, {:.1%} of graded frames missing".format(
        len(gap_lengths), missing_share))
    rng = np.random.default_rng(SEED)
    matches = {}
    for match_id in idsse.MATCHES:
        matches[match_id] = match_arrays(match_id, rng, gap_lengths, missing_share)
        print("  built", match_id, matches[match_id]["X"].shape)

    grid = [(s, d) for s in SWITCH_COSTS for d in DEAD_COSTS]
    results = {key: [] for key in ["baseline", "v1"] + grid}
    for held in idsse.MATCHES:
        test = matches[held]
        model = fit([m for k, m in matches.items() if k != held])
        lp = log_proba(model, test["X"])
        results["baseline"].append(score(test["y"], baseline_predictions(test), test["period"]))
        results["v1"].append(score(test["y"], v1_predictions(test), test["period"]))
        for s, d in grid:
            results[(s, d)].append(score(test["y"], decode_periods(lp, test["period"], s, d),
                                         test["period"]))
        print("  fold {}: baseline {:.3f}  v1 {:.3f}  best v2 {:.3f}".format(
            held, results["baseline"][-1][0], results["v1"][-1][0],
            max(results[g][-1][0] for g in grid)))

    print("\nleave-one-match-out on degraded IDSSE (7 folds), mean spellF1 / macroF1")
    summary = {k: np.mean(v, axis=0) for k, v in results.items()}
    for key in ["baseline", "v1"]:
        print("  {:<24} {:.4f}  {:.4f}".format(key, *summary[key]))
    best = max(grid, key=lambda g: summary[g][0])
    for g in sorted(grid, key=lambda g: -summary[g][0])[:5]:
        print("  v2 switch={:<5g} dead={:<5g} {:.4f}  {:.4f}".format(g[0], g[1], *summary[g]))

    model = fit(list(matches.values()))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump({"model": model, "names": matches[idsse.MATCHES[0]]["names"],
                     "switch_cost": best[0], "dead_cost": best[1],
                     "cv": {str(k): v.tolist() for k, v in summary.items()}}, handle)
    print("\nfinal model on all 7 matches, costs switch={} dead={} -> {}".format(
        best[0], best[1], os.path.relpath(MODEL_PATH, ROOT)))


if __name__ == "__main__":
    main()
