"""How much ball does M4 have to recover to be worth it? Runs on IDSSE alone, no GPU, no video.

Every arm keeps the degraded players: M4 recovers the ball, never the players. That separates this
from team_exp.py's `clean`, which also restores the players and so overstates what M4 can reach.

    python possession/experiments/dose_exp.py                 # every arm, skips the ones done
    python possession/experiments/dose_exp.py ref41 ideal20   # just these

Arms are paired: one seed per match, so all of them share the same player degradation and the same
41% ball mask before any recovery. Each arm trains and tests in its own conditions, as train.py
will once the ball share changes. Results go to data/experiments/dose_exp.json.

ERROR_M and OUTLIER_SHARE are assumptions until WASB is measured against the ATD ball. Without the
challenge tracking the gap lengths are a stand-in too, flagged in the output: rerun then.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from possession import features, idsse, train  # noqa: E402
from possession.train import GAP_GIVEN_ALIVE, ball_mask, dead_mask  # noqa: E402

OUT = os.path.join(train.DATA_DIR, "experiments", "dose_exp.json")

# Params
MISSING_SHARE = 0.41        # published in DATA.md, used only as the stand-in's target
ERROR_M = 1.5               # assumed error of a recovered ball, metres
ERROR_FRAMES = 5
OUTLIER_SHARE = 0.03        # assumed share of recovered frames that are false detections
GRID = [(s, d) for s in (5.0, 10.0, 20.0) for d in (20.0, 40.0, 80.0)]

# name: (ball missing after recovery, recovered ball is noisy, where the gaps go)
# For "dead" gaps the first number is the missing share of alive frames instead.
ARMS = {
    "ref41": (MISSING_SHARE, False, "uniform"),
    "hard41": (MISSING_SHARE, False, "changes"),
    "ideal30": (0.30, False, "uniform"),
    "ideal20": (0.20, False, "uniform"),
    "ideal10": (0.10, False, "uniform"),
    "noisy30": (0.30, True, "uniform"),
    "noisy20": (0.20, True, "uniform"),
    "noisy10": (0.10, True, "uniform"),
    "ball_all": (0.0, False, "uniform"),
    "dead41": (GAP_GIVEN_ALIVE, False, "dead"),
    "dead_m4": (0.0, False, "dead"),
}


def gap_profile():
    """(lengths, share, is_real). The stand-in has a 1 s median and a long tail, not measured."""
    try:
        lengths, share = train.challenge_gap_profile()
        return lengths, float(share), True
    except (IOError, OSError) as error:
        print("STAND-IN gap lengths, provisional result:", error, flush=True)
        rng = np.random.default_rng(train.SEED)
        lengths = np.maximum(rng.lognormal(np.log(25), 1.3, 5000).astype(int), 1)
        return lengths, MISSING_SHARE, False


def recover(hidden, share, rng):
    """Give back whole gaps at random until `share` of the frames stay hidden."""
    hidden = hidden.copy()
    edges = np.diff(np.r_[0, hidden.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    recovered = np.zeros(len(hidden), dtype=bool)
    for i in rng.permutation(len(starts)):
        if hidden.mean() <= share:
            break
        hidden[starts[i]:ends[i]] = False
        recovered[starts[i]:ends[i]] = True
    return hidden, recovered


def build(index, match_id, arm, lengths, share):
    target, noisy, place = ARMS[arm]
    data = idsse.load(match_id)
    y = np.array([train.CODE[v] for v in data["labels"]], dtype=np.int8)
    n = len(y)
    pitch = tuple(data["pitch"])
    # Separate streams, so that no arm shifts the randomness of another.
    streams = np.random.default_rng([train.SEED, index]).spawn(3)
    players, ball = train.degrade(data, streams[0], lengths, 0.0)
    if place == "dead":
        hidden = dead_mask(y, streams[1], lengths, target)
        recovered = np.zeros(n, dtype=bool)
    else:
        events = np.flatnonzero(np.diff(y)) if place == "changes" else None
        hidden = ball_mask(n, streams[1], lengths, share, events)
        hidden, recovered = recover(hidden, min(target, share), streams[2])
    if noisy:
        error = train.smooth_noise(streams[2], (n, 2), ERROR_M, ERROR_FRAMES)
        ball[recovered, :2] += error[recovered]
        false = recovered & (streams[2].random(n) < OUTLIER_SHARE)
        ball[false, :2] = streams[2].uniform((0, 0), pitch, (int(false.sum()), 2))
    ball[hidden] = np.nan
    period = data["period"]
    X, _ = features.build(players, ball, data["team"], period, pitch)
    X_swap, _ = features.build(players, ball, 1 - data["team"], period, pitch)
    return dict(X=X, X_swap=X_swap, y=y, y_swap=np.where(y == 2, 2, 1 - y).astype(np.int8),
                period=period, missing=float(hidden.mean()))


def run(arm, lengths, share):
    t = time.time()
    matches = {m: build(i, m, arm, lengths, share) for i, m in enumerate(idsse.MATCHES)}
    print(arm, "built", round(time.time() - t), flush=True)
    scores = []
    for held in idsse.MATCHES:
        test = matches[held]
        model = train.fit([m for k, m in matches.items() if k != held])
        lp = train.log_proba(model, test["X"])
        scores.append([train.score(test["y"], train.decode_periods(lp, test["period"], *g),
                                   test["period"]) for g in GRID])
        print(" ", arm, held, round(time.time() - t), flush=True)
    scores = np.array(scores, dtype=np.float64)     # folds, grid, (spell, macro)
    best = int(np.argmax(scores[:, :, 0].mean(axis=0)))
    return dict(spell=float(scores[:, best, 0].mean()), macro=float(scores[:, best, 1].mean()),
                folds=scores[:, best, 0].round(4).tolist(), costs=GRID[best],
                missing=float(np.mean([m["missing"] for m in matches.values()])))


def report(results):
    ref = results.get("ref41")
    print("\narm        missing  spellF1  macroF1  vs ref41  folds up")
    for arm, r in results.items():
        if arm not in ARMS:
            continue
        diff = np.array(r["folds"]) - np.array(ref["folds"]) if ref else np.zeros(7)
        print("{:<10} {:.3f}   {:.4f}   {:.4f}   {:+.4f}    {}/7".format(
            arm, r["missing"], r["spell"], r["macro"], diff.mean(), int(np.sum(diff > 0))))


if __name__ == "__main__":
    lengths, share, real = gap_profile()
    results = {}
    if os.path.isfile(OUT):
        with open(OUT) as handle:
            results = json.load(handle)
    if results.get("real_gaps", real) != real:
        results = {}                    # never mix stand-in arms with real ones
    results["real_gaps"] = real
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for arm in sys.argv[1:] or ARMS:
        if arm in results:
            continue
        results[arm] = run(arm, lengths, share)
        with open(OUT, "w") as handle:
            json.dump(results, handle, indent=1)
    print("gap lengths:", "challenge" if real else "STAND-IN, provisional")
    report(results)
