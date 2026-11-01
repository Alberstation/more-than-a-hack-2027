"""What can the decoder add on top of improve.py's log-probabilities? No model is trained here.

    python possession/experiments/decoder_exp.py check <arm>
    python possession/experiments/decoder_exp.py mbr <arm> <gaps> <labels>
    python possession/experiments/decoder_exp.py tags <arm> <gaps> <labels>

`mbr` samples paths from the HMM and keeps the Viterbi candidate with the best expected spellF1.
`tags` simulates Smart Tagging on IDSSE, since cross-validation cannot measure it: the phases keep
the challenge file's codes, counts and lengths, and the tagged team is the phase's majority team,
flipped with probability `flip`, or replaced with share `shared` by what the model already believes.
That second dose is the pessimistic one: the tagger reads the same tracking, so its errors are
likely to fall where the model's do.

Costs for a match are the ones improve.py chose on the other six. So are temperature and gain.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from possession import idsse, train  # noqa: E402
import improve  # noqa: E402

# Params
TEMPERATURES = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
SAMPLE_COSTS = (6.5, 6.5)      # log-rates, about one change every 170 frames
N_SAMPLES = 100
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
GAINS = (0.01, 0.02, 0.05, 0.1, 0.2)
DOSES = ((0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.1, 0.5), (0.2, 0.5), (0.3, 0.5))
PHASES = ("BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "LONG BALL", "BALL IN FINAL THIRD")
TRANSITION_FALSE = 0.3
TRANSITION_JITTER = 25
TRANSITION_WINDOW = 50
TRANSITION_DISCOUNT = 0.5


@numba.njit(cache=True)
def spells(path, period):
    cut = np.empty(len(path) + 1, dtype=np.int64)
    n, start = 0, 0
    out = np.empty((len(path) // 2 + 2, 3), dtype=np.int64)
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[start] or period[i] != period[start]:
            if path[start] < 2:
                out[n, 0], out[n, 1], out[n, 2] = start, i, path[start]
                n += 1
            start = i
    return out[:n]


@numba.njit(cache=True)
def spell_f1(a, b):
    """Official greedy match, same order: IoU descending, then truth index, then prediction index."""
    ious, ai, bi = [], [], []
    first = 0
    for i in range(len(a)):
        while first < len(b) and b[first, 1] <= a[i, 0]:
            first += 1
        for j in range(first, len(b)):
            if b[j, 0] >= a[i, 1]:
                break
            if a[i, 2] != b[j, 2] or b[j, 1] <= a[i, 0]:
                continue
            inter = min(a[i, 1], b[j, 1]) - max(a[i, 0], b[j, 0])
            union = max(a[i, 1], b[j, 1]) - min(a[i, 0], b[j, 0])
            ious.append(inter / union)
            ai.append(i)
            bi.append(j)
    if len(ious) == 0:
        return 0.0
    v, ia, ib = np.array(ious), np.array(ai), np.array(bi)
    order = np.argsort(-v, kind="mergesort")
    total = 0.0
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        used_a = np.zeros(len(a), dtype=np.bool_)
        used_b = np.zeros(len(b), dtype=np.bool_)
        tp = 0
        for k in order:
            if v[k] < threshold:
                break
            if not used_a[ia[k]] and not used_b[ib[k]]:
                used_a[ia[k]] = True
                used_b[ib[k]] = True
                tp += 1
        if tp:
            p, r = tp / len(b), tp / len(a)
            total += 2 * p * r / (p + r)
    return total / 5


@numba.njit(cache=True)
def ffbs(em, switch, dead, n_samples, seed):
    """Forward filtering, backward sampling. One path per row."""
    np.random.seed(seed)
    n = em.shape[0]
    t = np.array([[0.0, -switch, -dead], [-switch, 0.0, -dead], [-dead, -dead, 0.0]])
    alpha = np.empty((n, 3))
    alpha[0] = em[0] - np.max(em[0])
    for i in range(1, n):
        for to in range(3):
            top = -1e18
            for fr in range(3):
                top = max(top, alpha[i - 1, fr] + t[fr, to])
            total = 0.0
            for fr in range(3):
                total += np.exp(alpha[i - 1, fr] + t[fr, to] - top)
            alpha[i, to] = top + np.log(total) + em[i, to]
        alpha[i] -= np.max(alpha[i])
    out = np.empty((n_samples, n), dtype=np.int8)
    p = np.empty(3)
    for s in range(n_samples):
        state = int(np.argmax(alpha[n - 1] + np.random.gumbel(0.0, 1.0, 3)))
        out[s, n - 1] = state
        for i in range(n - 2, -1, -1):
            for fr in range(3):
                p[fr] = np.exp(alpha[i, fr] + t[fr, state])
            u = np.random.random() * (p[0] + p[1] + p[2])
            state = 0 if u < p[0] else (1 if u < p[0] + p[1] else 2)
            out[s, i] = state
    return out


def load_oof(arm, held, gap):
    with np.load(os.path.join(improve.OUT, "oof", arm, held + ".npz")) as f:
        lp = f["lp_" + gap].astype(np.float64)
        logit = f["logit_" + gap] if "logit_" + gap in f.files else None
    with np.load(improve.cache_path(held, "uniform")) as f:
        return lp, logit, f["y_raw"], f["y_ctrl"], f["period"]


def picked_costs(arm, gap, label):
    with open(os.path.join(improve.OUT, "summary.json")) as handle:
        return json.load(handle)["{}|{}|{}".format(arm, gap, label)]["costs"]


def sample_paths(lp, period, temperature):
    parts = []
    for p in np.unique(period):
        r = np.flatnonzero(period == p)
        parts.append(ffbs(lp[r] / temperature, SAMPLE_COSTS[0], SAMPLE_COSTS[1],
                          N_SAMPLES, int(temperature) + int(p)))
    return np.hstack(parts)


def mbr_job(args):
    arm, held, gap, label = args
    lp, logit, y_raw, y_ctrl, period = load_oof(arm, held, gap)
    truth = y_ctrl if label == "ctrl" else y_raw
    grid = [(s, 0.0, d) for s in improve.SWITCH_GRID for d in improve.DEAD_GRID]
    grid += [(s, 0.0, d) for s in (1.25, 0.6) for d in improve.DEAD_GRID]
    paths = [improve.decode(lp, None, period, s, g, d) for s, g, d in grid]
    cands = [spells(p, period) for p in paths]
    out = {}
    for temperature in TEMPERATURES:
        samples = [spells(row, period) for row in sample_paths(lp, period, temperature)]
        expected = [np.mean([spell_f1(s, c) for s in samples]) for c in cands]
        best = int(np.argmax(expected))
        out[temperature] = dict(
            spell=train.score(truth, paths[best], period)[0], macro=train.score(truth, paths[best], period)[1],
            count=len(cands[best]), expected=float(expected[best]), pick=grid[best],
            samples=float(np.mean([len(s) for s in samples])))
    return held, out


def run_mbr(arm, gap, label):
    base = picked_costs(arm, gap, label)
    with ProcessPoolExecutor(7) as pool:
        done = dict(pool.map(mbr_job, [(arm, m, gap, label) for m in idsse.MATCHES]))
    names = idsse.MATCHES
    reference, mbr, picked = [], [], []
    for i, held in enumerate(names):
        lp, logit, y_raw, y_ctrl, period = load_oof(arm, held, gap)
        truth = y_ctrl if label == "ctrl" else y_raw
        s, g, d = base[i]
        reference.append(train.score(truth, improve.decode(lp, logit, period, s, g, d), period)[0])
        others = [m for m in names if m != held]
        best = max(TEMPERATURES, key=lambda t: np.mean([done[m][t]["spell"] for m in others]))
        mbr.append(done[held][best]["spell"])
        picked.append(best)
    d = np.array(mbr) - np.array(reference)
    print("{} {} {}: viterbi {:.4f}  mbr {:.4f}  diff {:+.4f}  up {}/7  temperatures {}".format(
        arm, gap, label, np.mean(reference), np.mean(mbr), d.mean(), int(np.sum(d > 0)), picked))
    for t in TEMPERATURES:
        print("  T={:<5} spell {:.4f} count {:.0f} sampled spells {:.0f} expected {:.4f}".format(
            t, np.mean([done[m][t]["spell"] for m in names]), np.mean([done[m][t]["count"] for m in names]),
            np.mean([done[m][t]["samples"] for m in names]), np.mean([done[m][t]["expected"] for m in names])))
    return dict(reference=reference, mbr=mbr, temperatures=picked)


# -- Smart Tagging, simulated ------------------------------------------------------------------

def tag_lengths():
    from load_smart_tagging import load
    out = {}
    for event in load():
        if event["code"] in PHASES and event["tags"].get("Team") not in (None, "N/A"):
            out.setdefault(event["code"], []).append(event["end_frame"] - event["start_frame"])
    return out


def simulate_tags(truth, believed, lengths, flip, shared, rng):
    """(frame, frame, team) per phase. Team comes from the phase's majority, as the tagger's would."""
    alive = np.flatnonzero(truth < 2)
    phases = []
    for code, sizes in lengths.items():
        for size in sizes:
            start = int(rng.choice(alive))
            stop = min(start + size, len(truth))
            source = believed if rng.random() < shared else truth
            window = source[start:stop]
            team = int(np.sum(window == 1) > np.sum(window == 0))
            if rng.random() < flip:
                team = 1 - team
            phases.append((start, stop, team))
    return phases


def tag_evidence(n, phases):
    bias = np.zeros(n)
    for start, stop, team in phases:
        bias[start:stop] += 1.0 if team == 0 else -1.0
    return bias


def transition_costs(truth, n_true, rng):
    alive = np.flatnonzero(truth < 2)
    changes = alive[1:][truth[alive[1:]] != truth[alive[:-1]]]
    picked = rng.choice(changes, min(n_true, len(changes)), replace=False)
    wrong = rng.choice(alive, int(round(n_true * TRANSITION_FALSE)), replace=False)
    scale = np.ones(len(truth))
    for frame in np.r_[picked, wrong]:
        centre = frame + rng.integers(-TRANSITION_JITTER, TRANSITION_JITTER + 1)
        scale[max(0, centre - TRANSITION_WINDOW):centre + TRANSITION_WINDOW] = TRANSITION_DISCOUNT
    return scale


def tags_job(args):
    arm, index, held, gap, label, costs, lengths = args
    lp, logit, y_raw, y_ctrl, period = load_oof(arm, held, gap)
    truth = y_ctrl if label == "ctrl" else y_raw
    s, g, d = costs
    ha = np.full(len(lp), -s) if logit is None else np.minimum(-s + g * logit, 0.0)
    believed = improve.decode(lp, logit, period, s, g, d)

    def run(em, cost):
        out = np.empty(len(em), dtype=np.int8)
        for p in np.unique(period):
            r = np.flatnonzero(period == p)
            out[r] = improve.viterbi_ha(em[r], cost[r], float(d))
        return train.score(truth, out, period)

    out = {"reference": run(lp, ha)}
    for flip, shared in DOSES:
        rng = np.random.default_rng([train.SEED, index, int(flip * 10), int(shared * 10)])
        bias = tag_evidence(len(lp), simulate_tags(truth, believed, lengths, flip, shared, rng))
        scale = transition_costs(truth, len(lengths["ATTACKING TRANSITION"]), rng)
        for gain in GAINS:
            em = lp.copy()
            em[:, 0] += gain * bias
            em[:, 1] -= gain * bias
            out[(flip, shared, gain, False)] = run(em, ha)
            out[(flip, shared, gain, True)] = run(em, ha * scale)
    return held, out


def run_tags(arm, gap, label):
    costs = picked_costs(arm, gap, label)
    lengths = tag_lengths()
    jobs = [(arm, i, m, gap, label, costs[i], lengths) for i, m in enumerate(idsse.MATCHES)]
    with ProcessPoolExecutor(7) as pool:
        done = dict(pool.map(tags_job, jobs))
    names = idsse.MATCHES
    reference = np.array([done[m]["reference"][0] for m in names])
    print("{} {} {}: reference {:.4f}".format(arm, gap, label, reference.mean()))
    print("flip shared transitions gain(nested) spellF1 diff up")
    table = {}
    for flip, shared in DOSES:
        for trans in (False, True):
            scores, picks = [], []
            for held in names:
                others = [m for m in names if m != held]
                gain = max(GAINS, key=lambda v: np.mean([done[m][(flip, shared, v, trans)][0] for m in others]))
                scores.append(done[held][(flip, shared, gain, trans)][0])
                picks.append(gain)
            diff = np.array(scores) - reference
            table["{}|{}|{}".format(flip, shared, trans)] = scores
            print(" {:.1f}  {:.1f}  {:<5}  {:<5} {:.4f} {:+.4f} {}/7".format(
                flip, shared, str(trans), float(np.median(picks)), np.mean(scores), diff.mean(), int(np.sum(diff > 0))))
    return dict(reference=reference.tolist(), table=table)


def check(arm):
    lp, logit, y_raw, y_ctrl, period = load_oof(arm, idsse.MATCHES[0], "uniform")
    path = improve.decode(lp, None, period, 10.0, 0.0, 40.0)
    for truth in (y_raw, y_ctrl):
        print(train.score(truth, path, period)[0], spell_f1(spells(truth, period), spells(path, period)))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "check":
        check(sys.argv[2])
    else:
        result = (run_mbr if stage == "mbr" else run_tags)(*sys.argv[2:5])
        name = "{}_{}.json".format(stage, "_".join(sys.argv[2:5]))
        with open(os.path.join(improve.OUT, name), "w") as handle:
            json.dump(result, handle, indent=1, default=float)
