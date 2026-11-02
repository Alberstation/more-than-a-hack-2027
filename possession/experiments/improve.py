"""One change at a time against a common reference, all on IDSSE, so that each one earns its place.

    python possession/experiments/improve.py prep               # degraded matches, cached
    python possession/experiments/improve.py train [arm ...]    # per-fold log-probabilities
    python possession/experiments/improve.py score              # table, nested decoder costs

Every arm is scored on the same frames, with the raw BallPossession labels and with the control
labels, on two gap patterns: `uniform` (as train.py degrades today) and `dead` (gaps follow the
stoppages, as the challenge's seem to). Decoder costs for a match are chosen on the other six, so
the costs never see the match they are scored on.

Experiment-only dependencies, not in requirements-solution.txt: numba, lightgbm.
"""
import datetime as dt
import glob
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import features, idsse, train  # noqa: E402
from possession.features_v3 import control_features, shape_features, touch_features  # noqa: E402
import dose_exp  # noqa: E402

LIVE_DEAD = bool(os.environ.get("LIVE_DEAD"))   # the ball moves through stoppages, tests included
FROZEN_OUT = os.path.join(train.DATA_DIR, "experiments", "improve")
OUT = FROZEN_OUT + "_live" if LIVE_DEAD else FROZEN_OUT
EVENTS = os.path.join(train.DATA_DIR, "external", "idsse_events")
FPS = 25

# Params
CONTROL_WINDOW_S = 1.5      # fixed before looking at any score; never swept
SWITCH_GRID = (2.5, 5.0, 10.0, 20.0, 40.0)
DEAD_GRID = (10.0, 20.0, 40.0, 80.0, 160.0)
GAIN_GRID = (0.0, 2.0, 4.0)
CHANGE_WINDOW = 6
WEIGHT_NEAR_CHANGE = 3.0
WEIGHT_FRAMES = 25
GAPS = ("uniform", "dead")
LABELS = ("raw", "ctrl")
BLOCKS = ("base", "touch", "shape", "control")

ARMS = {
    "ref": {},
    "es": dict(early=True),
    "two_heads": dict(model="two_heads"),
    "symmetric": dict(symmetric=True),
    "weighted": dict(weighted=True),
    "seeds3": dict(seeds=3),
    "lightgbm": dict(model="lightgbm"),
    "ctrl_labels": dict(labels="ctrl"),
    "dead_gaps": dict(gaps="dead"),
    "touch": dict(blocks=("base", "touch")),
    "shape": dict(blocks=("base", "shape")),
    "control": dict(blocks=("base", "control")),
    "all_feats": dict(blocks=BLOCKS),
    "change": dict(blocks=("base", "touch"), change=True),
    "self_train": dict(self_train=True),
    "mixed_all": dict(gaps="mixed", blocks=BLOCKS),
    "mixed_all3": dict(gaps="mixed", blocks=BLOCKS, draws=3),
    "mixed_change3": dict(gaps="mixed", blocks=BLOCKS, change=True, draws=3),
    "ctrl_all": dict(labels="ctrl", blocks=BLOCKS),
    "ctrl_dead": dict(labels="ctrl", gaps="dead"),
    "dead_all": dict(gaps="dead", blocks=BLOCKS),
    "ctrl_dead_all": dict(labels="ctrl", gaps="dead", blocks=BLOCKS),
    "ctrl_mixed_all": dict(labels="ctrl", gaps="mixed", blocks=BLOCKS),
    "ctrl_mixed_all3": dict(labels="ctrl", gaps="mixed", blocks=BLOCKS, draws=3),
    "frozen_all3": dict(gaps="mixed", blocks=BLOCKS, draws=3, frozen=True),
    "ctrl_frozen_all3": dict(labels="ctrl", gaps="mixed", blocks=BLOCKS, draws=3, frozen=True),
    "ctrl_mixed_change": dict(labels="ctrl", gaps="mixed", blocks=BLOCKS, change=True),
}


# -- control labels ----------------------------------------------------------------------------

def runs(a):
    cut = np.flatnonzero(np.diff(a)) + 1
    start, end = np.r_[0, cut], np.r_[cut, len(a)]
    return start, end, a[start]


def event_frames(match_id, y, period):
    """(frame, team 0/1) of every Play and ShotAtGoal, aligned on the two kick-offs."""
    stamp = lambda text: dt.datetime.fromisoformat(text).timestamp()
    rows, kick = [], {}
    for event in ET.parse(os.path.join(EVENTS, match_id + ".xml")).getroot().iter("Event"):
        when = event.get("EventTime")
        if not when:
            continue
        for child in event.iter():
            if child.tag == "KickOff" and child.get("GameSection") in ("firstHalf", "secondHalf"):
                kick[child.get("GameSection")] = stamp(when)
            if child.tag in ("Play", "ShotAtGoal") and child.get("Team"):
                rows.append((stamp(when), child.get("Team")))
    teams = sorted({team for _, team in rows})
    first = np.flatnonzero(period == 1)
    second = np.flatnonzero(period == 2)
    frames, owners = [], []
    for when, team in rows:
        half, section = (second, "secondHalf") if when >= kick["secondHalf"] - 60 else (first, "firstHalf")
        alive = half[y[half] < 2][0]
        frame = alive + int(round((when - kick[section]) * FPS))
        frames.append(min(max(frame, 0), len(y) - 1))
        owners.append(teams.index(team))
    frames, owners = np.array(frames), np.array(owners)
    alive = y[frames] < 2
    if np.mean(y[frames][alive] == owners[alive]) < 0.5:
        owners = 1 - owners
    return frames, owners


def control_labels(match_id, y, period):
    """A spell of a team that plays no ball within the window is a touch: hand it back."""
    frames, owners = event_frames(match_id, y, period)
    window = int(CONTROL_WINDOW_S * FPS)
    out = y.copy()
    for _ in range(3):
        start, end, value = runs(out)
        for i in range(1, len(start)):
            if value[i] < 2 and value[i - 1] < 2:
                near = (frames >= start[i] - window) & (frames < end[i] + window)
                if not np.any(owners[near] == value[i]):
                    out[start[i]:end[i]] = out[start[i] - 1]
    return out


# -- preparation -------------------------------------------------------------------------------

def cache_path(match_id, gaps, draw=0, root=None):
    tail = "_d{}".format(draw) if draw else ""
    return os.path.join(root or OUT, "matches", "{}_{}{}.npz".format(match_id, gaps, tail))


def prepare(args):
    index, match_id, gaps, lengths, share, draw = args
    path = cache_path(match_id, gaps, draw)
    if os.path.isfile(path):
        return path
    data = idsse.load(match_id)
    y = np.array([train.CODE[v] for v in data["labels"]], dtype=np.int8)
    period, pitch, team = data["period"], tuple(data["pitch"]), data["team"]
    seed = [train.SEED, index, draw] if draw else [train.SEED, index]
    streams = np.random.default_rng(seed).spawn(3)
    if LIVE_DEAD:
        data = dict(data, ball=train.moving_dead_ball(data["players"], data["ball"], y, streams[2]))
    players, ball = train.degrade(data, streams[0], lengths, 0.0)
    if gaps == "dead":
        hidden = train.dead_mask(y, streams[1], lengths, train.GAP_GIVEN_ALIVE)
    else:
        hidden = train.ball_mask(len(y), streams[1], lengths, share)
    ball[hidden] = np.nan
    out = dict(y_raw=y, y_ctrl=control_labels(match_id, y, period), period=period)
    for suffix, side in (("", team), ("_swap", 1 - team)):
        out["base" + suffix] = features.build(players, ball, side, period, pitch)[0]
        out["touch" + suffix] = touch_features(players, ball, side, period)[0]
        out["shape" + suffix] = shape_features(players, side, period)
        out["control" + suffix] = control_features(players, ball, side)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    return path


def load_match(match_id, gaps, blocks, draw=0, root=None):
    with np.load(cache_path(match_id, gaps, draw, root)) as f:
        return dict(X=np.hstack([f[b] for b in blocks]), X_swap=np.hstack([f[b + "_swap"] for b in blocks]),
                    y_raw=f["y_raw"], y_ctrl=f["y_ctrl"], period=f["period"])


# -- models ------------------------------------------------------------------------------------

def swap_y(y):
    return np.where(y == 2, 2, 1 - y).astype(np.int8)


def near_change(y):
    weight = np.ones(len(y), dtype=np.float32)
    for c in np.flatnonzero(np.diff(y)) + 1:
        weight[max(0, c - WEIGHT_FRAMES):c + WEIGHT_FRAMES] = WEIGHT_NEAR_CHANGE
    return weight


def rows(matches, key, weighted, mask=None):
    X, y, w = [], [], []
    for m in matches:
        s = train.TRAIN_STRIDE
        yy = m[key]
        wt = near_change(yy) if weighted else np.ones(len(yy), dtype=np.float32)
        for XX, label in ((m["X"], yy), (m["X_swap"], swap_y(yy))):
            X.append(XX[::s])
            y.append(label[::s])
            w.append(wt[::s])
    return np.concatenate(X), np.concatenate(y), np.concatenate(w)


def hgb(seed, **extra):
    params = dict(train.HGB, random_state=seed, early_stopping=False)
    params.update(extra)
    return HistGradientBoostingClassifier(**params)


def lgbm(seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=63, min_child_samples=200,
                          reg_lambda=1.0, subsample=1.0, random_state=seed, verbose=-1, n_jobs=-1,
                          deterministic=True, force_row_wise=True)


def stopped_iterations(pool, inner, key, seed):
    X, y, _ = rows(pool, key, False)
    Xv, yv, _ = rows([inner], key, False)
    model = hgb(seed, max_iter=600, early_stopping=True, n_iter_no_change=20, tol=1e-4)
    model.fit(X, y, X_val=Xv, y_val=yv)
    return model.n_iter_


class Fitted:
    def __init__(self, cfg, matches, key, seed):
        self.kind = cfg.get("model", "three")
        weighted = cfg.get("weighted", False)
        X, y, w = rows(matches, key, weighted)
        extra = cfg.get("_extra", {})
        if self.kind == "two_heads":
            self.dead = hgb(seed, **extra).fit(X, (y == 2).astype(np.int8), sample_weight=w)
            alive = y != 2
            self.team = hgb(seed, **extra).fit(X[alive], y[alive], sample_weight=w[alive])
        elif self.kind == "lightgbm":
            self.model = lgbm(seed).fit(X, y, sample_weight=w)
        else:
            self.model = hgb(seed, **extra).fit(X, y, sample_weight=w)

    def proba(self, X):
        if self.kind == "two_heads":
            dead = self.dead.predict_proba(X)[:, 1]
            home = self.team.predict_proba(X)[:, 0]
            return np.column_stack([(1 - dead) * home, (1 - dead) * (1 - home), dead])
        return self.model.predict_proba(X)


def log_p(p):
    return np.log(np.clip(p, 1e-4, 1.0)).astype(np.float32)


def fit_flips(matches, key, seed):
    def flips(y):
        alive = np.flatnonzero(y != 2)
        changed = alive[1:][y[alive[1:]] != y[alive[:-1]]]
        out = np.zeros(len(y), dtype=np.int8)
        for f in changed:
            out[max(0, f - CHANGE_WINDOW):f + CHANGE_WINDOW + 1] = 1
        return out
    s = train.TRAIN_STRIDE
    X = np.concatenate([m["X"][::s] for m in matches])
    y = np.concatenate([flips(m[key])[::s] for m in matches])
    return hgb(seed, max_iter=200).fit(X, y)


def train_arm(arm):
    cfg = ARMS[arm]
    key = "y_" + cfg.get("labels", "raw")
    gaps = cfg.get("gaps", "uniform")
    blocks = cfg.get("blocks", ("base",))
    folder = os.path.join(OUT, "oof", arm)
    os.makedirs(folder, exist_ok=True)
    tests = {g: {m: load_match(m, g, blocks) for m in idsse.MATCHES} for g in GAPS}
    matches = tests.get(gaps)
    train_gaps = GAPS if gaps == "mixed" else (gaps,)
    # A frozen arm fits on the still-ball matches and is tested on the moving ones.
    fit_root = FROZEN_OUT if cfg.get("frozen") else None
    fit_on = {g: {m: load_match(m, g, blocks, 0, fit_root) if fit_root else tests[g][m] for m in idsse.MATCHES}
              for g in train_gaps}
    extra_draws = [(m, g, load_match(m, g, blocks, d, fit_root)) for d in range(1, cfg.get("draws", 1))
                   for g in train_gaps for m in idsse.MATCHES]
    for f, held in enumerate(idsse.MATCHES):
        path = os.path.join(folder, held + ".npz")
        if os.path.isfile(path):
            continue
        t = time.time()
        pool = [fit_on[g][m] for g in train_gaps for m in idsse.MATCHES if m != held]
        pool += [x for m, _, x in extra_draws if m != held]
        fold_cfg = dict(cfg)
        extra = {}
        if cfg.get("early"):
            inner = idsse.MATCHES[(f + 1) % len(idsse.MATCHES)]
            rest = [matches[m] for m in idsse.MATCHES if m not in (held, inner)]
            extra = dict(max_iter=int(stopped_iterations(rest, matches[inner], key, train.SEED)))
        fold_cfg["_extra"] = extra
        models = [Fitted(fold_cfg, pool, key, train.SEED + s) for s in range(cfg.get("seeds", 1))]
        if cfg.get("self_train"):
            pseudo = dict(matches[held])
            lp = log_p(models[0].proba(pseudo["X"]))
            pseudo[key] = train.decode_periods(lp, pseudo["period"], 10.0, 40.0).astype(np.int8)
            models = [Fitted(fold_cfg, pool + [pseudo], key, train.SEED)]
        detector = fit_flips(pool, key, train.SEED) if cfg.get("change") else None
        out = dict(iterations=extra.get("max_iter", train.HGB["max_iter"]))
        for g in GAPS:
            test = tests[g][held]
            proba = np.mean([m.proba(test["X"]) for m in models], axis=0)
            if cfg.get("symmetric"):
                other = np.mean([m.proba(test["X_swap"]) for m in models], axis=0)[:, [1, 0, 2]]
                proba = (proba + other) / 2
            out["lp_" + g] = log_p(proba)
            if detector is not None:
                p = np.clip(detector.predict_proba(test["X"])[:, 1], 1e-4, 1 - 1e-4)
                out["logit_" + g] = np.log(p / (1 - p)).astype(np.float32)
        np.savez(path, **out)
        print(arm, held, "iters", out["iterations"], round(time.time() - t), flush=True)


# -- decoding and scoring ----------------------------------------------------------------------

@numba.njit(cache=True)
def viterbi_ha(em, ha, dead):
    n = em.shape[0]
    score = em[0].astype(np.float64)
    back = np.empty((n, 3), dtype=np.int8)
    new = np.empty(3)
    for i in range(1, n):
        for to in range(3):
            best, arg = -1e18, 0
            for fr in range(3):
                if fr == to:
                    t = 0.0
                elif fr == 2 or to == 2:
                    t = -dead
                else:
                    t = ha[i]
                c = score[fr] + t
                if c > best:
                    best, arg = c, fr
            back[i, to] = arg
            new[to] = best + em[i, to]
        score[:] = new
    path = np.empty(n, dtype=np.int8)
    path[n - 1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def decode(lp, logit, period, switch, gain, dead):
    ha = np.full(len(lp), -switch) if logit is None else np.minimum(-switch + gain * logit, 0.0)
    out = np.empty(len(lp), dtype=np.int8)
    for p in np.unique(period):
        r = np.flatnonzero(period == p)
        out[r] = viterbi_ha(lp[r].astype(np.float64), ha[r].astype(np.float64), float(dead))
    return out


def grid_job(args):
    arm, held, gap = args
    with np.load(os.path.join(OUT, "oof", arm, held + ".npz")) as f:
        lp = f["lp_" + gap]
        logit = f["logit_" + gap] if "logit_" + gap in f.files else None
    with np.load(cache_path(held, "uniform")) as f:
        truth = {k: f["y_" + k] for k in LABELS}
        period = f["period"]
    gains = GAIN_GRID if logit is not None else (0.0,)
    table = {}
    for s in SWITCH_GRID:
        for g in gains:
            for d in DEAD_GRID:
                path = decode(lp, logit, period, s, g, d)
                count = sum(1 for _, _, v in zip(*runs(path)) if v < 2)
                table[(s, g, d)] = [train.score(truth[k], path, period) for k in LABELS] + [count]
    return arm, held, gap, table


def score(arms):
    jobs = [(a, m, g) for a in arms for m in idsse.MATCHES for g in GAPS
            if os.path.isfile(os.path.join(OUT, "oof", a, m + ".npz"))]
    with ProcessPoolExecutor(6) as pool:
        done = list(pool.map(grid_job, jobs))
    results = {}
    for arm, held, gap, table in done:
        results.setdefault(arm, {}).setdefault(gap, {})[held] = table
    summary = {}
    for arm, by_gap in results.items():
        for gap, by_match in by_gap.items():
            for i, label in enumerate(LABELS):
                names = list(by_match)
                folds, macro, counts, picked = [], [], [], []
                for held in names:
                    others = [m for m in names if m != held]
                    best = max(by_match[held], key=lambda g: np.mean([by_match[m][g][i][0] for m in others]))
                    folds.append(by_match[held][best][i][0])
                    macro.append(by_match[held][best][i][1])
                    counts.append(by_match[held][best][2])
                    picked.append(best)
                summary["{}|{}|{}".format(arm, gap, label)] = dict(
                    spell=float(np.mean(folds)), macro=float(np.mean(macro)), folds=[round(float(v), 4) for v in folds],
                    count=float(np.mean(counts)), costs=picked)
    path = os.path.join(OUT, "summary.json")
    if os.path.isfile(path):
        with open(path) as handle:
            summary = dict(json.load(handle), **summary)
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=1)
    return summary


def report(summary):
    for gap in GAPS:
        for label in LABELS:
            ref = summary.get("ref|{}|{}".format(gap, label))
            print("\ngaps={} labels={}   arm, spellF1, macroF1, n, vs ref, folds up".format(gap, label))
            rows_ = [(k.split("|")[0], v) for k, v in summary.items() if k.endswith("|{}|{}".format(gap, label))]
            for arm, v in sorted(rows_, key=lambda kv: -kv[1]["spell"]):
                d = np.array(v["folds"]) - np.array(ref["folds"]) if ref else np.zeros(7)
                print("  {:<14} {:.4f} {:.4f} {:4.0f}  {:+.4f}  {}/7".format(
                    arm, v["spell"], v["macro"], v["count"], d.mean(), int(np.sum(d > 0))))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "prep":
        lengths, share, real = dose_exp.gap_profile()
        assert real
        jobs = [(i, m, g, lengths, share, d) for d in range(3) for i, m in enumerate(idsse.MATCHES) for g in GAPS]
        with ProcessPoolExecutor(4) as pool:
            for path in pool.map(prepare, jobs):
                print("ready", os.path.basename(path), flush=True)
    elif stage == "train":
        for arm in sys.argv[2:] or ARMS:
            train_arm(arm)
    elif stage == "score":
        report(score(sys.argv[2:] or list(ARMS)))
