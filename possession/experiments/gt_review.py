"""What the public five minutes say about our assumptions. Understanding only, nothing is chosen here.

    python possession/experiments/gt_review.py

Compares the shape of the public labels with IDSSE raw and control labels, describes where v3 and
the ground truth differ, and checks where the ball is missing. No constant, version or threshold is
picked from this output (RULES.md). Prints aggregates only: no frame of the challenge leaves this PC.
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import half_video_blocks  # noqa: E402
from possession import atd, features, idsse, train  # noqa: E402
import decoder_exp  # noqa: E402
import improve  # noqa: E402

CODE = {"H": 0, "A": 1, "D": 2, "X": -1}
FPS = 25
WINDOW = 5 * 60 * FPS
STEP = 60 * FPS
NEAR_S = 5


def read_codes(path):
    with open(path) as handle:
        return np.array([CODE[row["label"]] for row in csv.DictReader(handle)], dtype=np.int8)


def shape(y):
    """Label statistics of one contiguous stretch."""
    start, end, value = improve.runs(y)
    seconds = (end - start) / FPS
    alive = value < 2
    pairs = [(a, b) for a, b in zip(value[:-1], value[1:]) if a < 2 and b < 2]
    alive_min = np.mean(y < 2) * len(y) / FPS / 60
    return dict(n=int(alive.sum()), mean=seconds[alive].mean(), median=np.median(seconds[alive]),
                lt2=np.mean(seconds[alive] < 2), lt5=np.mean(seconds[alive] < 5),
                per_min=alive.sum() / alive_min, direct_min=len(pairs) / alive_min,
                dead=np.mean(y == 2), dead_n=int((~alive).sum()),
                dead_mean=seconds[~alive].mean() if (~alive).any() else 0.0)


def windows(y):
    return [shape(y[a:a + WINDOW]) for a in range(0, len(y) - WINDOW, STEP)
            if np.any(y[a:a + WINDOW] < 2) and np.any(y[a:a + WINDOW] == 2)]


def changes(y):
    """(frame, kind) of label changes; kind 0 H<->A, 1 alive->dead, 2 dead->alive."""
    at = np.flatnonzero(np.diff(y)) + 1
    kind = np.where(y[at] == 2, 1, np.where(y[at - 1] == 2, 2, 0))
    return at, kind


def offsets(truth, pred):
    """Signed lateness in seconds of the nearest same-kind predicted change, per kind."""
    t_at, t_kind = changes(truth)
    p_at, p_kind = changes(pred)
    out = {}
    for kind in (0, 1, 2):
        lag, missed = [], 0
        for f in t_at[t_kind == kind]:
            near = p_at[p_kind == kind] - f
            near = near[np.abs(near) <= NEAR_S * FPS]
            if len(near):
                lag.append(near[np.argmin(np.abs(near))] / FPS)
            else:
                missed += 1
        out[kind] = (lag, missed)
    return out


def show_offsets(name, parts):
    for kind, title in ((0, "H<->A"), (1, "alive->dead"), (2, "dead->alive")):
        lag = np.concatenate([np.array(p[kind][0], dtype=np.float64) for p in parts])
        missed = sum(p[kind][1] for p in parts)
        total = len(lag) + missed
        if len(lag):
            print("  {:<22} {:<12} n {:4d} found {:.2f}  median lag {:+.2f} s  |lag|<0.5 s {:.2f}".format(
                name, title, total, len(lag) / total, np.median(lag), np.mean(np.abs(lag) < 0.5)))


def spell_review(truth, pred):
    """Each true spell: best same-team IoU and what the prediction said during it."""
    start, end, value = improve.runs(truth)
    rows = []
    for a, b, v in zip(start, end, value):
        if v > 1:
            continue
        inside = pred[a:b]
        same, other, dead = np.mean(inside == v), np.mean(inside == 1 - v), np.mean(inside == 2)
        best = 0.0
        p_start, p_end, p_value = improve.runs(pred)
        for c, d, w in zip(p_start, p_end, p_value):
            if w == v and d > a and c < b:
                best = max(best, (min(b, d) - max(a, c)) / (max(b, d) - min(a, c)))
        rows.append((best, same, other, dead, (b - a) / FPS))
    return np.array(rows)


def lateness_to_ball(y, players, ball, team):
    """Seconds from the nearest-team switch to the label change, for H<->A changes with ball seen."""
    d_home, d_away = features.team_distances(players, ball, team)
    seen = ~(np.isnan(d_home) | np.isnan(d_away))
    near = np.where(d_home <= d_away, 0, 1)
    at, kind = changes(y)
    out = []
    for f in at[kind == 0]:
        new = y[f]
        a, b = max(0, f - 3 * FPS), min(len(y), f + 3 * FPS)
        hits = np.flatnonzero(seen[a:b] & (near[a:b] == new) & (np.fmin(d_home, d_away)[a:b] < 2.0))
        if len(hits):
            out.append((a + hits[np.argmin(np.abs(a + hits - f))] - f) / FPS)
    return out


def main():
    truth = read_codes(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    shown = np.flatnonzero(truth >= 0)
    a, b = shown[0], shown[-1] + 1
    gt = truth[a:b]

    print("\n== A. Label shape: public window vs 5-minute windows of IDSSE (p = share of windows below it)")
    pool = {"raw": [], "ctrl": []}
    whole = {"raw": [], "ctrl": []}
    for path in sorted(glob.glob(os.path.join(improve.OUT, "matches", "*_uniform.npz"))):
        with np.load(path) as f:
            for key in pool:
                for p in np.unique(f["period"]):
                    y = f["y_" + key][f["period"] == p]
                    pool[key] += windows(y)
                    whole[key].append(shape(y))
    g = shape(gt)
    print("  public:", {k: round(float(v), 3) for k, v in g.items()})
    for key in pool:
        print("  whole halves", key, {k: round(float(np.mean([s[k] for s in whole[key]])), 3) for k in g})
    for name in ("mean", "median", "lt2", "lt5", "per_min", "direct_min", "dead", "dead_mean"):
        line = []
        for key in pool:
            v = np.array([s[name] for s in pool[key]])
            line.append("{} p{:.0f} [{:.2f} {:.2f} {:.2f}]".format(
                key, 100 * np.mean(v < g[name]), *np.percentile(v, [10, 50, 90])))
        print("  {:<11} public {:6.2f}   {}".format(name, g[name], "   ".join(line)))

    print("\n== B. Where v3 and v2 differ from the public labels")
    for version in ("v3", "v2"):
        pred = read_codes(os.path.join(ROOT, "evaluation", "results", "possession_{}.csv".format(version)))[a:b]
        s = shape(pred)
        print(" ", version, "spells", s["n"], "mean", round(s["mean"], 1), "dead", round(s["dead"], 3),
              "dead segments", s["dead_n"], "frame agreement", round(float(np.mean(pred == gt)), 3))
        confusion = np.array([[np.sum((gt == i) & (pred == j)) for j in range(3)] for i in range(3)])
        print("  confusion rows true HAD, share of row", np.round(confusion / confusion.sum(1, keepdims=True), 3).tolist())
        rows = spell_review(gt, pred)
        print("  true spells {}: IoU>=.7 {} .5-.7 {} .3-.5 {} <.3 {}; mean share same {:.2f} other {:.2f} dead {:.2f}".format(
            len(rows), int(np.sum(rows[:, 0] >= .7)), int(np.sum((rows[:, 0] >= .5) & (rows[:, 0] < .7))),
            int(np.sum((rows[:, 0] >= .3) & (rows[:, 0] < .5))), int(np.sum(rows[:, 0] < .3)),
            *rows[:, 1:4].mean(0)))
        show_offsets(version + " vs public", [offsets(gt, pred)])

    print("\n  Same lateness in IDSSE, v3 out-of-fold on both gap patterns, costs (5, 40)")
    parts = {"raw": [], "ctrl": []}
    for held in idsse.MATCHES:
        for gap in improve.GAPS:
            lp, _, y_raw, y_ctrl, period = decoder_exp.load_oof("mixed_all3", held, gap)
            path = improve.decode(lp, None, period, 5.0, 0.0, 40.0)
            for p in np.unique(period):
                r = period == p
                parts["raw"].append(offsets(y_raw[r], path[r]))
                parts["ctrl"].append(offsets(y_ctrl[r], path[r]))
    for key in parts:
        show_offsets("v3 vs IDSSE " + key, parts[key])

    print("\n== C. Where the ball is missing")
    players, ball, team = atd.load()
    missing = np.isnan(ball[:, 0])
    m = missing[a:b]
    print("  window: missing {:.3f}; given dead {:.3f}; given alive {:.3f}; share of missing frames in dead {:.3f}".format(
        m.mean(), m[gt == 2].mean(), m[gt < 2].mean(), np.mean(gt[m] == 2)))
    graded = np.zeros(len(ball), dtype=bool)
    for _, first, last in half_video_blocks("BAR-ATH"):
        graded[first:last + 1] = True
    print("  whole match graded: missing {:.3f}; assumed P(gap | alive) in dead mask {}".format(
        missing[graded].mean(), train.GAP_GIVEN_ALIVE))
    edges = np.diff(np.r_[0, m.astype(np.int8), 0])
    lengths = (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) / FPS
    print("  gaps in window: {} median {:.1f} s max {:.1f} s".format(len(lengths), np.median(lengths), lengths.max()))
    at, kind = changes(gt)
    around = np.array([m[max(0, f - 12):f + 13].mean() for f in at[kind == 0]])
    print("  H<->A changes {}; ball missing share within +-0.5 s {:.2f}; fully hidden {}".format(
        len(around), around.mean(), int(np.sum(around == 1))))

    v3 = read_codes(os.path.join(ROOT, "evaluation", "results", "possession_v3.csv"))
    w = v3[a:b]
    print("  window by v3 labels: given dead {:.3f}; given alive {:.3f}; share of missing frames in dead {:.3f}".format(
        m[w == 2].mean(), m[w < 2].mean(), np.mean(w[m] == 2)))
    g_v3, g_miss = v3[graded], missing[graded]
    print("  match by v3 labels: dead share {:.3f}; given dead {:.3f}; given alive {:.3f}; share of missing in dead {:.3f}".format(
        np.mean(g_v3 == 2), g_miss[g_v3 == 2].mean(), g_miss[g_v3 < 2].mean(), np.mean(g_v3[g_miss] == 2)))
    bins, dead_bins = [], []
    for _, first, last in half_video_blocks("BAR-ATH"):
        for s in range(first, last + 1 - WINDOW, WINDOW):
            bins.append(missing[s:s + WINDOW].mean())
            dead_bins.append(np.mean(v3[s:s + WINDOW] == 2))
    bins = np.array(bins)
    print("  5-minute blocks of the match, missing share: min {:.2f} p25 {:.2f} median {:.2f} p75 {:.2f} max {:.2f}; "
          "public window rank {}/{}; first block of each half {}".format(
              *np.percentile(bins, [0, 25, 50, 75, 100]), int(np.sum(bins < m.mean())) + 1, len(bins),
              np.round(bins[[0, len(bins) // 2]], 2).tolist()))
    print("  correlation of block missing share with v3 dead share {:.2f}".format(np.corrcoef(bins, dead_bins)[0, 1]))

    print("\n== C2. Boundaries and dead ball, v3 on the public window vs v3 out-of-fold on IDSSE")
    lags = offsets(gt, w)
    for kind, title in ((0, "H<->A"), (1, "alive->dead"), (2, "dead->alive")):
        print("  public {:<12} sorted lags s {}".format(title, np.round(np.sort(lags[kind][0]), 1).tolist()))
    for key in parts:
        for kind, title in ((0, "H<->A"), (1, "alive->dead"), (2, "dead->alive")):
            lag = np.concatenate([np.array(p[kind][0], dtype=np.float64) for p in parts[key]])
            print("  IDSSE {:<4} {:<12} |lag| median {:.2f} p75 {:.2f}; late share {:.2f}".format(
                key, title, np.median(np.abs(lag)), np.percentile(np.abs(lag), 75), np.mean(lag > 0)))
    recall, shares = [], []
    for held in idsse.MATCHES:
        for gap in improve.GAPS:
            lp, _, y_raw, _, period = decoder_exp.load_oof("mixed_all3", held, gap)
            path = improve.decode(lp, None, period, 5.0, 0.0, 40.0)
            recall.append(np.mean(path[y_raw == 2] == 2))
            shares.append(np.mean(path == 2) / np.mean(y_raw == 2))
    print("  IDSSE dead recall {:.3f} (min {:.3f}), predicted/true dead share {:.3f}; public recall {:.3f}, ratio {:.3f}".format(
        np.mean(recall), np.min(recall), np.mean(shares), np.mean(w[gt == 2] == 2), np.mean(w == 2) / np.mean(gt == 2)))

    print("\n== D. Label change vs nearest-team switch (seconds, negative = ball reached new team first)")
    lag = lateness_to_ball(gt, players[a:b], ball[a:b], team)
    print("  public n {} median {:+.2f}  p25 {:+.2f}  p75 {:+.2f}".format(len(lag), *np.percentile(lag, [50, 25, 75])))
    pooled = {"raw": [], "ctrl": []}
    for held in idsse.MATCHES:
        data = idsse.load(held)
        with np.load(improve.cache_path(held, "uniform")) as f:
            for key in pooled:
                pooled[key] += lateness_to_ball(f["y_" + key], data["players"], data["ball"], data["team"])
    for key in pooled:
        print("  IDSSE clean {} n {} median {:+.2f}  p25 {:+.2f}  p75 {:+.2f}".format(
            key, len(pooled[key]), *np.percentile(pooled[key], [50, 25, 75])))


if __name__ == "__main__":
    main()
