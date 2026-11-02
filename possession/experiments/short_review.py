"""Short possessions: how many the labels have, how many v5 finds, and who loses them. LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/short_review.py

1. Spell lengths: public window, IDSSE raw and IDSSE control labels.
2. v5 recall by true spell length (IoU >= 0.5, same team): IDSSE out of fold against control
   labels, and the public window.
3. For true spells v5 misses: mean probability of the true team before decoding. High means the
   decoder's switch cost removed it; low means the model never saw it.
Understanding only; nothing is chosen here (RULES.md). Aggregates only.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import idsse, v5  # noqa: E402
import error_review as er  # noqa: E402
import gt_review  # noqa: E402
import improve  # noqa: E402

FPS = 25
EDGES_S = (0, 2, 4, 8, 16, 1e9)
NAMES = ("<2 s", "2-4 s", "4-8 s", "8-16 s", ">16 s")


def bucket(seconds):
    return int(np.searchsorted(EDGES_S, seconds, side="right") - 1)


def lengths(y):
    s, e, _ = er.spells(y)
    return (e - s) / FPS


def describe(name, secs):
    counts = np.bincount([bucket(v) for v in secs], minlength=5)
    print("   {:<16} n {:5d}  ".format(name, len(secs)) + "  ".join(
        "{} {:4.0%}".format(k, c / len(secs)) for k, c in zip(NAMES, counts)))


def recall(y, path, p):
    """Per true spell: (bucket, found, mean p of the true team over the spell)."""
    found = list(zip(*er.spells(path)))
    out = []
    for s, e, v in zip(*er.spells(y)):
        best = max([er.iou((s, e), (fs, fe)) for fs, fe, fv in found if fv == v and fe > s and fs < e], default=0)
        out.append((bucket((e - s) / FPS), best >= 0.5, float(p[s:e, v].mean())))
    return out


def table(name, rows):
    rows = np.array(rows, dtype=float)
    print("   {}".format(name))
    print("     duracion  reales  encontradas  p(equipo real) en las perdidas")
    for b, k in enumerate(NAMES):
        r = rows[rows[:, 0] == b]
        if len(r):
            lost = r[r[:, 1] == 0]
            print("     {:<8}  {:6d}  {:10.0%}  {}".format(
                k, len(r), r[:, 1].mean(), "{:.2f} (n {})".format(lost[:, 2].mean(), len(lost)) if len(lost) else "-"))


def by_ball():
    """5. v5 recall on IDSSE by spell length and by the share of the spell with the ball seen."""
    seen_names = ("balon < 20 %", "20-80 %", "> 80 %")
    for g in improve.GAPS:
        cells = {}
        for m in idsse.MATCHES:
            with np.load(improve.cache_path(m, "uniform")) as f:
                y, period = f["y_ctrl"], f["period"]
            with np.load(improve.cache_path(m, g)) as f:
                seen = f["base"][:, 0] > 0.5          # features.build puts ball_seen first
            with np.load(os.path.join(improve.OUT, "oof", "ctrl_mixed_all3", m + ".npz")) as f:
                p_hgb = np.exp(f["lp_" + g].astype(np.float64))
            p_enc = []
            for arm in er.ENCODERS:
                with np.load(os.path.join(improve.OUT, "oof", arm, m + ".npz")) as f:
                    p_enc.append(np.exp(f["lp_" + g].astype(np.float64)))
            p = (1 - v5.ENCODER_WEIGHT) * p_hgb + v5.ENCODER_WEIGHT * np.mean(p_enc, axis=0)
            path = improve.decode(np.log(np.clip(p, 1e-4, 1)), None, period, v5.SWITCH_COST, 0.0, v5.DEAD_COST)
            for (b, ok, pt), (s, e, _) in zip(recall(y, path, p), zip(*er.spells(y))):
                share = seen[s:e].mean()
                k = 0 if share < 0.2 else (1 if share <= 0.8 else 2)
                cells.setdefault((b, k), []).append((ok, pt))
        print("   huecos {}: encontradas (n) [p equipo real]".format(g))
        print("     duracion  " + "  ".join("{:>22}".format(n) for n in seen_names))
        for b, name in enumerate(NAMES):
            row = []
            for k in range(3):
                c = np.array(cells.get((b, k), []), dtype=float)
                row.append("{:>4.0%} ({:4d}) [{:.2f}]".format(c[:, 0].mean(), len(c), c[:, 1].mean()) if len(c) else " " * 18 + "-")
            print("     {:<8}  ".format(name) + "  ".join("{:>22}".format(r) for r in row))


def main():
    if sys.argv[1:2] == ["ball"]:
        print("5. v5 en IDSSE, por duracion y por parte de la posesion con balon visible")
        by_ball()
        return
    print("1. Duracion de las posesiones")
    gt = gt_review.read_codes(er.GT)
    window = np.flatnonzero(gt >= 0)
    describe("ventana publica", lengths(gt[window]))
    raw, ctrl = [], []
    for m in idsse.MATCHES:
        with np.load(improve.cache_path(m, "uniform")) as f:
            raw += list(lengths(f["y_raw"]))
            ctrl += list(lengths(f["y_ctrl"]))
    describe("IDSSE raw", raw)
    describe("IDSSE ctrl", ctrl)

    print("\n2-3. v5: posesiones reales encontradas por duracion")
    for g in improve.GAPS:
        rows = []
        for m in idsse.MATCHES:
            with np.load(improve.cache_path(m, "uniform")) as f:
                y, period = f["y_ctrl"], f["period"]
            with np.load(os.path.join(improve.OUT, "oof", "ctrl_mixed_all3", m + ".npz")) as f:
                p_hgb = np.exp(f["lp_" + g].astype(np.float64))
            p_enc = []
            for arm in er.ENCODERS:
                with np.load(os.path.join(improve.OUT, "oof", arm, m + ".npz")) as f:
                    p_enc.append(np.exp(f["lp_" + g].astype(np.float64)))
            p = (1 - v5.ENCODER_WEIGHT) * p_hgb + v5.ENCODER_WEIGHT * np.mean(p_enc, axis=0)
            path = improve.decode(np.log(np.clip(p, 1e-4, 1)), None, period, v5.SWITCH_COST, 0.0, v5.DEAD_COST)
            rows += recall(y, path, p)
        table("IDSSE fuera de pliegue, huecos {}, contra ctrl".format(g), rows)
    p, pred = er.challenge()
    table("ventana publica", recall(gt[window], pred[window], p[window]))
    need = 2 * v5.SWITCH_COST
    print("\n4. Coste del decodificador: una posesion de t s necesita {:.0f} nats de evidencia, "
          "{:.2f} por frame a 1 s, {:.2f} a 2 s, {:.2f} a 4 s".format(need, need / 25, need / 50, need / 100))


if __name__ == "__main__":
    main()
