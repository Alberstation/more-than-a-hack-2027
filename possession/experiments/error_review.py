"""Why v5 gains on IDSSE but not on the public five minutes. Understanding only; nothing is chosen.

    LIVE_DEAD=1 python possession/experiments/error_review.py

1. Public window: every true possession, whether v5 finds it, and if not why (merged with a
   neighbour, wrong team, called dead, boundary); true stoppages found; frame confusion.
2. Possessions per live minute: v5 on the whole match, against the public window and IDSSE.
3. The same count on IDSSE out of fold, against its control labels: does v5 merge there too?
4. Confidence: how sure the mixed probabilities are on the challenge and on IDSSE.
No constant, version or threshold is picked from this output (RULES.md). Aggregates only.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import idsse, pipeline, v5  # noqa: E402
import gt_review  # noqa: E402
import improve  # noqa: E402

FPS = 25
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
CACHE = os.path.join(improve.OUT, "challenge_v5_proba.npz")
ENCODERS = ("encoder", "encoder_s2028", "encoder_s2029")


def spells(y):
    start, end, value = improve.runs(y)
    keep = value < 2
    return start[keep], end[keep], value[keep]


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def per_minute(y):
    """Possessions per minute of live ball."""
    return len(spells(y)[0]) / (np.sum(y < 2) / FPS / 60)


def challenge():
    if not os.path.isfile(CACHE):
        p_hgb, p_enc, _ = pipeline.proba_v5()
        np.savez(CACHE, p_hgb=p_hgb, p_enc=p_enc)
    with np.load(CACHE) as f:
        p = (1 - v5.ENCODER_WEIGHT) * f["p_hgb"] + v5.ENCODER_WEIGHT * f["p_enc"]
    labels = pipeline.predict_v5()
    pred = np.array([{"H": 0, "A": 1, "D": 2, "X": -1}[v] for v in labels], dtype=np.int8)
    return p, pred


def public_window(pred, p):
    gt = gt_review.read_codes(GT)
    window = np.flatnonzero(gt >= 0)
    y, z = gt[window], pred[window]
    true, found = list(zip(*spells(y))), list(zip(*spells(z)))
    print("\n1. Ventana publica: {} posesiones reales, {} predichas".format(len(true), len(found)))
    print("   inicio_s  dur_s  equipo  IoU   causa")
    causes = {}
    for s, e, v in true:
        same = [(fs, fe) for fs, fe, fv in found if fv == v and fe > s and fs < e]
        best = max([iou((s, e), q) for q in same], default=0.0)
        share = np.bincount(z[s:e], minlength=3) / (e - s)
        if best >= 0.5:
            cause = "ok"
        elif share[2] > 0.5:
            cause = "llamada muerta"
        elif share[1 - v] > 0.5:
            cause = "equipo cambiado"
        elif any(sum(1 for ts, te, tv in true if tv == v and te > fs and ts < fe) > 1 for fs, fe in same):
            cause = "fusionada con otra"
        else:
            cause = "frontera"
        causes[cause] = causes.get(cause, 0) + 1
        print("   {:7.1f}  {:5.1f}  {}       {:.2f}  {}".format(s / FPS, (e - s) / FPS, "HA"[v], best, cause))
    print("   resumen:", causes)
    start, end, value = improve.runs(y)
    dead = [(s, e) for s, e, v in zip(start, end, value) if v == 2]
    hit = sum(np.mean(z[s:e] == 2) > 0.5 for s, e in dead)
    print("   paradas reales: {}, detectadas (>50 % de sus frames en D): {}".format(len(dead), hit))
    conf = np.zeros((3, 3), dtype=int)
    np.add.at(conf, (y, z), 1)
    print("   confusion (filas real H/A/D, columnas predicha), en segundos:")
    for k, row in zip("HAD", conf):
        print("     {}  {}".format(k, "  ".join("{:6.1f}".format(v / FPS) for v in row)))
    print("   posesiones por minuto vivo: real {:.2f}, v5 {:.2f}".format(per_minute(y), per_minute(z)))
    print("   confianza media (max p) en la ventana: {:.3f}".format(p[window].max(axis=1).mean()))
    return per_minute(y)


def idsse_counts():
    rows = {g: [] for g in improve.GAPS}
    conf = {g: [] for g in improve.GAPS}
    for m in idsse.MATCHES:
        with np.load(improve.cache_path(m, "uniform")) as f:
            y, period = f["y_ctrl"], f["period"]
        for g in improve.GAPS:
            with np.load(os.path.join(improve.OUT, "oof", "ctrl_mixed_all3", m + ".npz")) as f:
                p_hgb = np.exp(f["lp_" + g].astype(np.float64))
            p_enc = []
            for arm in ENCODERS:
                with np.load(os.path.join(improve.OUT, "oof", arm, m + ".npz")) as f:
                    p_enc.append(np.exp(f["lp_" + g].astype(np.float64)))
            p = (1 - v5.ENCODER_WEIGHT) * p_hgb + v5.ENCODER_WEIGHT * np.mean(p_enc, axis=0)
            path = improve.decode(np.log(np.clip(p, 1e-4, 1)), None, period, v5.SWITCH_COST, 0.0, v5.DEAD_COST)
            rows[g].append((per_minute(y), per_minute(path)))
            conf[g].append(p.max(axis=1).mean())
    return rows, conf


def main():
    p, pred = challenge()
    public_rate = public_window(pred, p)
    graded = pred >= 0
    print("\n2. Partido completo: v5 {:.2f} posesiones por minuto vivo (ventana publica real {:.2f})".format(
        per_minute(pred[graded]), public_rate))
    rows, conf = idsse_counts()
    print("\n3. IDSSE fuera de pliegue, v5 frente a ctrl (posesiones por minuto vivo):")
    for g in improve.GAPS:
        r = np.array(rows[g])
        print("   huecos {:<7}  ctrl {:.2f}  v5 {:.2f}  (ratio {:.2f})".format(g, r[:, 0].mean(), r[:, 1].mean(),
                                                                          r[:, 1].mean() / r[:, 0].mean()))
    print("\n4. Confianza media (max p):  reto {:.3f}  |  IDSSE uniforme {:.3f}  muerto {:.3f}".format(
        p[graded].max(axis=1).mean(), np.mean(conf["uniform"]), np.mean(conf["dead"])))
    print("   frames con max p < 0,6:  reto {:.1%}".format(np.mean(p[graded].max(axis=1) < 0.6)))


if __name__ == "__main__":
    main()
