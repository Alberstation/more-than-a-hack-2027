"""Are the short control-label possessions real? Two fast checks on clean IDSSE, no model. ~1 min.

    python possession/experiments/label_audit.py

1. Kinematics: for every ctrl possession, on the clean tracking (ball included), whether the team
   touches the ball (a jump in ball velocity with one of its players nearest and close), how much
   of the spell it is the nearest team, whether there is a duel, and the events around it.
   Possessions under 4 s are compared with longer ones; the start of a possession is checked in
   the same +-0.5 s window for every length, so the comparison is fair.
2. Clock: every event shifted by -1, -0.5, +0.5 and +1 s, ctrl rebuilt; how many short possessions
   appear or vanish, how many frames change, and how many of today's short ones survive.
Thresholds are fixed here from the literature (Vidal-Codina 2022, Link & Hoernig 2017) before any
output was seen. Understanding only: nothing is fitted or chosen from it.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from possession import control, idsse, train  # noqa: E402

FPS = 25
EDGES_S = (0, 2, 4, 8, 16, 1e9)
NAMES = ("<2 s", "2-4 s", "4-8 s", "8-16 s", ">16 s")

# Params
KICK_DV_MS = 2.0            # ball velocity change over +-KICK_HALF frames that counts as a touch
KICK_HALF = 3
TOUCH_RADIUS_M = 1.5        # the nearest player must be this close for the touch to be his team's
DUEL_RADIUS_M = 1.0         # players of both teams this close to the ball: a duel
START_S = 0.5               # window around the start of a possession for "touch at the gain"
SHIFTS_S = (-1.0, -0.5, 0.5, 1.0)


def spells(y):
    start, end, value = control.runs(y)
    keep = value < 2
    return start[keep], end[keep], value[keep]


def bucket(frames):
    return int(np.searchsorted(EDGES_S, frames / FPS, side="right") - 1)


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def kinematics(data):
    """Per frame: nearest team (-1 unknown), touch team (-1 none), duel flag."""
    P, B, team = data["players"][:, :, :2].astype(np.float64), data["ball"][:, :2].astype(np.float64), data["team"]
    d = np.hypot(P[:, :, 0] - B[:, None, 0], P[:, :, 1] - B[:, None, 1])
    filled = np.where(np.isnan(d), np.inf, d)
    slot = np.argmin(filled, axis=1)
    close = filled[np.arange(len(B)), slot]
    nearest = np.where(np.isinf(close), -1, team[slot])
    v = np.full(B.shape, np.nan)
    v[1:-1] = (B[2:] - B[:-2]) * FPS / 2
    dv = np.full(len(B), 0.0)
    k = KICK_HALF
    dv[k:-k] = np.nan_to_num(np.hypot(*(v[2 * k:] - v[:-2 * k]).T))
    touch = np.where((dv >= KICK_DV_MS) & (close <= TOUCH_RADIUS_M), nearest, -1)
    near = filled <= DUEL_RADIUS_M
    duel = near[:, team == 0].any(axis=1) & near[:, team == 1].any(axis=1)
    return nearest, touch, duel


def audit_job(match_id):
    data = idsse.load(match_id)
    y = np.array([train.CODE[v] for v in data["labels"]], dtype=np.int8)
    period = data["period"]
    frames, owners = control.event_frames(match_id, y, period)
    ctrl = control.relabel(y, frames, owners)
    nearest, touch, duel = kinematics(data)
    w0, ev = int(START_S * FPS), int(control.CONTROL_WINDOW_S * FPS)
    rows = []
    for s, e, v in zip(*spells(ctrl)):
        lo, hi = max(0, s - w0), min(len(y), e + w0)
        seen = nearest[s:e] >= 0
        near_ev = (frames >= s - ev) & (frames < e + ev)
        rows.append(dict(
            bucket=bucket(e - s),
            start_touch=bool(np.any(touch[max(0, s - w0):s + w0] == v)),
            start_opp=bool(np.any(touch[max(0, s - w0):s + w0] == 1 - v)),
            own_touch=bool(np.any(touch[lo:hi] == v)),
            near_share=float(np.mean(nearest[s:e][seen] == v)) if seen.any() else np.nan,
            duel=bool(duel[s:e].any()),
            own_events=int(np.sum(owners[near_ev] == v)),
            opp_events=int(np.sum(owners[near_ev] != v))))
    # Clock: rebuild ctrl with every event shifted.
    base = list(zip(*spells(ctrl)))
    short = [(s, e, v) for s, e, v in base if e - s < 4 * FPS]
    clock = {}
    for shift in SHIFTS_S:
        moved = control.relabel(y, np.clip(frames + int(round(shift * FPS)), 0, len(y) - 1), owners)
        found = list(zip(*spells(moved)))
        lengths = np.array([e - s for s, e, _ in found])
        survive = [any(fv == v and fe > s and fs < e and iou((s, e), (fs, fe)) >= 0.5 for fs, fe, fv in found)
                   for s, e, v in short]
        clock[shift] = dict(changed=float(np.mean(moved != ctrl)),
                            n2=int(np.sum(lengths < 2 * FPS)), n4=int(np.sum((lengths >= 2 * FPS) & (lengths < 4 * FPS))),
                            n=len(found), survive=float(np.mean(survive)) if short else np.nan)
    base_lengths = np.array([e - s for s, e, _ in base])
    clock[0.0] = dict(changed=0.0, n2=int(np.sum(base_lengths < 2 * FPS)),
                      n4=int(np.sum((base_lengths >= 2 * FPS) & (base_lengths < 4 * FPS))), n=len(base), survive=1.0)
    return rows, clock


def main():
    with ProcessPoolExecutor(7) as pool:
        done = list(pool.map(audit_job, idsse.MATCHES))
    rows = [r for d, _ in done for r in d]
    print("1. Cinematica de las posesiones ctrl (tracking limpio de IDSSE, 7 partidos)")
    print("   duracion    n   toque al inicio propio / rival   toque propio   cercania media   duelo   eventos propios / rivales")
    for b, name in enumerate(NAMES):
        r = [x for x in rows if x["bucket"] == b]
        if r:
            print("   {:<7}  {:4d}   {:>16.0%} / {:<12.0%}   {:12.0%}   {:14.2f}   {:5.0%}   {:.1f} / {:.1f}".format(
                name, len(r), np.mean([x["start_touch"] for x in r]), np.mean([x["start_opp"] for x in r]),
                np.mean([x["own_touch"] for x in r]),
                np.nanmean([x["near_share"] for x in r]), np.mean([x["duel"] for x in r]),
                np.mean([x["own_events"] for x in r]), np.mean([x["opp_events"] for x in r])))
    print("\n   posesiones < 4 s por tipo:")
    short = [x for x in rows if x["bucket"] <= 1]
    kind = lambda x: ("sin toque propio" if not x["own_touch"] else "duelo" if x["duel"]
                      else "control (equipo mas cercano)" if x["near_share"] >= 0.5 else "toque, rival mas cercano")
    for b, name in enumerate(NAMES[:2]):
        r = [kind(x) for x in short if x["bucket"] == b]
        print("     {:<6} ".format(name) + "  ".join("{} {:.0%}".format(k, r.count(k) / len(r)) for k in
              ("control (equipo mas cercano)", "duelo", "toque, rival mas cercano", "sin toque propio")))
    print("\n2. Sensibilidad de ctrl al reloj de eventos (suma de 7 partidos)")
    print("   desfase   posesiones   < 2 s   2-4 s   frames cambiados   cortas de hoy que sobreviven (IoU >= 0,5)")
    for shift in sorted(done[0][1]):
        c = [d[shift] for _, d in done]
        print("   {:+5.1f} s   {:10d}   {:5d}   {:5d}   {:16.2%}   {:.0%}".format(
            shift, sum(x["n"] for x in c), sum(x["n2"] for x in c), sum(x["n4"] for x in c),
            np.mean([x["changed"] for x in c]), np.mean([x["survive"] for x in c])))


if __name__ == "__main__":
    main()
