"""SoccerTrack v2 (Scott et al., arXiv 2508.01802, CC BY 4.0): 10 university matches as an unseen provider.

    python possession/experiments/soccertrack_data.py            # parse and cache every match, print label stats

Source: the provider's raw XML (raw/<match>/<match>_tracker_box_data.xml, revision v1.2), 25 fps,
positions normalised 0-1 with two decimals (a 1.05 x 0.68 m grid), and a per-frame `ballStatus`
of HOME / AWAY / NEUTRAL / BALLOUT. Only FIRST_HALF and SECOND_HALF are used (the files also hold
an EXTRA_FIRST_HALF that is not part of the release). M9 and M10 (132831, 132877) carry no
possession (every frame INPLAY) and are left out: 8 matches.

Rules, fixed before any model was scored on this data:
- Positions are smoothed with a Gaussian of SMOOTH_S per player track to undo the 1 m grid;
  speeds are the provider's own (`speed`, m/s).
- Labels: HOME -> 0, AWAY -> 1, BALLOUT -> 2; NEUTRAL (loose ball) stays with the last team in
  possession, or the next one when the live stretch opens with it.
- `ctrl`: the same control rule as IDSSE and Metrica (control.relabel, 1.5 s window), with the
  ball-action events (bas/) that play the ball: PLAY_EVENTS. Each half's events are placed on the
  raw frames with the lag (|lag| <= MAX_LAG frames) that best matches their team to the label,
  and left/right is mapped to home/away by the same agreement.
- The ball is a reconstruction anchored at those events, so models see it only where stated.
External data only.
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402
from possession import control  # noqa: E402

FOLDER = os.path.join(DATA_DIR, "external", "soccertrack_v2")
CACHE = os.path.join(FOLDER, "cache")
NO_STATUS = ("132831", "132877")   # M9 and M10: every frame INPLAY, no provider possession
MATCHES = sorted(m for m in os.listdir(os.path.join(FOLDER, "raw")) if m not in NO_STATUS) if os.path.isdir(
    os.path.join(FOLDER, "raw")) else []
PITCH = (105.0, 68.0)
FPS = 25
PERIODS = {"FIRST_HALF": 1, "SECOND_HALF": 2}
STATUS = {"HOME": 0, "AWAY": 1, "BALLOUT": 2, "NEUTRAL": -1, "INPLAY": -9}
PLAY_EVENTS = ("PASS", "HIGH PASS", "CROSS", "SHOT", "DRIVE", "THROW IN", "FREE KICK")
SMOOTH_S = 0.4
MAX_LAG = 75
CACHE_VERSION = 1

FRAME = re.compile(r'<frame matchTime="(\d+)" frameNumber="(\d+)" eventPeriod="(\w+)" ballStatus="(\w+)"')
OBJ = re.compile(r'<(player|ball) playerId="(\w+)" loc="\[([^,\]]+), ([^\]]+)\]" speed="([^"]+)"')


def _num(text):
    try:
        return float(text)
    except ValueError:
        return np.nan


def metadata(match):
    root = ET.parse(os.path.join(FOLDER, "raw", match, match + "_tracker_box_metadata.xml")).getroot()
    home = [t.get("id") for t in root.iter("team") if t.get("side") == "home"][0]
    ids = [p.get("id") for p in root.iter("player")]
    team = np.array([0 if p.get("teamId") == home else 1 for p in root.iter("player")], dtype=np.int8)
    return ids, team


def parse(match):
    """Raw arrays on the source frames of the two halves."""
    ids, team = metadata(match)
    col = {i: k for k, i in enumerate(ids)}
    period, mtime, status, players, ball = [], [], [], [], []
    keep = False
    row = brow = None
    path = os.path.join(FOLDER, "raw", match, match + "_tracker_box_data.xml")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            m = FRAME.search(line)
            if m:
                keep = m.group(3) in PERIODS
                if keep:
                    period.append(PERIODS[m.group(3)])
                    mtime.append(int(m.group(1)))
                    status.append(STATUS[m.group(4)])
                    row = np.full((len(ids), 3), np.nan, dtype=np.float32)
                    brow = np.full(3, np.nan, dtype=np.float32)
                    players.append(row)
                    ball.append(brow)
                continue
            if not keep:
                continue
            o = OBJ.search(line)
            if o is None:
                continue
            x, y, s = _num(o.group(3)), _num(o.group(4)), _num(o.group(5))
            if o.group(1) == "ball":
                brow[:] = (x * PITCH[0], y * PITCH[1], s)
            elif o.group(2) in col:
                row[col[o.group(2)]] = (x * PITCH[0], y * PITCH[1], s)
    return dict(period=np.array(period, dtype=np.int8), mtime=np.array(mtime), status=np.array(status, dtype=np.int8),
                players=np.stack(players), ball=np.stack(ball), team=team)


def smooth_tracks(players, period):
    from scipy.ndimage import gaussian_filter1d
    out = players.copy()
    sigma = SMOOTH_S * FPS
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        xy = out[rows, :, :2].astype(np.float64)
        seen = np.isfinite(xy[..., 0])
        w = gaussian_filter1d(seen.astype(np.float64), sigma, axis=0, mode="nearest")
        tot = gaussian_filter1d(np.where(seen[..., None], xy, 0.0), sigma, axis=0, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[rows, :, :2] = np.where(seen[..., None], tot / w[..., None], np.nan).astype(np.float32)
    return out


def fill_neutral(status, period):
    y = status.copy()
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        v = y[rows]
        last = -1
        for i in range(len(v)):          # forward: the last team keeps a loose ball
            if v[i] == 2:
                last = -1
            elif v[i] >= 0:
                last = v[i]
            elif last >= 0:
                v[i] = last
        nxt = -1
        for i in range(len(v) - 1, -1, -1):   # a live stretch that opens loose: the next team
            if v[i] == 2:
                nxt = -1
            elif v[i] >= 0:
                nxt = v[i]
            elif nxt >= 0:
                v[i] = nxt
        v[v < 0] = 2
        y[rows] = v
    return y


def events(match):
    d = json.load(open(os.path.join(FOLDER, "bas", match, match + "_12_class_events.json"), encoding="utf-8"))
    out = []
    for a in d["actions"]:
        half = int(str(a["gameTime"]).split(" - ")[0])
        out.append((half, int(a["frame"]), a["label"], a["team"]))
    return out


def place_events(ev, y, period, mtime):
    """(frames, owners, info): play events on the arrays' frames, lag and side fitted per half."""
    frames, owners, info = [], [], {}
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        # video frame k (1-based) of the released half sits at matchTime ~ (k - 1) * 40 ms after its start
        t0 = mtime[rows[0]]
        base = np.rint((mtime[rows] - t0) / 40.0).astype(int)
        e = [(f, team) for half, f, label, team in ev if half == p and label in PLAY_EVENTS]
        f_ev = np.array([f - 1 for f, _ in e])
        side = np.array([0 if t == "left" else 1 for _, t in e])
        best = None
        for lag in range(-MAX_LAG, MAX_LAG + 1):
            idx = np.searchsorted(base, f_ev + lag).clip(0, len(rows) - 1)
            lab = y[rows[idx]]
            live = lab < 2
            if live.sum() == 0:
                continue
            agree = np.mean(lab[live] == side[live])
            score = max(agree, 1 - agree)
            if best is None or score > best[0]:
                best = (score, lag, agree < 0.5, idx, live.mean())
        score, lag, flip, idx, live_share = best
        frames.append(rows[idx])
        owners.append(1 - side if flip else side)
        info[p] = dict(agree=round(float(score), 3), lag=lag, events=len(e), live=round(float(live_share), 3))
    return np.concatenate(frames), np.concatenate(owners), info


def load(match):
    path = os.path.join(CACHE, "{}.v{}.npz".format(match, CACHE_VERSION))
    if os.path.isfile(path):
        with np.load(path, allow_pickle=True) as f:
            out = {k: f[k] for k in f.files}
        out["info"] = out["info"].item()
        return out
    os.makedirs(CACHE, exist_ok=True)
    raw = parse(match)
    period = raw["period"]
    players = smooth_tracks(raw["players"], period)
    y = fill_neutral(raw["status"], period)
    frames, owners, info = place_events(events(match), y, period, raw["mtime"])
    y_ctrl = control.relabel(y, frames, owners)
    info.update(neutral=float(np.mean(raw["status"] == -1)), dead=float(np.mean(y == 2)),
                ctrl_changed=float(np.mean(y_ctrl != y)), minutes=len(y) / FPS / 60,
                players_median=float(np.median(np.isfinite(players[..., 0]).sum(1))))
    out = dict(players=players, players_raw=raw["players"], ball=raw["ball"], team=raw["team"], period=period,
               y=y, y_ctrl=y_ctrl, ev_frames=frames, ev_owners=owners, info=np.array(info, dtype=object))
    np.savez(path, **out)
    out["info"] = info
    return out


def job(match):
    d = load(match)
    return match, d["info"]


def main():
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(5) as pool:
        for match, info in pool.map(job, MATCHES):
            print(match, json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
