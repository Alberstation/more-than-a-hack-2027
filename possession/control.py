"""Control labels for IDSSE: CHALLENGE.md's controlled possession, built from the official events.

A spell of a team that plays no ball (Play or ShotAtGoal) within CONTROL_WINDOW_S of it is a touch,
a deflection or a won duel without control, so it goes back to the team before it. Dead ball is
left as it is. v5 fits on these labels; experiments/improve.py builds its caches with them.
"""

import datetime as dt
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

EVENTS = os.path.join(DATA_DIR, "external", "idsse_events")
FPS = 25

# Params
CONTROL_WINDOW_S = 1.5      # fixed before looking at any score; never swept


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
    return relabel(y, *event_frames(match_id, y, period))


def relabel(y, frames, owners):
    """control_labels from events already placed on frames (frames, owners as event_frames gives)."""
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
