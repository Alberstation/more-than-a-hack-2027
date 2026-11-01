"""What does the Smart Tagging team mean? Agreement of each code with v2's predictions, no labels."""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from load_smart_tagging import load  # noqa: E402

with open(os.path.join(ROOT, "evaluation", "results", "possession_v2.csv")) as handle:
    labels = np.array([row["label"] for row in csv.DictReader(handle)])

stats = {}
for event in load():
    team = event["tags"].get("Team", "")
    side = "H" if team.lower().startswith("barc") else "A" if team.lower().startswith("at") else None
    window = labels[event["start_frame"]:event["end_frame"]]
    alive = window[np.isin(window, ["H", "A"])]
    if side is None or not len(alive):
        continue
    s = stats.setdefault(event["code"], [0, 0, 0, 0])
    s[0] += 1
    s[1] += np.mean(alive == side) > 0.5
    s[2] += float(np.mean(alive == side)) * len(alive)
    s[3] += len(alive)
print("code, n, share of phases with tagged team the majority, frame share, mean length s")
for code, s in sorted(stats.items(), key=lambda kv: -kv[1][0]):
    print("{:<32} {:>3}  {:.2f}  {:.2f}  {:.1f}".format(code, s[0], s[1] / s[0], s[2] / s[3], s[3] / s[0] / 25))
