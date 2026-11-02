"""Fit the v4 model on IDSSE: v3 with a dead ball that moves, as it does in the challenge.

    python possession/train_v4.py

IDSSE freezes the ball while play is dead, so v3 learned a signal the challenge does not have.
train.moving_dead_ball replaces those frames before degrading. This is improve.py's arm mixed_all3
run with LIVE_DEAD=1. The switch cost favours the challenge's controlled possession: chosen against
the control labels on IDSSE and Metrica, never on the public five minutes.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

from possession import train_v3  # noqa: E402

MODEL_PATH = os.path.join(DATA_DIR, "models", "hgb_v4.pkl")

# Params
SWITCH_COST = 40.0


if __name__ == "__main__":
    train_v3.main(moving=True, model_path=MODEL_PATH, switch_cost=SWITCH_COST)
