"""v5: v4's HGB fitted on control labels, mixed with a set encoder, then one Viterbi per half.

    python possession/v5.py                          # fit the HGB on IDSSE (CPU, deterministic)
    python possession/pipeline.py --version v5       # write the submission

The encoder is trained apart, on GPU, by experiments/encoder_train.py (three seeds); its weights are
committed in possession/models/ and run in NumPy (encoder.py), so the submission is regenerated
from the repository without torch. Weight and costs: best mean over the seven IDSSE matches and
both gap patterns against the control labels (encoder_score.py pick), never the public five minutes.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

HGB_PATH = os.path.join(DATA_DIR, "models", "hgb_v5.pkl")
ENCODER_PATHS = [os.path.join(ROOT, "possession", "models", "{}.npz".format(arm))
                 for arm in ("encoder", "encoder_s2028", "encoder_s2029")]

# Params
ENCODER_WEIGHT = 0.75
SWITCH_COST = 20.0
DEAD_COST = 20.0


if __name__ == "__main__":
    from possession import train_v3

    train_v3.main(moving=True, model_path=HGB_PATH, switch_cost=SWITCH_COST, dead_cost=DEAD_COST, labels="ctrl")
