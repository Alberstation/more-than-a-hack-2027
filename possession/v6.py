"""v6: v5 with the set encoder also trained on Metrica's three sample games, then one Viterbi per half.

    python possession/v5.py                          # fit the HGB on IDSSE (CPU, deterministic; same as v5)
    python possession/pipeline.py                    # write the submission (v6 is the default)

Only the encoder changes: it learns from two providers, IDSSE/DFL (7 matches) and Metrica (3 sample
games, github.com/metrica-sports/sample-data, credited as its README asks), each degraded to look
like the challenge tracking. Held out one match at a time, this beat v5 on an unseen IDSSE match
(+0.013 / +0.019) and on an unseen Metrica game (+0.011 / +0.005) (experiments/metrica_mix.py).
Its weights are committed in possession/models/ and run in NumPy. Weight and costs: best mean over
the seven IDSSE matches and both gap patterns (encoder_score.py pick), never the challenge.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import v5  # noqa: E402

HGB_PATH = v5.HGB_PATH
ENCODER_PATHS = [os.path.join(ROOT, "possession", "models", "encoder_metrica_s{}.npz".format(seed))
                 for seed in (2027, 2028, 2029)]

# Params
ENCODER_WEIGHT = 0.75
SWITCH_COST = 14.0
DEAD_COST = 20.0
