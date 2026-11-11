"""v7: v6 with the set encoder also trained on SkillCorner's 20 open matches.

    python possession/v5.py                          # fit the HGB on IDSSE (CPU, deterministic; same as v5)
    python possession/pipeline.py                    # write the submission (v7 is the default)

Only the encoder changes: it learns from three providers, IDSSE/DFL (7 matches), Metrica (3 sample
games) and SkillCorner (20 A-League matches, github.com/SkillCorner/opendata, MIT), each degraded
to look like the challenge tracking. Held out one match at a time against v6 (control labels,
constants nested, 3 seeds): an unseen Metrica game +0.029 / +0.027 (3/3), an unseen IDSSE match
-0.008 / -0.010. Trained without Metrica, it still beat v6 on Metrica, so the gain carries to a
provider it never saw (experiments/gate_sk.py, BITACORA 3j). Weights in possession/models/, NumPy.
Weight and costs: best mean over the seven IDSSE matches and both gap patterns
(encoder_score.py pick), never the challenge; the weight 1.0 leaves the HGB out of the mix.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import v5  # noqa: E402

HGB_PATH = v5.HGB_PATH
ENCODER_PATHS = [os.path.join(ROOT, "possession", "models", "encoder_metrica_sk_s{}.npz".format(seed))
                 for seed in (2027, 2028, 2029)]

# Params
ENCODER_WEIGHT = 1.0
SWITCH_COST = 14.0
DEAD_COST = 20.0
