"""Leave-one-match-out predictions for the current model, cached so decoders can be tried cheaply."""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from possession import idsse, train  # noqa: E402

OUT = os.path.join(train.DATA_DIR, "experiments", "oof_base")
os.makedirs(OUT, exist_ok=True)

gap_lengths, missing_share = train.challenge_gap_profile()
print("gaps", len(gap_lengths), round(float(missing_share), 4), flush=True)
rng = np.random.default_rng(train.SEED)
matches = {}
for match_id in idsse.MATCHES:
    t = time.time()
    matches[match_id] = train.match_arrays(match_id, rng, gap_lengths, missing_share)
    print("built", match_id, matches[match_id]["X"].shape, round(time.time() - t), flush=True)

for held in idsse.MATCHES:
    t = time.time()
    test = matches[held]
    model = train.fit([m for k, m in matches.items() if k != held])
    lp = train.log_proba(model, test["X"])
    lp_swap = train.log_proba(model, test["X_swap"])[:, [1, 0, 2]]
    base = train.baseline_predictions(test)
    np.savez_compressed(os.path.join(OUT, held + ".npz"), lp=lp, lp_swap=lp_swap, y=test["y"],
                        period=test["period"], base=base)
    print("fold", held, round(time.time() - t), flush=True)
print("done", flush=True)
