"""Frozen v5 / v6 / v7 on SoccerTrack v2 (8 matches, a provider none of them saw), no training.

    python possession/experiments/soccertrack_v6.py

The production inference (possession.encoder, the v5 HGB, possession.decode) on soccertrack_data's
arrays. Players degraded as for IDSSE (train.degrade: dropped tracklets, drifting error), one draw
per match. Two ball conditions:
  noball  every ball frame hidden: the reconstructed ball never reaches the model
  gaps    the reconstructed ball with the challenge's gap lengths and share, uniform; it is built
          from the same events as the labels, so this one reads optimistic
Scored with spellF1 / macroF1 against ctrl, each version at its own constants; also the encoder
alone (weight 1). Output: data/experiments/soccertrack/frozen.json.
"""
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import DATA_DIR  # noqa: E402
from possession import decode, encoder, features_v3, train, v5, v6, v7  # noqa: E402
import dose_exp  # noqa: E402
import soccertrack_data as st  # noqa: E402

OUT = os.path.join(DATA_DIR, "experiments", "soccertrack")
VERSIONS = {"v5": v5, "v6": v6, "v7": v7}
BALLS = ("noball", "gaps")


def decode_all(p, period, switch, dead):
    lp = np.log(np.clip(p, 1e-4, 1.0))
    t = decode.transition_matrix(switch, dead)
    out = np.full(len(p), 2, dtype=np.int8)
    for q in np.unique(period):
        r = np.flatnonzero(period == q)
        out[r] = decode.viterbi(lp[r], t)
    return out


def job(args):
    match, ball_mode, lengths, share = args
    d = st.load(match)
    index = st.MATCHES.index(match)
    rng = np.random.default_rng([train.SEED, 700 + index, 0])
    data = dict(players=d["players"], ball=d["ball"].copy())
    players, ball = train.degrade(data, rng, lengths, share if ball_mode == "gaps" else 0.0)
    if ball_mode == "noball":
        ball[:] = np.nan
    team, period, y = d["team"], d["period"].astype(np.int8), d["y_ctrl"]
    with open(v5.HGB_PATH, "rb") as handle:
        hgb = pickle.load(handle)["model"]
    p_hgb = hgb.predict_proba(features_v3.build(players, ball, team, period, st.PITCH))
    rows = {}
    for name, ver in VERSIONS.items():
        p_enc = encoder.proba([encoder.load(p) for p in ver.ENCODER_PATHS], players, ball, team, period, st.PITCH)
        for label, w in ((name, ver.ENCODER_WEIGHT), (name + "_enc", 1.0)):
            p = (1 - w) * p_hgb + w * p_enc
            path = decode_all(p, period, ver.SWITCH_COST, ver.DEAD_COST)
            spell, macro = train.score(y, path, period)
            rows[label] = dict(spell=float(spell), macro=float(macro),
                               pred_spells=int(np.sum(np.diff(path) != 0)), dead_pred=float(np.mean(path == 2)))
    rows["gt_spells"] = int(np.sum(np.diff(y) != 0))
    return match, ball_mode, rows


def main():
    os.makedirs(OUT, exist_ok=True)
    lengths, share, real = dose_exp.gap_profile()
    assert real
    jobs = [(m, b, lengths, share) for b in BALLS for m in st.MATCHES]
    res = {}
    with ProcessPoolExecutor(4) as pool:
        for match, ball_mode, rows in pool.map(job, jobs):
            res.setdefault(ball_mode, {})[match] = rows
            print(ball_mode, match, " ".join("{} {:.3f}".format(k, v["spell"]) for k, v in rows.items()
                                             if isinstance(v, dict)), flush=True)
    json.dump(res, open(os.path.join(OUT, "frozen.json"), "w"), indent=1)
    for b in BALLS:
        labels = [k for k, v in next(iter(res[b].values())).items() if isinstance(v, dict)]
        print("\n", b, "spellF1 / macroF1, media de", len(res[b]), "partidos")
        for k in labels:
            s = np.array([res[b][m][k]["spell"] for m in st.MATCHES])
            mac = np.array([res[b][m][k]["macro"] for m in st.MATCHES])
            print("  {:<8} {:.3f} / {:.3f}   por partido {}".format(k, s.mean(), mac.mean(),
                                                                 " ".join("{:.3f}".format(v) for v in s)))
        for a, c in (("v6", "v5"), ("v7", "v6")):
            dlt = np.array([res[b][m][a]["spell"] - res[b][m][c]["spell"] for m in st.MATCHES])
            print("  {} - {}: {:+.4f} ({}/{})".format(a, c, dlt.mean(), int(np.sum(dlt > 0)), len(dlt)))


if __name__ == "__main__":
    main()
