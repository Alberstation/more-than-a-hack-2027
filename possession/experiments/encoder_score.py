"""v4 mixed with the set encoder by one weight, against control labels only. Run with LIVE_DEAD=1.

Done for each reference in REFS: v4 fitted on control labels, and v4 as shipped.

    LIVE_DEAD=1 python possession/experiments/encoder_score.py [encoder_arm ...]

Several encoder arms (seeds) are averaged in probability before mixing; the default is `encoder`.
    LIVE_DEAD=1 python possession/experiments/encoder_score.py pick <ref> <encoder_arm> ...   # v5 constants

p = (1 - w) p_v4 + w p_encoder. The weight and the Viterbi costs for a match are picked on the
other six; Metrica uses their medians. w = 1 is the encoder alone. Paired with v4 picked the same way.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from possession import idsse, train  # noqa: E402
import improve  # noqa: E402
import live_review  # noqa: E402
import metrica_exp  # noqa: E402

REFS = ("ctrl_mixed_all3", "mixed_all3")   # v4 fitted on control labels; v4 as shipped
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
SWITCH = (14.0, 20.0, 28.0, 40.0, 56.0)
DEAD = (20.0, 40.0, 80.0)


def table(lp_ref, lp_enc, y, period):
    out = {}
    for w in WEIGHTS:
        lp = np.log(np.clip((1 - w) * np.exp(lp_ref) + w * np.exp(lp_enc), 1e-4, 1.0))
        for s in SWITCH:
            for d in DEAD:
                out["{:g}_{:g}_{:g}".format(w, s, d)] = train.score(y, improve.decode(lp, None, period, s, 0.0, d), period)[0]
    return out


def averaged(lps):
    return np.log(np.clip(np.mean(np.exp(lps), axis=0), 1e-4, 1.0))


def idsse_job(args):
    ref, encoders, held, gap = args
    lps = []
    for arm in (ref,) + encoders:
        with np.load(os.path.join(improve.OUT, "oof", arm, held + ".npz")) as f:
            lps.append(f["lp_" + gap].astype(np.float64))
    lps = [lps[0], averaged(lps[1:])]
    with np.load(improve.cache_path(held, "uniform")) as f:
        y, period = f["y_ctrl"], f["period"]
    return ref, held, gap, table(*lps, y, period)


def metrica_job(args):
    ref, encoders, game, repeat, gap = args
    lps = []
    for arm in (ref,) + encoders:
        with np.load(os.path.join(metrica_exp.OUT, "lp_{}_{}_{}_{}.npz".format(arm, game, repeat, gap))) as f:
            lps.append(f["lp"].astype(np.float64))
            y, period = f["y"], f["period"]
    y = live_review.metrica_control(y, *live_review.metrica_events(game))
    return ref, gap, table(lps[0], averaged(lps[1:]), y, period)


def nested(by, keys):
    folds, picked = [], []
    for held in idsse.MATCHES:
        best = max(keys, key=lambda k: np.mean([by[m][k] for m in idsse.MATCHES if m != held]))
        folds.append(by[held][best])
        picked.append(best)
    median = [float(np.median([float(k.split("_")[i]) for k in picked])) for i in range(3)]
    grids = (WEIGHTS, SWITCH, DEAD)
    key = "_".join("{:g}".format(min(g, key=lambda v: abs(v - x))) for g, x in zip(grids, median))
    return np.array(folds), picked, key


def main():
    encoders = tuple(sys.argv[1:]) or ("encoder",)
    with ProcessPoolExecutor(6) as pool:
        id_done = list(pool.map(idsse_job, [(r, encoders, m, g) for r in REFS for m in idsse.MATCHES for g in improve.GAPS]))
        me_done = list(pool.map(metrica_job, [(ref, encoders, game, r, g) for ref in REFS for game in metrica_exp.GAMES
                                              for r in range(metrica_exp.REPEATS) for g in improve.GAPS]))
    summary = {}
    for ref, gap in [(r, g) for r in REFS for g in improve.GAPS]:
        by = {h: t for r, h, g, t in id_done if r == ref and g == gap}
        me = [t for r, g, t in me_done if r == ref and g == gap]
        keys = list(next(iter(by.values())))
        arms = dict(v4=[k for k in keys if k.startswith("0_")], encoder=[k for k in keys if k.startswith("1_")],
                    mix=keys)
        res = {}
        for name, ks in arms.items():
            folds, picked, key = nested(by, ks)
            res[name] = dict(folds=folds, picked=picked, key=key, metrica=float(np.mean([t[key] for t in me])))
        row = summary["{}|{}".format(ref, gap)] = {}
        for name in ("encoder", "mix"):
            diff = res[name]["folds"] - res["v4"]["folds"]
            row[name] = dict(score=float(res[name]["folds"].mean()), diff=float(diff.mean()),
                             up=int(np.sum(diff > 0)), folds=[round(float(v), 4) for v in diff],
                             picked=res[name]["picked"], metrica_key=res[name]["key"],
                             metrica_diff=res[name]["metrica"] - res["v4"]["metrica"])
        row["v4"] = dict(score=float(res["v4"]["folds"].mean()), metrica=res["v4"]["metrica"])
    with open(os.path.join(improve.OUT, "encoder_{}.json".format("+".join(encoders))), "w") as handle:
        json.dump(summary, handle, indent=1)
    for gap, v in summary.items():
        print("{} v4 IDSSE {:.3f} Metrica {:.3f}".format(gap, v["v4"]["score"], v["v4"]["metrica"]))
        for name in ("encoder", "mix"):
            r = v[name]
            print("  {:<8} IDSSE {:.3f} {:+.4f} ({}/7)  Metrica {:+.4f} [{}]  pesos {}".format(
                name, r["score"], r["diff"], r["up"], r["metrica_diff"], r["metrica_key"],
                sorted(set(k.split("_")[0] for k in r["picked"]))))


def pick(ref, encoders):
    """Production constants: the weight and costs with the best mean over all seven matches and
    both gap patterns. Not nested; the nested scores above are the estimate."""
    with ProcessPoolExecutor(6) as pool:
        done = list(pool.map(idsse_job, [(ref, encoders, m, g) for m in idsse.MATCHES for g in improve.GAPS]))
    keys = list(done[0][3])
    mean = {k: float(np.mean([t[k] for *_, t in done])) for k in keys}
    for k in sorted(keys, key=lambda k: -mean[k])[:8]:
        print("w_switch_dead {}  {:.4f}".format(k, mean[k]))


if __name__ == "__main__":
    if sys.argv[1:2] == ["pick"]:
        pick(sys.argv[2], tuple(sys.argv[3:]))
    else:
        main()
