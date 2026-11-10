"""Plan 3j block D, no training: decoding and mixing ideas on the saved out-of-fold predictions. LIVE_DEAD=1.

    LIVE_DEAD=1 python possession/experiments/decode_fast.py

Every arm picks its constants from the same small grid, nested on IDSSE (the 6 other matches),
Metrica at the arm's best key over all 7. Paired with v6 (ctrl_mixed_all3 + encoder_metrica3,
arithmetic mix) picked the same way. Control labels only.
  v6        p = 0.25 p_hgb + 0.75 p_enc
  ens9      encoder = mean of the v6 (Metrica) and v5 (IDSSE only) encoders, in probability   (idea 9)
  logit10   log-linear mix: lp ∝ (1-w) lp_hgb + w lp_enc, renormalised                     (idea 10)
  gap12     switch cost x f on frames without the ball                                     (idea 12)
  long11    switch cost x f only after 4 s without the ball (long gaps)                    (idea 11)
Pass: >= +0.005 on IDSSE (nested, paired) and on Metrica, both gap patterns.
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
import metrica_mix  # noqa: E402

HGB, ENC, ENC5 = "ctrl_mixed_all3", "encoder_metrica3", "encoder3"
SWITCH = (10.0, 14.0, 20.0, 28.0)
DEAD = (14.0, 20.0, 28.0)
LONG = 100                     # frames without the ball before idea 11 lowers the cost (4 s)


def arithmetic(lp_h, lp_e, w):
    return np.log(np.clip((1 - w) * np.exp(lp_h) + w * np.exp(lp_e), 1e-4, 1.0))


def geometric(lp_h, lp_e, w):
    z = (1 - w) * lp_h + w * lp_e
    z = z - z.max(1, keepdims=True)
    return np.log(np.clip(np.exp(z) / np.exp(z).sum(1, keepdims=True), 1e-4, 1.0))


def since(seen):
    out = np.zeros(len(seen))
    run = 0
    for i, s in enumerate(seen):
        run = 0 if s else run + 1
        out[i] = run
    return out


def table(lp_h, lp_e, lp_e5, seen, y, period):
    out = {}
    gap, long_gap = (~seen).astype(np.float64), (since(seen) > LONG).astype(np.float64)
    def run(name, lp, mask=None, f=1.0):
        for s in SWITCH:
            for d in DEAD:
                # decode: ha = min(-s + gain * logit, 0); gain = s (1 - f) on masked frames lowers the cost to s f
                path = improve.decode(lp, None if mask is None else mask, period, s, s * (1 - f), d)
                out["{}|{:g}_{:g}".format(name, s, d)] = train.score(y, path, period)[0]
    mix = arithmetic(lp_h, lp_e, 0.75)
    run("v6", mix)
    run("ens9", arithmetic(lp_h, np.log(np.clip((np.exp(lp_e) + np.exp(lp_e5)) / 2, 1e-4, 1)), 0.75))
    for w in (0.5, 0.75):
        run("logit10_w{:g}".format(w), geometric(lp_h, lp_e, w))
    for f in (0.5, 0.75):
        run("gap12_f{:g}".format(f), mix, gap, f)
        run("long11_f{:g}".format(f), mix, long_gap, f)
    return out


def idsse_job(args):
    m, gap = args
    load = lambda a: np.load(os.path.join(improve.OUT, "oof", a, m + ".npz"))["lp_" + gap].astype(np.float64)
    with np.load(improve.cache_path(m, gap)) as f:
        seen = f["base"][:, 0] > 0.5
    with np.load(improve.cache_path(m, "uniform")) as f:
        y, period = f["y_ctrl"], f["period"]
    return "idsse", m, gap, table(load(HGB), load(ENC), load(ENC5), seen, y, period)


def metrica_job(args):
    game, repeat, gap = args
    def load(a):
        with np.load(metrica_mix.lp_path(a, game, repeat, gap)) as f:
            return f["lp"].astype(np.float64), f["y"], f["period"]
    lp_h, y, period = load(HGB)
    with np.load(os.path.join(metrica_exp.OUT, "g{}_{}_{}.npz".format(game, repeat, gap))) as f:
        seen = f["base"][:, 0] > 0.5
    y = live_review.metrica_control(y, *live_review.metrica_events(game))
    return "metrica", game, gap, table(lp_h, load(ENC)[0], load(ENC5)[0], seen, y, period)


def main():
    jobs_i = [(m, g) for m in idsse.MATCHES for g in improve.GAPS]
    jobs_m = [(game, r, g) for game in metrica_exp.ALL_GAMES for r in range(metrica_exp.REPEATS) for g in improve.GAPS]
    with ProcessPoolExecutor(8) as pool:
        done = list(pool.map(idsse_job, jobs_i)) + list(pool.map(metrica_job, jobs_m))
    arms = sorted({k.split("|")[0] for k in done[0][3]}, key=lambda a: (a != "v6", a))
    summary = {}
    for gap in improve.GAPS:
        by = {m: t for src, m, g, t in done if src == "idsse" and g == gap}
        me = [t for src, _, g, t in done if src == "metrica" and g == gap]
        res = {}
        for arm in arms:
            keys = [k for k in by[idsse.MATCHES[0]] if k.split("|")[0] == arm]
            folds = np.array([by[h][max(keys, key=lambda k: np.mean([by[m][k] for m in idsse.MATCHES if m != h]))]
                              for h in idsse.MATCHES])
            best = max(keys, key=lambda k: np.mean([by[m][k] for m in idsse.MATCHES]))
            res[arm] = dict(folds=folds, metrica=float(np.mean([t[best] for t in me])), key=best)
        print("\nhuecos {}  (IDSSE anidado | Metrica)".format(gap))
        for arm in arms:
            d = res[arm]["folds"] - res["v6"]["folds"]
            dm = res[arm]["metrica"] - res["v6"]["metrica"]
            summary["{}|{}".format(arm, gap)] = dict(idsse=float(res[arm]["folds"].mean()), diff=float(d.mean()),
                                                     up=int(np.sum(d > 0)), metrica=res[arm]["metrica"],
                                                     metrica_diff=dm, key=res[arm]["key"])
            print("  {:<14} {:.3f} {:+.4f} ({}/7) | {:.3f} {:+.4f}  [{}]".format(
                arm, res[arm]["folds"].mean(), d.mean(), int(np.sum(d > 0)), res[arm]["metrica"], dm,
                res[arm]["key"].split("|")[1]))
    with open(os.path.join(improve.OUT, "decode_fast.json"), "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
