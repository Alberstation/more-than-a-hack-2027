"""Plan 3j confirmation: v6 encoders against the SkillCorner ones (encoder_train.py metrica_sk), paired.

    LIVE_DEAD=1 python possession/experiments/gate_sk.py <seeds...>      e.g. 2027   or   2027 2028 2029
    GATE_ARM=encoder_sk LIVE_DEAD=1 python possession/experiments/gate_sk.py 2027 2028 2029   (no Metrica in training)

Both mixed with ctrl_mixed_all3 (out of fold on IDSSE, all-IDSSE on Metrica), seeds averaged in
probability. Weight and Viterbi costs picked on the other six IDSSE matches (nested); Metrica held
games at the median of the picks. Also at v6's constants (0.75, 14, 20). Control labels only.
Gate: +0.01 to +0.02 mean, most folds up.
"""
import itertools
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

HGB = "ctrl_mixed_all3"
WEIGHTS, SWITCH, DEAD = (0.5, 0.75, 1.0), (10.0, 14.0, 20.0, 28.0), (14.0, 20.0, 28.0)
KEYS = list(itertools.product(WEIGHTS, SWITCH, DEAD))
V6 = (0.75, 14.0, 20.0)


def table(lp_h, lp_e, y, period):
    out = {}
    for w, s, d in KEYS:
        lp = np.log(np.clip((1 - w) * np.exp(lp_h) + w * np.exp(lp_e), 1e-4, 1.0))
        out[(w, s, d)] = train.score(y, improve.decode(lp, None, period, s, 0.0, d), period)[0]
    return out


def avg(lps):
    return np.log(np.mean([np.exp(v) for v in lps], axis=0))


def idsse_job(args):
    arms, m, g = args
    with np.load(improve.cache_path(m, "uniform")) as f:
        y, period = f["y_ctrl"], f["period"]
    load = lambda a: np.load(os.path.join(improve.OUT, "oof", a, m + ".npz"))["lp_" + g].astype(np.float64)
    return m, g, table(load(HGB), avg([load(a) for a in arms]), y, period)


def metrica_job(args):
    arms, game, r, g = args
    with np.load(metrica_mix.lp_path(HGB, game, r, g)) as f:
        lp_h, period, y = f["lp"].astype(np.float64), f["period"], f["y"]
    y = live_review.metrica_control(y, *live_review.metrica_events(game))
    lps = [np.load(os.path.join(metrica_exp.OUT, "lp_{}_{}_{}_{}.npz".format(a, game, r, g)))["lp"].astype(np.float64)
           for a in arms]
    return game, g, table(lp_h, avg(lps), y, period)


def evaluate(arms, pool):
    ids = list(pool.map(idsse_job, [(arms, m, g) for m in idsse.MATCHES for g in improve.GAPS]))
    met = list(pool.map(metrica_job, [(arms, x, r, g) for x in metrica_exp.ALL_GAMES
                                      for r in range(metrica_exp.REPEATS) for g in improve.GAPS]))
    out = {}
    for g in improve.GAPS:
        by = {m: t for m, gg, t in ids if gg == g}
        folds, picked = [], []
        for held in idsse.MATCHES:
            best = max(KEYS, key=lambda k: np.mean([by[m][k] for m in idsse.MATCHES if m != held]))
            folds.append(by[held][best])
            picked.append(best)
        key = tuple(min(grid, key=lambda v: abs(v - float(np.median([p[i] for p in picked]))))
                    for i, grid in enumerate((WEIGHTS, SWITCH, DEAD)))
        games = [np.mean([t[key] for x, gg, t in met if gg == g and x == game]) for game in metrica_exp.ALL_GAMES]
        games_v6 = [np.mean([t[V6] for x, gg, t in met if gg == g and x == game]) for game in metrica_exp.ALL_GAMES]
        out[g] = dict(idsse=np.array(folds), idsse_v6=np.array([by[m][V6] for m in idsse.MATCHES]),
                      metrica=np.array(games), metrica_v6=np.array(games_v6), key=key)
    return out


def main(seeds):
    base = ["encoder_metrica_s{}".format(s) for s in seeds]
    sk = ["{}_s{}".format(os.environ.get("GATE_ARM", "encoder_metrica_sk"), s) for s in seeds]
    with ProcessPoolExecutor(8) as pool:
        a, b = evaluate(base, pool), evaluate(sk, pool)
    print("semillas", " ".join(map(str, seeds)), "| v6 ->", sk[0].rsplit("_s", 1)[0], ", contra ctrl")
    for g in improve.GAPS:
        for mode, ik, mk in (("anidado", "idsse", "metrica"), ("const v6", "idsse_v6", "metrica_v6")):
            di, dm = b[g][ik] - a[g][ik], b[g][mk] - a[g][mk]
            print("  {:<8} {:<9} IDSSE {:.3f} -> {:.3f} {:+.4f} ({}/7)   Metrica {:.3f} -> {:.3f} {:+.4f} ({}/3)   por partido {}".format(
                g, mode, a[g][ik].mean(), b[g][ik].mean(), di.mean(), int(np.sum(di > 0)),
                a[g][mk].mean(), b[g][mk].mean(), dm.mean(), int(np.sum(dm > 0)),
                " ".join("{:+.3f}".format(v) for v in np.r_[di, dm])))
        print("  {:<8} claves Metrica: v6 {}  SK {}".format(g, a[g]["key"], b[g]["key"]))


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [2027])
