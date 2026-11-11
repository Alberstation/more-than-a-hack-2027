"""Plan 3j phase 1: short-recipe screen of encoder arms. Torch env.

    <torch-python> possession/experiments/screen.py <improve_live> <arm> [<arm> ...]
    LIVE_DEAD=1 python possession/experiments/screen_score.py <arm> ...     (.venv, scores against `ref`)

Short recipe: 1 seed (2027), SCREEN_ITERS iterations (default 1000), 4 folds: IDSSE J03WMX and
J03WN1 held out (trained on the other 6 IDSSE + Metrica 1-3, as v6), Metrica 1 and 2 held out
(trained on all IDSSE + the other two games). Each arm is paired with `ref` (the v6 recipe, short).
Every arm also writes <arm>_swa: the weights averaged over the last third of training (idea 7).
Arms, fixed before any score was seen:
  ref    v6 recipe                      nomet  without Metrica (calibration: ref must beat it)
  P03/P1 possessor head, lambda 0.3/1 (idea 1): per player, logit = <phi(player), q(TCN at t)> +
         u(phi), softmax over the seen players of both teams, target the possessing team's player
         nearest the clean ball (sets_aux); unused at inference
  PS     P03 + ball-state head (control / contest / loose / dead), lambda 0.3 (idea 2)
  C      consistency with the clean view of the same crop, KL to its stop-gradient output (idea 3)
  F      pitch mirrored top-bottom with probability 0.5 (idea 8)
  R      trained on sets_rand (random degradation per file, idea 6); tested as the others
  RX     two more ball-input channels: each team's mean velocity towards the opponent goal (delayed
         reaction, plan 3j)
Outputs: <improve_live>/screen/<arm>/idsse_<match>.npz (lp_uniform, lp_dead) and
metrica_<game>_<repeat>_<gap>.npz (lp), gaps uniform, dead and rand (the S6 bench).
"""
import os
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import encoder_train as et  # noqa: E402

ITERS = int(os.environ.get("SCREEN_ITERS", "1000"))
SEED = 2027
FOLDS = (("idsse", "J03WMX"), ("idsse", "J03WN1"), ("metrica", 1), ("metrica", 2))
if os.environ.get("SCREEN_FOLDS") == "all":   # confirmation: every held match / game
    FOLDS = tuple(("idsse", m) for m in ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"))         + tuple(("metrica", g) for g in (1, 2, 3))
ARMS = {"ref": {}, "nomet": dict(nomet=True), "P03": dict(poss=0.3), "P1": dict(poss=1.0),
        "PS": dict(poss=0.3, state=0.3), "C": dict(cons=1.0), "F": dict(flip=True), "R": dict(rand=True),
        "RX": dict(react=True), "SK": dict(sk=20), "SK10": dict(sk=10)}
FLIP_P, FLIP_B = [1, 3, 5], [1, 3]
S = 20


class Screen(et.Encoder):
    def __init__(self, ball, poss, state):
        super().__init__(ball=ball, outputs=3 + (4 if state else 0))
        self.poss = poss
        if poss:
            self.q = nn.Linear(et.WIDTH, 64)
            self.u = nn.Linear(64, 1)

    def forward(self, players, ball, extras=False):
        seen = players[..., 7:8]
        h = self.phi(players) * seen
        present = seen.sum(dim=3)
        mean = h.sum(dim=3) / present.clamp(min=1)
        top = torch.where(seen > 0, h, torch.full_like(h, -1e4)).amax(dim=3)
        top = torch.where(present > 0.5, top, torch.zeros_like(top))
        near = torch.where(seen > 0, -players[..., 6:7] * 10, torch.full_like(seen, -1e4))
        attn = (torch.softmax(near, dim=3) * h).sum(dim=3)
        z = torch.cat([mean, top, attn], dim=-1).flatten(2)
        x = self.mix(torch.cat([z, ball], dim=-1)).transpose(1, 2)
        for conv, norm in zip(self.tcn, self.norms):
            x = x + self.drop(nn.functional.gelu(norm(conv(x))))
        out = self.head(x)
        if not extras or not self.poss:
            return out, None
        q = self.q(x.transpose(1, 2))                                   # (B, T, H)
        logit = (h * q[:, :, None, None, :]).sum(-1) + self.u(h)[..., 0]
        logit = torch.where(seen[..., 0] > 0, logit, torch.full_like(logit, -1e4))
        return out, logit.flatten(2)                                    # (B, T, 2S)


def react(players):
    """(views, T, 2): each grid team's mean velocity towards the opponent goal."""
    vx, seen = players[..., 2].astype(np.float32), players[..., 7].astype(np.float32)
    return ((vx * seen).sum(-1) / np.maximum(seen.sum(-1), 1)).astype(np.float16)


def ball_in(d, cfg):
    return np.concatenate([d["ball"], react(d["players"])], axis=-1) if cfg.get("react") else d["ball"]


def files(fold, cfg):
    kind, held = fold
    root = os.path.join(OUT, "sets_rand" if cfg.get("rand") else "sets")
    ids = lambda ms: [os.path.join(root, "{}_{}_d{}.npz".format(m, g, d)) for m in ms for g in et.GAPS for d in range(et.DRAWS)]
    met = lambda gs: [] if cfg.get("nomet") else [os.path.join(root, "metrica_g{}_{}_{}.npz".format(x, r, g))
                                                 for x in gs for r in range(et.METRICA_REPEATS) for g in et.GAPS]
    # skillcorner_data.py sets, sorted match ids; SK10 takes every other one
    sk_ids = sorted({n.split("_")[1] for n in os.listdir(os.path.join(OUT, "sets")) if n.startswith("skillcorner_")})
    sk = [os.path.join(OUT, "sets", "skillcorner_{}_0_{}.npz".format(m, g))
          for m in sk_ids[::20 // cfg["sk"]] for g in et.GAPS] if cfg.get("sk") else []
    if kind == "idsse":
        return ids([m for m in et.MATCHES if m != held]) + met(et.METRICA_GAMES) + sk
    return ids(et.MATCHES) + met([x for x in et.METRICA_GAMES if x != held]) + sk


def clean_key(path):
    base = os.path.basename(path)
    return base.split("_")[0] + "_" + base.split("_")[1] if base.startswith("metrica") else base.split("_")[0]


def train(paths, cfg):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = et.DEVICE
    g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    clean = {}
    data = []
    for path in paths:
        d = et.load(path)
        y = d["y_ctrl"][d["steps"]]
        b = ball_in(d, cfg)
        aux = et.load(os.path.join(OUT, "sets_aux", os.path.basename(path))) if (cfg.get("poss") or cfg.get("state")) else None
        if cfg.get("cons"):
            key = clean_key(path)
            if key not in clean:
                c = et.load(os.path.join(OUT, "sets_clean", key + ".npz"))
                clean[key] = [g(c["players"][v]) for v in (0, 1)], [g(c["ball"][v]) for v in (0, 1)]
        for view in (0, 1):
            item = dict(p=g(d["players"][view]), b=g(b[view]),
                        y=g((y if view == 0 else et.swap_y(y)).astype(np.int64)))
            if aux is not None:
                item["poss"] = g(aux["poss"][view].astype(np.int64))
                item["state"] = g(aux["state"].astype(np.int64))
            if cfg.get("cons"):
                item["cp"], item["cb"] = clean[key][0][view], clean[key][1][view]
            data.append(item)
    model = Screen(ball=6 + (2 if cfg.get("react") else 0), poss=bool(cfg.get("poss")), state=bool(cfg.get("state"))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=ITERS)
    swa, swa_n = None, 0
    model.train()
    for it in range(ITERS):
        batch = {k: [] for k in ("p", "b", "y", "poss", "state", "cp", "cb")}
        for _ in range(et.BATCH):
            item = data[rng.integers(len(data))]
            s = int(rng.integers(0, len(item["y"]) - et.CROP))
            for k in batch:
                if k in item:
                    batch[k].append(item[k][s:s + et.CROP])
        P, B = torch.stack(batch["p"]).float(), torch.stack(batch["b"]).float()
        if cfg.get("flip"):
            f = torch.from_numpy(rng.random(et.BATCH) < 0.5).to(dev)
            sp = torch.ones(et.BATCH, 1, 1, 1, P.shape[-1], device=dev)
            sp[..., FLIP_P] = -1
            sb = torch.ones(et.BATCH, 1, B.shape[-1], device=dev)
            sb[..., FLIP_B] = -1
            P = torch.where(f[:, None, None, None, None], P * sp, P)
            B = torch.where(f[:, None, None], B * sb, B)
        Y = torch.stack(batch["y"])
        with torch.autocast(dev, dtype=torch.bfloat16):
            out, plog = model(P, B, extras=True)
        out = out.float()
        loss = nn.functional.cross_entropy(out[:, :3], Y)
        if cfg.get("poss"):
            target = torch.stack(batch["poss"])
            ok = target >= 0
            seen_t = torch.gather(P[..., 7].flatten(2), 2, target.clamp(min=0)[..., None])[..., 0] > 0
            target = torch.where(ok & seen_t, target, torch.full_like(target, -1))
            if (target >= 0).any():
                loss = loss + cfg["poss"] * nn.functional.cross_entropy(plog.float().transpose(1, 2), target, ignore_index=-1)
        if cfg.get("state"):
            loss = loss + cfg["state"] * nn.functional.cross_entropy(out[:, 3:7], torch.stack(batch["state"]))
        if cfg.get("cons"):
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
                teacher = torch.softmax(model(torch.stack(batch["cp"]).float(), torch.stack(batch["cb"]).float())[0][:, :3].float(), 1)
            loss = loss + cfg["cons"] * nn.functional.kl_div(torch.log_softmax(out[:, :3], 1), teacher, reduction="none").sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it >= ITERS * 2 // 3 and it % 20 == 0:
            sd = {k: v.detach().float().clone() for k, v in model.state_dict().items()}
            swa = sd if swa is None else {k: swa[k] + (sd[k] - swa[k]) / (swa_n + 1) for k in swa}
            swa_n += 1
    model.eval()
    averaged = Screen(ball=model.mix.in_features - 2 * 3 * 64, poss=model.poss, state=bool(cfg.get("state"))).to(dev)
    averaged.load_state_dict(swa)
    return model, averaged.eval()


@torch.no_grad()
def predict(model, path, cfg):
    d = et.load(path)
    steps, period = d["steps"], d["period"]
    b_all = ball_in(d, cfg)
    p_steps = np.zeros((len(steps), 3))
    for q in np.unique(period[steps]):
        r = np.flatnonzero(period[steps] == q)
        views = []
        for view in (0, 1):
            pl = torch.from_numpy(d["players"][view][r]).to(et.DEVICE).float()[None]
            bl = torch.from_numpy(b_all[view][r]).to(et.DEVICE).float()[None]
            prob = torch.softmax(model(pl, bl)[0][0, :3].float(), dim=0).T.cpu().numpy()
            views.append(prob if view == 0 else prob[:, [1, 0, 2]])
        p_steps[r] = np.mean(views, axis=0)
    index = np.searchsorted(steps, np.arange(len(period)), side="right") - 1
    return np.log(np.clip(p_steps[index], 1e-4, 1.0)).astype(np.float32)


def run(arm):
    cfg = ARMS[arm]
    for fold in FOLDS:
        kind, held = fold
        folders = [os.path.join(OUT, "screen", a) for a in (arm, arm + "_swa")]
        done = os.path.join(folders[1], "idsse_{}.npz".format(held) if kind == "idsse" else "metrica_{}_2_rand.npz".format(held))
        if os.path.isfile(done):
            continue
        for f in folders:
            os.makedirs(f, exist_ok=True)
        t = time.time()
        models = train(files(fold, cfg), cfg)
        sets = os.path.join(OUT, "sets")
        for model, folder in zip(models, folders):
            if kind == "idsse":
                np.savez(os.path.join(folder, "idsse_{}.npz".format(held)),
                         **{"lp_" + g: predict(model, os.path.join(sets, "{}_{}_d0.npz".format(held, g)), cfg) for g in et.GAPS})
            else:
                for r in range(et.METRICA_REPEATS):
                    for g in et.GAPS + ("rand",):
                        np.savez(os.path.join(folder, "metrica_{}_{}_{}.npz".format(held, r, g)),
                                 lp=predict(model, os.path.join(sets, "metrica_g{}_{}_{}.npz".format(held, r, g)), cfg))
        print(arm, kind, held, round(time.time() - t), "s", flush=True)


if __name__ == "__main__":
    OUT = sys.argv[1]
    torch.backends.cudnn.benchmark = False
    for a in sys.argv[2:]:
        run(a)
