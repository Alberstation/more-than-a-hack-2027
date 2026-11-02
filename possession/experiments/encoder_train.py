"""Set encoder per team plus a dilated TCN, out of fold on IDSSE. Runs in the separate torch env.

    <torch-python> possession/experiments/encoder_train.py <data>/experiments/improve_live [seed] [variant]

Reads sets/ from encoder_data.py. Per player a shared MLP; per team masked mean, max and a
softmax by closeness to the ball; then a non-causal TCN of about +-50 s at 5 steps per second.
Labels are the control labels (CHALLENGE.md asks for controlled possession), both team views,
symmetric average at inference. Writes
oof/<arm>/<match>.npz and metrica/lp_<arm>_<game>_<repeat>_<gap>.npz in the v4 formats, and the
all-IDSSE weights to encoder_models/<arm>.pt. arm is encoder, or encoder_s<seed> for other seeds,
or encoder_<variant>_s<seed>. Variants, fixed before any score was seen:
  w05, w1  each step weighted by (mean spell / its spell)^alpha, capped at WEIGHT_CAP: spellF1 counts
           every possession once, frame loss counts it by length.
  trans    a fourth output, the logit that possession changes hands here (H<->A within one step),
           written as logit_<gap> for a decoder whose switch cost moves with it.
Experiment only: not part of the submission environment.
"""
import os
import sys
import time

import numpy as np
import torch
from torch import nn

MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
GAPS = ("uniform", "dead")
DRAWS = 3
SEED = 2027
ITERS = 3000
BATCH = 24
CROP = 512
WIDTH = 128
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_CAP = 25.0
VARIANTS = {"base": {}, "w05": dict(alpha=0.5), "w1": dict(alpha=1.0), "trans": dict(trans=True)}


class Encoder(nn.Module):
    def __init__(self, channels=8, ball=6, hidden=64, outputs=3):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.mix = nn.Linear(2 * 3 * hidden + ball, WIDTH)
        self.tcn = nn.ModuleList(nn.Conv1d(WIDTH, WIDTH, 3, dilation=d, padding=d) for d in DILATIONS)
        self.norms = nn.ModuleList(nn.GroupNorm(8, WIDTH) for _ in DILATIONS)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Conv1d(WIDTH, outputs, 1)

    def forward(self, players, ball):
        seen = players[..., 7:8]
        h = self.phi(players) * seen                                     # (B, T, 2, S, H)
        present = seen.sum(dim=3)
        mean = h.sum(dim=3) / present.clamp(min=1)
        top = torch.where(seen > 0, h, torch.full_like(h, -1e4)).amax(dim=3)
        top = torch.where(present > 0.5, top, torch.zeros_like(top))   # an empty team pools to zero
        near = torch.where(seen > 0, -players[..., 6:7] * 10, torch.full_like(seen, -1e4))
        attn = (torch.softmax(near, dim=3) * h).sum(dim=3)
        z = torch.cat([mean, top, attn], dim=-1).flatten(2)             # (B, T, 2 * 3H)
        x = self.mix(torch.cat([z, ball], dim=-1)).transpose(1, 2)      # (B, W, T)
        for conv, norm in zip(self.tcn, self.norms):
            x = x + self.drop(nn.functional.gelu(norm(conv(x))))
        return self.head(x)                                             # (B, 3 [+1], T)


def swap_y(y):
    return np.where(y == 2, 2, 1 - y)


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def runs(a):
    cut = np.flatnonzero(np.diff(a)) + 1
    return np.r_[0, cut], np.r_[cut, len(a)]


def spell_weights(labels, alpha):
    """(mean spell / own spell)^alpha per step, capped, scaled to mean 1 over all training steps."""
    mean = np.mean([e - s for y in labels for s, e in zip(*runs(y))])
    out = []
    for y in labels:
        own = np.concatenate([np.repeat(e - s, e - s) for s, e in zip(*runs(y))]).astype(np.float64)
        out.append(np.minimum((mean / own) ** alpha, WEIGHT_CAP))
    total = np.concatenate(out).mean()
    return [w / total for w in out]


def changes(y):
    """1 where possession passes H<->A at this step, and on the step before it."""
    flip = np.zeros(len(y), dtype=np.float32)
    hit = np.flatnonzero((y[1:] < 2) & (y[:-1] < 2) & (y[1:] != y[:-1])) + 1
    for k in (-1, 0):
        flip[np.clip(hit + k, 0, len(y) - 1)] = 1
    return flip


def train(files, seed, variant="base"):
    cfg = VARIANTS[variant]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data, labels = [], []
    for path in files:
        d = load(path)
        y = d["y_ctrl"][d["steps"]]
        labels.append(y)
        for view in (0, 1):
            data.append([torch.from_numpy(d["players"][view]).to(DEVICE), torch.from_numpy(d["ball"][view]).to(DEVICE),
                         torch.from_numpy((y if view == 0 else swap_y(y)).astype(np.int64)).to(DEVICE)])
    weights = spell_weights(labels, cfg["alpha"]) if "alpha" in cfg else [np.ones(len(y)) for y in labels]
    for i, item in enumerate(data):
        item.append(torch.from_numpy(weights[i // 2].astype(np.float32)).to(DEVICE))
        item.append(torch.from_numpy(changes(labels[i // 2])).to(DEVICE))
    model = Encoder(outputs=4 if cfg.get("trans") else 3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=ITERS)
    model.train()
    for it in range(ITERS):
        P, B, Y, W, C = [], [], [], [], []
        for _ in range(BATCH):
            p, b, y, w, c = data[rng.integers(len(data))]
            s = int(rng.integers(0, len(y) - CROP))
            P.append(p[s:s + CROP])
            B.append(b[s:s + CROP])
            Y.append(y[s:s + CROP])
            W.append(w[s:s + CROP])
            C.append(c[s:s + CROP])
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            out = model(torch.stack(P).float(), torch.stack(B).float()).float()
        if "alpha" in cfg:
            W = torch.stack(W)
            loss = (nn.functional.cross_entropy(out[:, :3], torch.stack(Y), reduction="none") * W).sum() / W.sum()
        else:
            loss = nn.functional.cross_entropy(out[:, :3], torch.stack(Y))
        if cfg.get("trans"):
            loss = loss + nn.functional.binary_cross_entropy_with_logits(out[:, 3], torch.stack(C))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0:
            print("  iter", it, round(loss.item(), 4), flush=True)
    return model.eval()


@torch.no_grad()
def predict(model, path):
    """(log-probabilities, data, change logit) per frame at 25 fps: each period run whole, both
    views averaged. The change logit is None for a model without that output."""
    d = load(path)
    steps, period = d["steps"], d["period"]
    p_steps = np.zeros((len(steps), 3))
    logit = np.zeros(len(steps))
    for q in np.unique(period[steps]):
        r = np.flatnonzero(period[steps] == q)
        views, flips = [], []
        for view in (0, 1):
            players = torch.from_numpy(d["players"][view][r]).to(DEVICE).float()[None]
            ball = torch.from_numpy(d["ball"][view][r]).to(DEVICE).float()[None]
            out = model(players, ball)[0].float()
            prob = torch.softmax(out[:3], dim=0).T.cpu().numpy()
            views.append(prob if view == 0 else prob[:, [1, 0, 2]])
            if out.shape[0] > 3:
                flips.append(out[3].cpu().numpy())
        p_steps[r] = np.mean(views, axis=0)
        if flips:
            logit[r] = np.mean(flips, axis=0)
    index = np.searchsorted(steps, np.arange(len(period)), side="right") - 1
    lp = np.log(np.clip(p_steps[index], 1e-4, 1.0)).astype(np.float32)
    return lp, d, (logit[index].astype(np.float32) if model.head.out_channels > 3 else None)


def main():
    out_root = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else SEED
    variant = sys.argv[3] if len(sys.argv) > 3 else "base"
    if variant != "base":
        arm = "encoder_{}_s{}".format(variant, seed)
    else:
        arm = "encoder" if seed == SEED else "encoder_s{}".format(seed)
    sets = os.path.join(out_root, "sets")
    name = lambda m, g, d: os.path.join(sets, "{}_{}_d{}.npz".format(m, g, d))
    folder = os.path.join(out_root, "oof", arm)
    os.makedirs(folder, exist_ok=True)
    for held in MATCHES:
        target = os.path.join(folder, held + ".npz")
        if os.path.isfile(target):
            continue
        t = time.time()
        model = train([name(m, g, d) for m in MATCHES if m != held for g in GAPS for d in range(DRAWS)], seed, variant)
        out = {}
        for g in GAPS:
            lp, _, logit = predict(model, name(held, g, 0))
            out["lp_" + g] = lp
            if logit is not None:
                out["logit_" + g] = logit
        np.savez(target, **out)
        print(arm, held, round(time.time() - t), "s", flush=True)
    model = train([name(m, g, d) for m in MATCHES for g in GAPS for d in range(DRAWS)], seed, variant)
    os.makedirs(os.path.join(out_root, "encoder_models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_root, "encoder_models", arm + ".pt"))
    for game in (1, 2):
        for repeat in range(3):
            for g in GAPS:
                lp, d, logit = predict(model, os.path.join(sets, "metrica_g{}_{}_{}.npz".format(game, repeat, g)))
                extra = {} if logit is None else dict(logit=logit)
                np.savez(os.path.join(out_root, "metrica", "lp_{}_{}_{}_{}.npz".format(arm, game, repeat, g)),
                         lp=lp, y=d["y_raw"], period=d["period"], **extra)
    print(arm, "metrica done", flush=True)


if __name__ == "__main__":
    main()
