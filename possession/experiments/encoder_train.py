"""Set encoder per team plus a dilated TCN, out of fold on IDSSE. Runs in the separate torch env.

    <torch-python> possession/experiments/encoder_train.py <data>/experiments/improve_live [seed]

Reads sets/ from encoder_data.py. Per player a shared MLP; per team masked mean, max and a
softmax by closeness to the ball; then a non-causal TCN of about +-50 s at 5 steps per second.
Labels are the control labels (CHALLENGE.md asks for controlled possession), both team views,
symmetric average at inference. Writes
oof/<arm>/<match>.npz and metrica/lp_<arm>_<game>_<repeat>_<gap>.npz in the v4 formats, and the
all-IDSSE weights to encoder_models/<arm>.pt. arm is encoder, or encoder_s<seed> for other seeds.
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


class Encoder(nn.Module):
    def __init__(self, channels=8, ball=6, hidden=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.mix = nn.Linear(2 * 3 * hidden + ball, WIDTH)
        self.tcn = nn.ModuleList(nn.Conv1d(WIDTH, WIDTH, 3, dilation=d, padding=d) for d in DILATIONS)
        self.norms = nn.ModuleList(nn.GroupNorm(8, WIDTH) for _ in DILATIONS)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Conv1d(WIDTH, 3, 1)

    def forward(self, players, ball):
        seen = players[..., 7:8]
        h = self.phi(players) * seen                                     # (B, T, 2, S, H)
        count = seen.sum(dim=3).clamp(min=1)
        mean = h.sum(dim=3) / count
        top = torch.where(seen > 0, h, torch.full_like(h, -1e4)).amax(dim=3)
        top = torch.where(count > 0.5, top, torch.zeros_like(top))
        near = torch.where(seen > 0, -players[..., 6:7] * 10, torch.full_like(seen, -1e4))
        attn = (torch.softmax(near, dim=3) * h).sum(dim=3)
        z = torch.cat([mean, top, attn], dim=-1).flatten(2)             # (B, T, 2 * 3H)
        x = self.mix(torch.cat([z, ball], dim=-1)).transpose(1, 2)      # (B, W, T)
        for conv, norm in zip(self.tcn, self.norms):
            x = x + self.drop(nn.functional.gelu(norm(conv(x))))
        return self.head(x)                                             # (B, 3, T)


def swap_y(y):
    return np.where(y == 2, 2, 1 - y)


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def train(files, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = []
    for path in files:
        d = load(path)
        y = d["y_ctrl"][d["steps"]]
        for view in (0, 1):
            data.append((torch.from_numpy(d["players"][view]).to(DEVICE), torch.from_numpy(d["ball"][view]).to(DEVICE),
                         torch.from_numpy((y if view == 0 else swap_y(y)).astype(np.int64)).to(DEVICE)))
    model = Encoder().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=ITERS)
    model.train()
    for it in range(ITERS):
        P, B, Y = [], [], []
        for _ in range(BATCH):
            p, b, y = data[rng.integers(len(data))]
            s = int(rng.integers(0, len(y) - CROP))
            P.append(p[s:s + CROP])
            B.append(b[s:s + CROP])
            Y.append(y[s:s + CROP])
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            out = model(torch.stack(P).float(), torch.stack(B).float())
        loss = nn.functional.cross_entropy(out.float(), torch.stack(Y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0:
            print("  iter", it, round(loss.item(), 4), flush=True)
    return model.eval()


@torch.no_grad()
def predict(model, path):
    """Log-probabilities per frame (25 fps): each period run whole, both views averaged."""
    d = load(path)
    steps, period = d["steps"], d["period"]
    p_steps = np.zeros((len(steps), 3))
    for q in np.unique(period[steps]):
        r = np.flatnonzero(period[steps] == q)
        views = []
        for view in (0, 1):
            players = torch.from_numpy(d["players"][view][r]).to(DEVICE).float()[None]
            ball = torch.from_numpy(d["ball"][view][r]).to(DEVICE).float()[None]
            prob = torch.softmax(model(players, ball)[0].float(), dim=0).T.cpu().numpy()
            views.append(prob if view == 0 else prob[:, [1, 0, 2]])
        p_steps[r] = np.mean(views, axis=0)
    index = np.searchsorted(steps, np.arange(len(period)), side="right") - 1
    return np.log(np.clip(p_steps[index], 1e-4, 1.0)).astype(np.float32), d


def main():
    out_root = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else SEED
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
        model = train([name(m, g, d) for m in MATCHES if m != held for g in GAPS for d in range(DRAWS)], seed)
        np.savez(target, **{"lp_" + g: predict(model, name(held, g, 0))[0] for g in GAPS})
        print(arm, held, round(time.time() - t), "s", flush=True)
    model = train([name(m, g, d) for m in MATCHES for g in GAPS for d in range(DRAWS)], seed)
    os.makedirs(os.path.join(out_root, "encoder_models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_root, "encoder_models", arm + ".pt"))
    for game in (1, 2):
        for repeat in range(3):
            for g in GAPS:
                lp, d = predict(model, os.path.join(sets, "metrica_g{}_{}_{}.npz".format(game, repeat, g)))
                np.savez(os.path.join(out_root, "metrica", "lp_{}_{}_{}_{}.npz".format(arm, game, repeat, g)),
                         lp=lp, y=d["y_raw"], period=d["period"])
    print(arm, "metrica done", flush=True)


if __name__ == "__main__":
    main()
