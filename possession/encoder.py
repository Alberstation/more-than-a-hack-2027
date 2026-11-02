"""The v5 set encoder at inference: per-player tensors and a NumPy forward pass of the trained net.

Trained in a separate torch environment (experiments/encoder_train.py); the weights are exported
to .npz and run here in float64 on one thread, so the submission needs no torch and no GPU and
gives the same numbers every run. The network is a set function over players, so the number of
track slots per team does not matter: the challenge's 24 + 22 tracks go in whole.
"""

import numpy as np
from scipy.special import erf

from possession import features

STEP = 5                    # frames per encoder step: 5 steps per second at 25 fps
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)
GROUPS = 8
CHUNK = 4096


def velocity(xy, period, fps=25):
    out = np.full(xy.shape, np.nan)
    for p in np.unique(period):
        r = np.flatnonzero(period == p)
        if len(r) > 4:
            out[r[2:-2]] = (xy[r[4:]] - xy[r[:-4]]) * fps / 4
    return out


def step_frames(period):
    return np.concatenate([np.flatnonzero(period == p)[::STEP] for p in np.unique(period)])


def tensors(players, ball, team, period, pitch, steps, slots=None):
    """(players, ball) per view, view 1 with the teams swapped.

    players: (2 views, steps, 2 teams, slots, 8): x, y, vx, vy, offset to the ball, distance to the
    ball, seen; in the home attacking frame, scaled. ball: (2, steps, 6): x, y, vx, vy, seen, time
    since seen.
    """
    length, width = pitch
    slots = slots or int(max(np.sum(team == 0), np.sum(team == 1)))
    P = players[:, :, :2].astype(np.float64)
    B = ball[:, :2].astype(np.float64)
    vP, vB = velocity(P, period), velocity(B, period)
    since = np.fmin(features.frames_since_valid(B[:, 0]), 250) / 250
    out_p, out_b = [], []
    for side in (team, 1 - team):
        right = features.home_attacks_right(players, side, period)[steps]
        sign = np.where(right, 1.0, -1.0)
        px = np.where(right[:, None], P[steps, :, 0], length - P[steps, :, 0])
        py = P[steps, :, 1]
        bx = np.where(right, B[steps, 0], length - B[steps, 0])
        by = B[steps, 1]
        dx, dy = px - bx[:, None], py - by[:, None]
        f = np.stack([px / length, py / width - 0.5, vP[steps, :, 0] * sign[:, None] / 10, vP[steps, :, 1] / 10,
                      dx / 20, dy / 20, np.hypot(dx, dy) / 20], axis=-1)
        seen = ~np.isnan(px)
        f = np.concatenate([np.nan_to_num(f), seen[..., None]], axis=-1)
        grid = np.zeros((len(steps), 2, slots, f.shape[-1]))
        for t in (0, 1):
            cols = np.flatnonzero(side == t)[:slots]
            grid[:, t, :len(cols)] = f[:, cols]
        ball_seen = ~np.isnan(bx)
        b = np.stack([bx / length, by / width - 0.5, vB[steps, 0] * sign / 10, vB[steps, 1] / 10], axis=-1)
        b = np.concatenate([np.nan_to_num(b), ball_seen[:, None], since[steps, None]], axis=-1)
        out_p.append(grid)
        out_b.append(b)
    return np.stack(out_p).astype(np.float16), np.stack(out_b).astype(np.float16)


# -- forward pass ------------------------------------------------------------------------------

def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def linear(w, name, x):
    return x @ w[name + ".weight"].T + w[name + ".bias"]


def conv(w, name, x, dilation):
    """Conv1d, kernel 3, padding = dilation: x is (channels, steps)."""
    weight, bias = w[name + ".weight"], w[name + ".bias"]
    n = x.shape[1]
    padded = np.pad(x, ((0, 0), (dilation, dilation)))
    out = bias[:, None] + sum(weight[:, :, k] @ padded[:, k * dilation:k * dilation + n] for k in range(3))
    return out


def group_norm(w, name, x, eps=1e-5):
    c, n = x.shape
    g = x.reshape(GROUPS, c // GROUPS * n)
    g = (g - g.mean(axis=1, keepdims=True)) / np.sqrt(g.var(axis=1, keepdims=True) + eps)
    return g.reshape(c, n) * w[name + ".weight"][:, None] + w[name + ".bias"][:, None]


def pooled(w, players):
    """(steps, 2 * 3 * hidden): masked mean, max and closeness-to-ball softmax per team."""
    out = []
    for s in range(0, len(players), CHUNK):
        p = players[s:s + CHUNK]
        seen = p[..., 7:8]
        h = gelu(linear(w, "phi.2", gelu(linear(w, "phi.0", p)))) * seen
        count = np.maximum(seen.sum(axis=2), 1)
        mean = h.sum(axis=2) / count
        top = np.where(seen > 0, h, -1e4).max(axis=2)
        top = np.where(count > 0.5, top, 0.0)
        near = np.where(seen > 0, -p[..., 6:7] * 10, -1e4)
        near = np.exp(near - near.max(axis=2, keepdims=True))
        attn = (near / near.sum(axis=2, keepdims=True) * h).sum(axis=2)
        z = np.concatenate([mean, top, attn], axis=-1)
        out.append(z.reshape(len(p), -1))
    return np.concatenate(out)


def forward(w, players, ball):
    """Class probabilities (steps, 3) for one uninterrupted period."""
    z = pooled(w, players.astype(np.float64))
    x = linear(w, "mix", np.concatenate([z, ball.astype(np.float64)], axis=-1)).T
    for i, d in enumerate(DILATIONS):
        x = x + gelu(group_norm(w, "norms.{}".format(i), conv(w, "tcn.{}".format(i), x, d)))
    logits = w["head.weight"][:, :, 0] @ x + w["head.bias"][:, None]
    e = np.exp(logits - logits.max(axis=0))
    return (e / e.sum(axis=0)).T


def proba(weights, players, ball, team, period, pitch):
    """Per-frame probabilities (n_frames, 3): every period run whole, both views averaged, every
    set of weights averaged. Frames with period 0 are left uniform."""
    from threadpoolctl import threadpool_limits

    steps = step_frames(np.where(period > 0, period, 0))
    steps = steps[period[steps] > 0]
    P, B = tensors(players, ball, team, period, pitch, steps)
    p_steps = np.zeros((len(steps), 3))
    with threadpool_limits(1):
        for w in weights:
            for q in np.unique(period[steps]):
                r = np.flatnonzero(period[steps] == q)
                for view in (0, 1):
                    prob = forward(w, P[view][r], B[view][r])
                    p_steps[r] += prob if view == 0 else prob[:, [1, 0, 2]]
    p_steps /= 2 * len(weights)
    out = np.full((len(period), 3), 1.0 / 3)
    index = np.searchsorted(steps, np.arange(len(period)), side="right") - 1
    graded = (period > 0) & (index >= 0)
    out[graded] = p_steps[index[graded]]
    return out


def load(path):
    with np.load(path) as f:
        return {k: f[k].astype(np.float64) for k in f.files}
