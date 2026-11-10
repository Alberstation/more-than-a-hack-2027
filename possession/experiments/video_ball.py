"""Ball from the tactical video with WASB (pretrained soccer weights, MIT), as an input only.

    <torch-python> possession/experiments/video_ball.py sample     # V1: ~5k frames, agreement with the ATD ball

Nothing is trained or tuned on the challenge video. The score threshold (0.5) and the blob method are
WASB's published defaults. The check is automatic and uses provided inputs only: where the ATD sees
the ball, how often WASB's top detection is within 2 m of it; in ATD gaps, how often WASB fires and
how often consecutive detections jump faster than 40 m/s. Outputs stay under data/video/ (gitignored).
WASB code and weights live outside the repository (WASB_DIR, WASB_WEIGHTS).
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402

WASB_DIR = os.environ.get("WASB_DIR", r"C:\scripts\hackathon_barsa\ext\WASB-SBDT\src")
WASB_WEIGHTS = os.environ.get("WASB_WEIGHTS", r"C:\scripts\hackathon_barsa\ext\weights\wasb_soccer_best.pth.tar")
sys.path.insert(0, WASB_DIR)
from models.hrnet import HRNet  # noqa: E402

VIDEO = os.path.join(ROOT, "data", "video", "BAR-ATH_tactical.mp4")
ATD = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.cache.v1.npz")
HOMOGRAPHY = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_homography.csv")
OUT = os.path.join(ROOT, "data", "video")
W, H = 512, 288
THRESHOLD = 0.5            # WASB default (configs/detector/tracknetv2.yaml)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLIP = 99                  # 33 triplets
FPS = 25


class Cfg(dict):
    """yaml dict with attribute access, as WASB's hydra config gives."""
    def __getattr__(self, key):
        return self[key]


def attr(value):
    if isinstance(value, dict):
        return Cfg({k: attr(v) for k, v in value.items()})
    return [attr(v) for v in value] if isinstance(value, list) else value


def model():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    with open(os.path.join(WASB_DIR, "configs", "model", "wasb.yaml")) as handle:
        cfg = yaml.safe_load(handle)
    net = HRNet(attr(cfg))
    state = torch.load(WASB_WEIGHTS, map_location="cpu", weights_only=False)["model_state_dict"]
    net.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
    return net.cuda().eval()


def homographies():
    rows = np.loadtxt(HOMOGRAPHY, delimiter=",", skiprows=1)
    return rows[:, 1:10].reshape(-1, 3, 3)


def to_pitch(Hm, px, py):
    inv = np.linalg.inv(Hm)
    d = inv[2, 0] * px + inv[2, 1] * py + inv[2, 2]
    return (inv[0, 0] * px + inv[0, 1] * py + inv[0, 2]) / d, (inv[1, 0] * px + inv[1, 1] * py + inv[1, 2]) / d


def top_blob(hm):
    """WASB's concomp blob, heat-weighted centroid; the highest-scoring blob or None."""
    if hm.max() <= THRESHOLD:
        return None
    n, labels = cv2.connectedComponents((hm > THRESHOLD).astype(np.uint8))
    best = None
    for m in range(1, n):
        ys, xs = np.where(labels == m)
        ws = hm[ys, xs]
        score = float(ws.sum())
        if best is None or score > best[2]:
            best = (float((xs * ws).sum() / ws.sum()), float((ys * ws).sum() / ws.sum()), score)
    return best


TILES = int(os.environ.get("TILES", "2"))   # TILES x TILES overlapping crops; 1 = whole frame


def tiles():
    """(x0, y0, w, h) crops covering the 1920x1080 frame with 1/8 overlap, so that the ball keeps a
    size closer to WASB's training cameras (each covers half a pitch)."""
    if TILES == 1:
        return [(0, 0, 1920, 1080)]
    w, h = int(1920 / TILES * 1.125), int(1080 / TILES * 1.125)
    xs = np.linspace(0, 1920 - w, TILES).astype(int)
    ys = np.linspace(0, 1080 - h, TILES).astype(int)
    return [(x, y, w, h) for y in ys for x in xs]


def detect_clip(net, cap, start, n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    raw = []
    for _ in range(n):
        ok, img = cap.read()
        if not ok:
            break
        raw.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    usable = len(raw) // 3 * 3
    best = [None] * usable
    for (x0, y0, w, h) in tiles():
        frames = [((cv2.resize(img[y0:y0 + h, x0:x0 + w], (W, H), interpolation=cv2.INTER_LINEAR)
                    .astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1) for img in raw[:usable]]
        x = torch.from_numpy(np.stack(frames).reshape(-1, 9, H, W))
        with torch.no_grad():
            hms = []
            for i in range(0, len(x), 16):
                pred = net(x[i:i + 16].cuda())
                pred = pred[0] if isinstance(pred, dict) else pred
                hms.append(torch.sigmoid(pred).cpu().numpy())
        hms = np.concatenate(hms).reshape(-1, H, W)
        for k, hm in enumerate(hms):
            blob = top_blob(hm)
            if blob is not None and (best[k] is None or blob[2] > best[k][2]):
                best[k] = (x0 + blob[0] * w / W, y0 + blob[1] * h / H, blob[2])
    return [(start + k, np.nan, np.nan, 0.0) if b is None else (start + k,) + b for k, b in enumerate(best)]


def runs(mask, length):
    """Start frames of windows of `length` frames where `mask` holds throughout."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return [s for s, e in zip(edges[::2], edges[1::2]) for s in range(s, e - length + 1, length)]


def sample():
    pitch = GAMES[GAME]["pitch"]
    with np.load(ATD) as f:
        ball = f["ball"][:, :2].astype(np.float64)
    seen = ~np.isnan(ball[:, 0])
    rng = np.random.default_rng(2027)
    seen_starts = np.sort(rng.choice(runs(seen, CLIP), 50, replace=False))
    gap_starts = np.sort(rng.choice(runs(~seen, CLIP), 50, replace=False))
    Hs = homographies()
    net = model()
    cap = cv2.VideoCapture(VIDEO)
    t0 = time.time()
    rows = []
    for kind, starts in (("seen", seen_starts), ("gap", gap_starts)):
        for s in starts:
            rows += [(kind == "seen",) + r for r in detect_clip(net, cap, int(s), CLIP)]
    elapsed = time.time() - t0
    rows = np.array(rows, dtype=np.float64)
    is_seen, frame, px, py, score = rows.T
    frame = frame.astype(int)
    nx, ny = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for i, f in enumerate(frame):
        if not np.isnan(px[i]):
            nx[i], ny[i] = to_pitch(Hs[f], px[i], py[i])
    mx, my = nx * pitch[0], ny * pitch[1]
    np.savez(os.path.join(OUT, "wasb_sample_t{}.npz".format(TILES)), seen=is_seen, frame=frame, px=px, py=py, score=score, x=mx, y=my)
    s = is_seen == 1
    dist = np.hypot(mx[s] - ball[frame[s], 0], my[s] - ball[frame[s], 1])
    det = ~np.isnan(mx)
    print("{} frames in {:.0f} s ({:.0f} fps incl. decoding)".format(len(rows), elapsed, len(rows) / elapsed))
    print("ATD ball seen ({} frames): WASB fires {:.1%}; within 2 m {:.1%} of all, {:.1%} of fired; median {:.2f} m".format(
        s.sum(), det[s].mean(), np.mean(dist <= 2.0), np.mean(dist[det[s]] <= 2.0), np.nanmedian(dist)))
    for q in (1.0, 3.0, 5.0):
        print("   within {:.0f} m: {:.1%} of fired".format(q, np.mean(dist[det[s]] <= q)))
    g = ~s
    gx, gy, gf = mx[g], my[g], frame[g]
    step = np.hypot(np.diff(gx), np.diff(gy)) * FPS
    ok = (np.diff(gf) == 1) & ~np.isnan(step)
    print("ATD gap ({} frames): WASB fires {:.1%}; jumps > 40 m/s {:.1%} of consecutive pairs".format(
        g.sum(), det[g].mean(), np.mean(step[ok] > 40)))
    inside = det[g] & (gx >= -2) & (gx <= pitch[0] + 2) & (gy >= -2) & (gy <= pitch[1] + 2)
    print("   detections on the pitch (+-2 m): {:.1%} of fired".format(inside.sum() / max(1, det[g].sum())))


if __name__ == "__main__":
    sample()
