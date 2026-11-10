"""A1: referee whistle from the tactical video's audio, no training (fixed rule, decided before looking).

    ffmpeg -i data/video/BAR-ATH_tactical.mp4 -vn -ac 1 -ar 16000 data/video/BAR-ATH_tactical_16k.wav
    python possession/experiments/audio_whistle.py

Per video frame (40 ms hop): share of spectral energy in 2.5-4.5 kHz (the whistle's fundamental) and
how tonal that band is (peak bin over band mean). A whistle = both above median + 6 MAD for >= 3
frames (0.12 s). Check against Smart Tagging only: every kick-off (start, after goal) is whistled,
so a usable detector fires within +-5 s of most of them, well above the chance rate of random frames.
Pass: hit rate >= 50 % and >= 3 x chance. Output stays in data/video/ (gitignored).
"""
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from load_smart_tagging import load  # noqa: E402

WAV = os.path.join(ROOT, "data", "video", "BAR-ATH_tactical_16k.wav")
SR, HOP, WIN = 16000, 640, 1024           # 640 samples = 40 ms = one video frame
BAND = (2500.0, 4500.0)
K_MAD, MIN_FRAMES, TOL_S = 6.0, 3, 5.0


def features():
    with wave.open(WAV) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    n = (len(audio) - WIN) // HOP + 1
    freqs = np.fft.rfftfreq(WIN, 1 / SR)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    window = np.hanning(WIN).astype(np.float32)
    share, tonal = np.empty(n), np.empty(n)
    for start in range(0, n, 4096):
        idx = np.arange(start, min(n, start + 4096))
        frames = audio[idx[:, None] * HOP + np.arange(WIN)] * window
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-12
        b = power[:, band]
        share[idx] = b.sum(1) / power.sum(1)
        tonal[idx] = b.max(1) / b.mean(1)
    return share, tonal


def above(x):
    med = np.median(x)
    return x > med + K_MAD * np.median(np.abs(x - med))


def whistles(share, tonal):
    on = above(share) & above(tonal)
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
    return np.array([s for s, e in zip(edges[::2], edges[1::2]) if e - s >= MIN_FRAMES])


def main():
    share, tonal = features()
    starts = whistles(share, tonal)
    n = len(share)
    hit_mask = np.zeros(n, dtype=bool)
    tol = int(TOL_S * 25)
    for s in starts:
        hit_mask[max(0, s - tol):s + tol + 1] = True
    kicks = [e for e in load() if (e["code"] or "").lower() == "set pieces"
             and (e["tags"].get("Type") or "").lower().startswith("kick off")]
    hits = [bool(hit_mask[e["start_frame"]:e["end_frame"] + 1].any()) for e in kicks]
    chance = hit_mask.mean()
    np.savez(os.path.join(ROOT, "data", "video", "whistles.npz"), starts=starts, share=share.astype(np.float32),
             tonal=tonal.astype(np.float32))
    print("{} whistles detected ({:.1f} per 10 min of audio)".format(len(starts), len(starts) / (n / 25 / 600)))
    print("kick-offs with a whistle within +-{:.0f} s: {}/{} = {:.0%}; chance {:.0%}".format(
        TOL_S, sum(hits), len(hits), np.mean(hits), chance))
    ok = np.mean(hits) >= 0.5 and np.mean(hits) >= 3 * chance
    print("A1:", "pasa" if ok else "fuera")


if __name__ == "__main__":
    main()
