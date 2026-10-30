"""Read the tactical ATD once into dense numpy arrays, cached next to the data.

Parsing 199319 lines of text takes a minute or two; every later run reads the .npz cache in under
a second. The cache lives under data/ (gitignored), because it is a frame-level export of the
tracking and must not end up in a public repository.

Arrays, all indexed by video frame (ATD frame_id - 1):
    players  (n_frames, n_slots, 3) float32   x, y in metres, speed in m/s; NaN when missing
    ball     (n_frames, 3)          float32   x, y in metres, speed in m/s; NaN when missing
    team     (n_slots,)             int8      0 home, 1 away
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

CACHE_VERSION = 1


def _cache_path(raw_path):
    return os.path.splitext(raw_path)[0] + ".cache.v{}.npz".format(CACHE_VERSION)


def _parse(raw_path, n_frames, n_slots, pitch):
    length, width = pitch
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            groups = parts[1].split(";")
            values = np.array([g.split(",") for g in groups], dtype=np.float32)
            players[frame, :len(groups)] = values
            if parts[2][0] != "N":
                ball[frame] = np.array(parts[2].split(","), dtype=np.float32)
    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return players, ball


def load(game=GAME, data_dir=None):
    """(players, ball, team) for the tactical source, parsing and caching on first use."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    cache = _cache_path(raw_path)
    if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(raw_path):
        with np.load(cache) as stored:
            return stored["players"], stored["ball"], team
    players, ball = _parse(raw_path, source_frames("tactical", game), len(team),
                           GAMES[game]["pitch"])
    np.savez(cache, players=players, ball=ball)
    return players, ball, team


if __name__ == "__main__":
    players, ball, team = load()
    print("players", players.shape, "ball", ball.shape, "slots home/away",
          int(np.sum(team == 0)), int(np.sum(team == 1)))
    print("ball present on {:.1%} of frames".format(np.mean(~np.isnan(ball[:, 0]))))
