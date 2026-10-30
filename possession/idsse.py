"""IDSSE / DFL open data (Bassek et al., Sci Data 2025, CC-BY 4.0) in the same arrays as atd.py.

Seven Bundesliga 2022/23 matches at 25 Hz with BallStatus (alive/dead) and BallPossession
(home/away) on every frame -- the H / A / D labels of this challenge, on OTHER games. They are
the only labelled data this repository fits or selects anything on. See docs/plan, line M5.

    python possession/idsse.py              # download all seven and build the caches

Files go to data/external/idsse/ (gitignored). Arrays per match, the same layout as atd.load():
    players (n, slots, 3)  x, y in metres from the corner, speed in m/s, NaN when absent
    ball    (n, 3)
    team    (slots,)       0 home, 1 away
    labels  (n,)           "H" / "A" / "D"
    period  (n,)           1 or 2
"""

import os
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

FOLDER = os.path.join(DATA_DIR, "external", "idsse")
MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
FPS = 25
CACHE_VERSION = 1


def _download(url, path):
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("  downloading", os.path.basename(path))
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def _convert(match_id):
    from kloppy import sportec
    from kloppy._providers.sportec import get_IDSSE_url
    from kloppy.domain import BallState

    meta = _download(get_IDSSE_url(match_id, "meta"),
                     os.path.join(FOLDER, match_id + "_meta.xml"))
    raw = _download(get_IDSSE_url(match_id, "tracking"),
                    os.path.join(FOLDER, match_id + "_tracking.xml"))
    dataset = sportec.load_tracking(raw_data=raw, meta_data=meta, only_alive=False,
                                    coordinates="kloppy")
    length = dataset.metadata.pitch_dimensions.pitch_length
    width = dataset.metadata.pitch_dimensions.pitch_width
    home, away = dataset.metadata.teams
    roster = [p for t in (home, away) for p in t.players]
    slot = {p.player_id: i for i, p in enumerate(roster)}
    team = np.array([0 if p.team == home else 1 for p in roster], dtype=np.int8)

    n = len(dataset.records)
    players = np.full((n, len(roster), 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    labels = np.full(n, "D", dtype="<U1")
    period = np.zeros(n, dtype=np.int8)
    for i, frame in enumerate(dataset.records):
        period[i] = frame.period.id
        for player, data in frame.players_data.items():
            if data.coordinates is not None and player.player_id in slot:
                players[i, slot[player.player_id], :2] = (data.coordinates.x * length,
                                                         data.coordinates.y * width)
        if frame.ball_coordinates is not None:
            ball[i, :2] = (frame.ball_coordinates.x * length, frame.ball_coordinates.y * width)
        if frame.ball_state == BallState.ALIVE and frame.ball_owning_team is not None:
            labels[i] = "H" if frame.ball_owning_team == home else "A"

    # Speeds from positions, per period, the way the provider's own speed would be: central
    # difference over +/- 2 frames, so a 1-frame jitter does not read as a sprint.
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        for array in (players, ball):
            xy = array[rows, ..., :2]
            v = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
            v[2:-2] = np.linalg.norm(xy[4:] - xy[:-4], axis=-1) * FPS / 4
            array[rows, ..., 2] = v
    return dict(players=players, ball=ball, team=team, labels=labels, period=period,
                pitch=np.array([length, width]))


def load(match_id):
    """Arrays for one match, converted once and cached as .npz."""
    cache = os.path.join(FOLDER, "{}.v{}.npz".format(match_id, CACHE_VERSION))
    if not os.path.isfile(cache):
        np.savez_compressed(cache, **_convert(match_id))
    with np.load(cache) as stored:
        return {key: stored[key] for key in stored.files}


if __name__ == "__main__":
    for match in MATCHES:
        data = load(match)
        labels = data["labels"]
        print("{}  {} frames  ball {:.1%}  H {:.1%}  A {:.1%}  D {:.1%}".format(
            match, len(labels), np.mean(~np.isnan(data["ball"][:, 0])),
            *[np.mean(labels == k) for k in "HAD"]))
