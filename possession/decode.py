"""Viterbi decoding of a three-state H / A / D chain.

The baseline's majority vote treats every frame independently and then blurs them. A possession
is an interval, so the decoder below scores whole label sequences instead: the per-frame evidence
(log-likelihood ratios, one column per state) is summed along a path, every change of state pays a
fixed cost, and the best path wins. A team change therefore needs sustained contrary evidence, not
one close call, and a frame with no evidence simply continues the current state.
"""

import numpy as np

STATES = np.array(["H", "A", "D"])


def transition_matrix(switch_cost, dead_cost):
    """Log transition scores. H<->A and alive<->D are charged separately."""
    t = np.zeros((3, 3))
    t[0, 1] = t[1, 0] = -switch_cost
    t[0, 2] = t[1, 2] = t[2, 0] = t[2, 1] = -dead_cost
    return t


def viterbi(emissions, transitions, initial=None):
    """Most likely state index per frame. emissions is (n_frames, 3) of log scores."""
    n = len(emissions)
    score = emissions[0] + (np.zeros(3) if initial is None else initial)
    back = np.empty((n, 3), dtype=np.int8)
    back[0] = np.arange(3)
    for i in range(1, n):
        candidates = score[:, None] + transitions
        back[i] = np.argmax(candidates, axis=0)
        score = candidates[back[i], np.arange(3)] + emissions[i]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path
