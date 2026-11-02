"""Export trained encoder weights to .npz and check the NumPy forward pass against torch.

    <torch-python> possession/experiments/encoder_export.py <improve_live> <out_dir> arm ...

For every arm, encoder_models/<arm>.pt becomes <out_dir>/<arm>.npz (float32). Then both forward
passes run on one Metrica set and the largest probability difference is printed.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import encoder_train  # noqa: E402
from possession import encoder  # noqa: E402


def main():
    source, target, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(target, exist_ok=True)
    with np.load(os.path.join(source, "sets", "metrica_g1_0_uniform.npz")) as f:
        P, B, steps, period = f["players"], f["ball"], f["steps"], f["period"]
    r = np.flatnonzero(period[steps] == 1)[:3000]
    for arm in arms:
        state = torch.load(os.path.join(source, "encoder_models", arm + ".pt"), map_location="cpu")
        np.savez(os.path.join(target, arm + ".npz"), **{k: v.numpy().astype(np.float32) for k, v in state.items()})
        model = encoder_train.Encoder()
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            p_torch = torch.softmax(model(torch.from_numpy(P[0][r]).float()[None],
                                          torch.from_numpy(B[0][r]).float()[None])[0], dim=0).T.numpy()
        p_numpy = encoder.forward(encoder.load(os.path.join(target, arm + ".npz")), P[0][r], B[0][r])
        print(arm, "max |p_numpy - p_torch| =", float(np.abs(p_numpy - p_torch).max()),
              "argmax agree", float(np.mean(p_numpy.argmax(1) == p_torch.argmax(1))))


if __name__ == "__main__":
    main()
