"""Convert BasicSR NAFNet `.pth` weights to MLX.

Key deltas vs our module tree:
  - `*.sca.1.{weight,bias}` -> `*.sca.{weight,bias}`  (upstream SCA is Sequential(pool, conv);
    ours is a bare Conv2d, the pool is functional).
  - `ups.N.0.weight`        -> `ups.N.weight`          (upstream ups is Sequential(conv, PixelShuffle)).
Tensor transforms:
  - Conv weight  (O, I/groups, kH, kW) -> (O, kH, kW, I/groups)   [PyTorch NCHW -> MLX NHWC].
  - beta/gamma   (1, C, 1, 1)          -> (1, 1, 1, C).
Everything else (norm weight/bias, conv bias) copies through unchanged.
"""

from __future__ import annotations

import re
from typing import Dict

import mlx.core as mx


def _convert_key(k: str) -> str:
    k = re.sub(r"\.sca\.1\.", ".sca.", k)
    k = re.sub(r"^(ups\.\d+)\.0\.", r"\1.", k)
    return k


def convert_state_dict(sd: Dict) -> Dict[str, mx.array]:
    """sd: PyTorch state dict (numpy-able tensors). Returns MLX arrays keyed for NAFNet."""
    import numpy as np

    out: Dict[str, mx.array] = {}
    for k, v in sd.items():
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        nk = _convert_key(k)
        if nk.endswith(".weight") and arr.ndim == 4:          # conv weight NCHW -> NHWC
            arr = arr.transpose(0, 2, 3, 1)
        elif nk.endswith(".beta") or nk.endswith(".gamma"):    # (1,C,1,1) -> (1,1,1,C)
            arr = arr.reshape(1, 1, 1, -1)
        out[nk] = mx.array(arr)
    return out


def load_pth(path: str) -> Dict:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "params" in ck:
        return ck["params"]
    return ck
