"""PT vs MLX parity for NAFNet (SIDD-width64).

Loads the upstream arch files in isolation (stubbing the `basicsr` package so we avoid
its cv2/torchvision import chain), loads the official .pth into PyTorch, the converted
weights into MLX, and compares full-model outputs on identical input.

Gate: full-model max_abs < 1e-3 (fp32). Skips if torch / weights unavailable.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
PTH = ROOT / "weights/NAFNet-SIDD-width64.pth"
ARCH_DIR = ROOT / "refs/NAFNet/basicsr/models/archs"

pytestmark = pytest.mark.skipif(
    not PTH.exists() or not ARCH_DIR.exists(),
    reason="needs refs/NAFNet + weights/NAFNet-SIDD-width64.pth (dev only)",
)


def _load_upstream_nafnet():
    """Import upstream NAFNet_arch without triggering basicsr.__init__ heavy deps."""
    pytest.importorskip("torch")
    for name in ["basicsr", "basicsr.models", "basicsr.models.archs", "basicsr.utils"]:
        m = types.ModuleType(name)
        m.__path__ = []  # mark as package
        sys.modules[name] = m
    sys.modules["basicsr.utils"].get_root_logger = lambda *a, **k: None

    def load(name, fname):
        spec = importlib.util.spec_from_file_location(name, ARCH_DIR / fname)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    load("basicsr.models.archs.arch_util", "arch_util.py")
    load("basicsr.models.archs.local_arch", "local_arch.py")
    return load("basicsr.models.archs.NAFNet_arch", "NAFNet_arch.py")


def test_sidd_full_model_parity():
    import torch

    from nafnet_mlx import NAFNet, NAFNetConfig
    from nafnet_mlx.convert import convert_state_dict, load_pth

    arch = _load_upstream_nafnet()
    cfg = dict(img_channel=3, width=64, middle_blk_num=12,
               enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    pt = arch.NAFNet(**cfg)
    sd = torch.load(str(PTH), map_location="cpu", weights_only=False)["params"]
    pt.load_state_dict(sd)
    pt.eval()

    mlx_model = NAFNet(NAFNetConfig.sidd_width64())
    mlx_model.update(__import__("mlx.utils", fromlist=["tree_unflatten"]).tree_unflatten(
        list(convert_state_dict(load_pth(str(PTH))).items())))
    mx.eval(mlx_model.parameters())

    rng = np.random.RandomState(0)
    img = rng.rand(1, 3, 64, 64).astype("float32")          # NCHW for torch

    with torch.no_grad():
        pt_out = pt(torch.from_numpy(img)).numpy()           # [1,3,64,64]
    mlx_out = np.array(mlx_model(mx.array(img.transpose(0, 2, 3, 1))))  # NHWC
    mlx_out = mlx_out.transpose(0, 3, 1, 2)                  # back to NCHW

    max_abs = float(np.max(np.abs(pt_out - mlx_out)))
    print(f"\nNAFNet SIDD full-model max_abs = {max_abs:.2e}")
    assert max_abs < 1e-3, f"parity fail: {max_abs:.2e}"
