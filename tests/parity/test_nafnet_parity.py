"""PT vs MLX parity for NAFNet width64 (SIDD denoise, GoPro/REDS deblur).

Loads the upstream arch files in isolation (stubbing the `basicsr` package to avoid its
cv2/torchvision import chain), loads each official .pth into PyTorch (NAFNet for SIDD,
NAFNetLocal/TLC for GoPro+REDS), the converted weights into MLX, and compares full-model
outputs on identical input. Gate: max_abs < 1e-3. Skips configs whose weights are absent.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
ARCH_DIR = ROOT / "refs/NAFNet/basicsr/models/archs"

# Use a REAL image (in-distribution): deep restoration nets amplify out-of-distribution
# uniform-random input chaotically through their block stacks (GoPro's 28-block stage
# diverges >1.0 even on CPU fp32 for random noise), so random-input parity is meaningless.
# (weight stem, MLX config factory, upstream class, arch kwargs, demo image)
CASES = [
    ("NAFNet-SIDD-width64", "sidd_width64", "NAFNet",
     dict(width=64, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]), "noisy.png"),
    ("NAFNet-GoPro-width64", "gopro_width64", "NAFNetLocal",
     dict(width=64, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]), "blurry.jpg"),
    ("NAFNet-REDS-width64", "reds_width64", "NAFNetLocal",
     dict(width=64, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]), "blurry.jpg"),
]


def _load_upstream():
    pytest.importorskip("torch")
    for name in ["basicsr", "basicsr.models", "basicsr.models.archs", "basicsr.utils"]:
        m = types.ModuleType(name); m.__path__ = []; sys.modules[name] = m
    sys.modules["basicsr.utils"].get_root_logger = lambda *a, **k: None

    def load(name, fname):
        spec = importlib.util.spec_from_file_location(name, ARCH_DIR / fname)
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
        spec.loader.exec_module(mod); return mod

    load("basicsr.models.archs.arch_util", "arch_util.py")
    load("basicsr.models.archs.local_arch", "local_arch.py")
    return load("basicsr.models.archs.NAFNet_arch", "NAFNet_arch.py")


@pytest.mark.parametrize("stem,factory,pt_cls,kwargs,demo", CASES, ids=[c[0] for c in CASES])
def test_full_model_parity(stem, factory, pt_cls, kwargs, demo):
    pth = ROOT / "weights" / f"{stem}.pth"
    demo_path = ROOT / "refs/NAFNet/demo" / demo
    if not pth.exists() or not demo_path.exists():
        pytest.skip(f"missing {pth.name} or refs/NAFNet demo (dev only)")
    import torch
    from PIL import Image
    from mlx.utils import tree_unflatten

    from nafnet_mlx import NAFNet, NAFNetConfig
    from nafnet_mlx.convert import convert_state_dict, load_pth

    arch = _load_upstream()
    sd = torch.load(str(pth), map_location="cpu", weights_only=False)["params"]
    pt = getattr(arch, pt_cls)(img_channel=3, **kwargs)
    pt.load_state_dict(sd); pt.eval()

    mlx_model = NAFNet(getattr(NAFNetConfig, factory)())
    mlx_model.update(tree_unflatten(list(convert_state_dict(load_pth(str(pth))).items())))
    mx.eval(mlx_model.parameters())

    # real image, 256 crop (in-distribution)
    img = (np.asarray(Image.open(demo_path).convert("RGB"), np.float32)[:256, :256] / 255.0)[None]
    img = img.transpose(0, 3, 1, 2)
    with torch.no_grad():
        pt_out = pt(torch.from_numpy(img)).numpy()
    mlx_out = np.array(mlx_model(mx.array(img.transpose(0, 2, 3, 1)))).transpose(0, 3, 1, 2)

    max_abs = float(np.max(np.abs(pt_out - mlx_out)))
    print(f"\n{stem}: max_abs = {max_abs:.2e}")
    assert max_abs < 1e-3, f"{stem} parity fail: {max_abs:.2e}"
