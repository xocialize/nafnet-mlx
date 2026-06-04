"""NAFNet inference: restore an image (deblur/denoise)."""
from __future__ import annotations
from pathlib import Path
import numpy as np, mlx.core as mx
from PIL import Image
from .config import NAFNetConfig
from .model import NAFNet

def load_model(weights: str | Path, config: NAFNetConfig) -> NAFNet:
    m = NAFNet(config)
    m.load_weights(str(weights))
    mx.eval(m.parameters())
    return m

def restore(model: NAFNet, image_path: str | Path) -> np.ndarray:
    img = np.asarray(Image.open(image_path).convert("RGB"), np.float32) / 255.0  # HWC
    x = mx.array(img[None])                       # [1,H,W,3] NHWC, [0,1]
    y = model(x); mx.eval(y)
    out = np.clip(np.array(y)[0], 0, 1)
    return (out * 255).astype(np.uint8)

def restore_to_file(model, image_path, out_path) -> str:
    Image.fromarray(restore(model, image_path)).save(out_path)
    return str(out_path)
