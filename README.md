# nafnet-mlx

Apple MLX port of **[NAFNet](https://github.com/megvii-research/NAFNet)** — *Simple Baselines for
Image Restoration* (ECCV 2022). On-device image **deblurring** and **denoising** on Apple Silicon.

Pure stock-MLX ops (conv + SimpleGate + Simplified Channel Attention + channel-axis LayerNorm),
faithful NHWC port of the official architecture, with the TLC (test-time local converter) variant
for the deblur checkpoints.

## Status
- [x] Architecture port (NHWC) — SimpleGate / SCA / LayerNorm2d / NAFBlock / UNet+PixelShuffle
- [x] Weight conversion (BasicSR `.pth` → MLX), exact key/shape match
- [x] PT-vs-MLX parity on real images: **SIDD 5.4e-7 · GoPro 8.9e-7 · REDS 7.8e-7**
- [x] TLC local-pool (NAFNetLocal) for deblur — per-block kernels match upstream
- [x] e2e denoise + deblur verified (deblur 0.6s @ 720×1280)
- [ ] Publish `mlx-community/NAFNet-{SIDD,REDS,GoPro}-width64` — pending
- [ ] MLX-Swift wrapper (after Python)

## Tasks / checkpoints
| Checkpoint | Task | Arch |
|---|---|---|
| NAFNet-SIDD-width64 | denoise | NAFNet (enc [2,2,4,8], mid 12) |
| NAFNet-GoPro-width64 | deblur | NAFNetLocal/TLC (enc [1,1,1,28], mid 1) |
| NAFNet-REDS-width64 | deblur | NAFNetLocal/TLC (enc [1,1,1,28], mid 1) |

## Usage
```python
from nafnet_mlx import NAFNetConfig
from nafnet_mlx.pipeline import load_model, restore_to_file
m = load_model("NAFNet-SIDD-width64.safetensors", NAFNetConfig.sidd_width64())
restore_to_file(m, "noisy.png", "denoised.png")
```

## Dev
```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python scripts/convert_to_safetensors.py --pth weights/NAFNet-SIDD-width64.pth --out w.safetensors
pytest tests/
```
License: MIT (derived from megvii-research/NAFNet, MIT). See `NOTICE`.
