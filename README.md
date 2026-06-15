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
- [x] fp32 `dist/` artifacts built for all three checkpoints (`model.safetensors` + `config.json` + model card)
- [ ] Push `mlx-community/NAFNet-{SIDD,REDS,GoPro}-width64` to the Hub — pending
- [ ] MLX-Swift wrapper (after Python)

## Tasks / checkpoints
| Checkpoint | Task | Arch | Config factory |
|---|---|---|---|
| NAFNet-SIDD-width64 | denoise | NAFNet (enc [2,2,4,8], mid 12) | `NAFNetConfig.sidd_width64()` |
| NAFNet-GoPro-width64 | deblur | NAFNetLocal/TLC (enc [1,1,1,28], mid 1) | `NAFNetConfig.gopro_width64()` |
| NAFNet-REDS-width64 | deblur | NAFNetLocal/TLC (enc [1,1,1,28], mid 1) | `NAFNetConfig.reds_width64()` |

## Usage
```python
from nafnet_mlx import NAFNetConfig
from nafnet_mlx.pipeline import load_model, restore_to_file

m = load_model("dist/NAFNet-SIDD-width64/model.safetensors", NAFNetConfig.sidd_width64())
restore_to_file(m, "noisy.png", "denoised.png")
```

`load_model(weights, config)` builds the network and loads the safetensors. `restore(model, path)`
returns an HxWx3 uint8 ndarray; `restore_to_file(model, in, out)` writes the result to disk.

> Note: this package ships **no console-script CLI** (`pyproject.toml` defines no
> `[project.scripts]`). Use the Python API above.

## Dev
```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                                 # pulls torch (parity) + pytest + gdown

# convert an upstream BasicSR .pth to MLX safetensors
python scripts/convert_to_safetensors.py --pth weights/NAFNet-SIDD-width64.pth --out w.safetensors

# build the full dist/ artifact set (safetensors + config.json + model card) for all checkpoints
python scripts/build_dist.py

# publish the dist/ folders to mlx-community (requires HF auth; network)
python scripts/publish_hf.py

pytest tests/                                           # smoke (tests/smoke) + parity (tests/parity)
```

Requires Python >= 3.10. Runtime deps: `mlx>=0.31.0`, `numpy>=1.26`, `safetensors>=0.4`,
`huggingface_hub>=0.34`, `pillow>=10`.

License: MIT (derived from megvii-research/NAFNet, MIT). See `NOTICE`.
