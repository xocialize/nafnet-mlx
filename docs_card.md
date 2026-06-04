---
license: mit
library_name: mlx
pipeline_tag: image-to-image
tags: [mlx, image-restoration, deblurring, denoising, nafnet]
base_model: megvii-research/NAFNet
---

# NAFNet width64 (MLX) — {TASK}

Apple MLX port of **[NAFNet](https://github.com/megvii-research/NAFNet)** (Simple Baselines for
Image Restoration, ECCV 2022). Runs on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

This checkpoint: **{VARIANT}** ({TASK}). width64.

## Usage
```python
from nafnet_mlx import NAFNetConfig
from nafnet_mlx.pipeline import load_model, restore_to_file
m = load_model("model.safetensors", NAFNetConfig.{FACTORY}())
restore_to_file(m, "input.png", "output.png")
```

## Validation
Faithful NHWC port (SimpleGate, Simplified Channel Attention, channel-axis LayerNorm2d,
UNet + PixelShuffle). PT-vs-MLX full-model parity on a real image ~1e-6.
{LOCAL_NOTE}

## License & attribution
MIT. Derived from megvii-research/NAFNet (MIT). See `NOTICE`.
