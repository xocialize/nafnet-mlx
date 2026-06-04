"""Build per-variant mlx-community dist dirs (no upload)."""
import json, shutil
from dataclasses import asdict
from pathlib import Path
from nafnet_mlx.config import NAFNetConfig

ROOT = Path(__file__).resolve().parents[1]
CARD = (ROOT / "docs_card.md").read_text()
NOTICE = (ROOT / "NOTICE").read_text()
VARIANTS = {
    "NAFNet-SIDD-width64":  ("Image denoising", "SIDD",  "sidd_width64",  ""),
    "NAFNet-GoPro-width64": ("Image deblurring", "GoPro", "gopro_width64", "Uses NAFNetLocal (TLC) local pooling."),
    "NAFNet-REDS-width64":  ("Image deblurring", "REDS",  "reds_width64",  "Uses NAFNetLocal (TLC) local pooling."),
}
for stem,(task,variant,factory,local) in VARIANTS.items():
    d = ROOT / "dist" / stem; d.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT/"weights"/f"{stem}.safetensors", d/"model.safetensors")
    cfg = getattr(NAFNetConfig, factory)()
    json.dump(asdict(cfg), open(d/"config.json","w"), indent=2)
    card = (CARD.replace("{TASK}",task).replace("{VARIANT}",variant)
                .replace("{FACTORY}",factory).replace("{LOCAL_NOTE}",local))
    (d/"README.md").write_text(card)
    (d/"NOTICE").write_text(NOTICE)
    print("built", d.name, "->", [p.name for p in d.iterdir()])
