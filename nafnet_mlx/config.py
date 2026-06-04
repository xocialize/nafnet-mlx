"""NAFNet configurations (the published width64 checkpoints).

Values verified against `refs/NAFNet/options/test/{REDS,SIDD,GoPro}/NAFNet-width64.yml`.
`local` selects the test-time local-pooling (TLC) variant used by the REDS/GoPro
checkpoints for their published PSNR; the weights are identical to the plain model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class NAFNetConfig:
    img_channel: int = 3
    width: int = 64
    middle_blk_num: int = 1
    enc_blk_nums: List[int] = field(default_factory=lambda: [1, 1, 1, 28])
    dec_blk_nums: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    local: bool = False                 # NAFNetLocal (TLC) at inference
    train_size: tuple = (1, 3, 256, 256)

    @classmethod
    def sidd_width64(cls) -> "NAFNetConfig":
        """Denoise. Plain NAFNet (no TLC)."""
        return cls(width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2], local=False)

    @classmethod
    def reds_width64(cls) -> "NAFNetConfig":
        """Deblur (REDS). NAFNetLocal."""
        return cls(width=64, middle_blk_num=1,
                   enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1], local=True)

    @classmethod
    def gopro_width64(cls) -> "NAFNetConfig":
        """Deblur (GoPro). NAFNetLocal."""
        return cls(width=64, middle_blk_num=1,
                   enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1], local=True)


# gdrive file ids for the official BasicSR .pth weights (from refs/NAFNet/readme.md)
GDRIVE_IDS = {
    "sidd_width64": "14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR",
    "reds_width64": "14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X",
    "gopro_width64": "1S0PVRbyTakYY9a82kujgZLbMihfNBLfC",
}
