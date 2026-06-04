"""NAFNet in MLX — faithful port of `megvii-research/NAFNet` (`NAFNet_arch.py`).

Isomorphic to upstream (same class names / decomposition): `SimpleGate`, `NAFBlock`,
`NAFNet`. The only framework change is **NHWC** tensor layout (MLX convs are NHWC vs
PyTorch NCHW), so: channel ops act on the last axis, `LayerNorm2d` is a layer-norm over
the last (channel) axis, SCA pools over the spatial axes (1, 2), and `beta`/`gamma` are
`(1,1,1,c)`. The forward keeps the same op order, UNet skips, global residual, and
`check_image_size` padding as upstream.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import NAFNetConfig


class LayerNorm2d(nn.Module):
    """Channel-axis LayerNorm (upstream `LayerNorm2d`): normalize over C per pixel.

    In NHWC the channel axis is last, so this is a standard layer norm over axis -1.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mu = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean((x - mu) ** 2, axis=-1, keepdims=True)
        y = (x - mu) * mx.rsqrt(var + self.eps)
        return y * self.weight + self.bias


class SimpleGate(nn.Module):
    """x1 * x2 where (x1, x2) = split over channels."""

    def __call__(self, x: mx.array) -> mx.array:
        x1, x2 = mx.split(x, 2, axis=-1)
        return x1 * x2


def _avgpool_global(x: mx.array) -> mx.array:
    """AdaptiveAvgPool2d(1): mean over spatial axes (H, W), keepdims. NHWC."""
    return mx.mean(x, axis=(1, 2), keepdims=True)


def local_avgpool(x: mx.array, kh: int, kw: int) -> mx.array:
    """TLC local windowed average pool (NHWC), matching `local_arch.AvgPool2d`.

    Fixed kernel (kh, kw) via a summed-area table; output is edge-padded back to the
    input spatial size so SCA produces a spatially-varying attention map. Falls back to
    global mean when the window covers the whole input (upstream's `>=` branch).
    """
    _, h, w, _ = x.shape
    k1, k2 = min(h, kh), min(w, kw)
    if k1 >= h and k2 >= w:
        return _avgpool_global(x)
    s = mx.cumsum(mx.cumsum(x, axis=1), axis=2)
    s = mx.pad(s, [(0, 0), (1, 0), (1, 0), (0, 0)])              # prepend zero row/col
    out = (s[:, k1:, k2:, :] + s[:, :-k1, :-k2, :]
           - s[:, :-k1, k2:, :] - s[:, k1:, :-k2, :]) / (k1 * k2)
    _h, _w = out.shape[1], out.shape[2]
    out = mx.pad(out, [(0, 0), ((h - _h) // 2, (h - _h + 1) // 2),
                       ((w - _w) // 2, (w - _w + 1) // 2), (0, 0)], mode="edge")
    return out


class NAFBlock(nn.Module):
    def __init__(self, c: int, DW_Expand: int = 2, FFN_Expand: int = 2):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)

        # Simplified Channel Attention: global pool -> 1x1 conv (no activation)
        self.sca = nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True)

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = mx.zeros((1, 1, 1, c))
        self.gamma = mx.zeros((1, 1, 1, c))

        self.pool_kernel = None   # (kh, kw) for TLC local pool; None -> global

    def _pool(self, x: mx.array) -> mx.array:
        if self.pool_kernel is None:
            return _avgpool_global(x)
        return local_avgpool(x, self.pool_kernel[0], self.pool_kernel[1])

    def __call__(self, inp: mx.array) -> mx.array:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(self._pool(x))
        x = self.conv3(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        return y + x * self.gamma


def pixel_shuffle(x: mx.array, r: int) -> mx.array:
    """NHWC PixelShuffle, matching PyTorch's (C, r, r) channel grouping.

    [N,H,W,C*r*r] -> [N,H*r,W*r,C], channel index c*r*r + i*r + j -> (c, offset i,j).
    """
    n, h, w, crr = x.shape
    c = crr // (r * r)
    x = x.reshape(n, h, w, c, r, r)
    x = x.transpose(0, 1, 4, 2, 5, 3)      # [N,H,r,W,r,C]
    return x.reshape(n, h * r, w * r, c)


class NAFNet(nn.Module):
    def __init__(self, config: NAFNetConfig | None = None):
        super().__init__()
        self.config = config = config or NAFNetConfig()
        width = config.width

        self.intro = nn.Conv2d(config.img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, config.img_channel, 3, padding=1, bias=True)

        self.encoders, self.decoders = [], []
        self.ups, self.downs = [], []

        chan = width
        for num in config.enc_blk_nums:
            self.encoders.append([NAFBlock(chan) for _ in range(num)])
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, stride=2))
            chan *= 2

        self.middle_blks = [NAFBlock(chan) for _ in range(config.middle_blk_num)]

        for num in config.dec_blk_nums:
            self.ups.append(nn.Conv2d(chan, chan * 2, 1, bias=False))   # + pixel_shuffle(2)
            chan //= 2
            self.decoders.append([NAFBlock(chan) for _ in range(num)])

        self.padder_size = 2 ** len(self.encoders)

        if config.local:
            self._setup_local(config.train_size)

    def _setup_local(self, train_size: tuple) -> None:
        """Assign each block's fixed TLC kernel from its training resolution (NAFNetLocal).

        Mirrors `Local_Base.convert`: kernel is cached per block at train res, where
        base_size = 1.5 * train spatial. kernel(res) = res * base_size // train_spatial = 1.5*res.
        """
        _, _, th, tw = train_size
        bh, bw = int(th * 1.5), int(tw * 1.5)

        def kern(rh, rw):
            return (rh * bh // th, rw * bw // tw)

        rh, rw = th, tw
        for stage in self.encoders:
            for blk in stage:
                blk.pool_kernel = kern(rh, rw)
            rh, rw = rh // 2, rw // 2
        for blk in self.middle_blks:
            blk.pool_kernel = kern(rh, rw)
        for stage in self.decoders:
            rh, rw = rh * 2, rw * 2
            for blk in stage:
                blk.pool_kernel = kern(rh, rw)

    @staticmethod
    def _run(blocks, x):
        for b in blocks:
            x = b(x)
        return x

    def check_image_size(self, x: mx.array) -> mx.array:
        _, h, w, _ = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        if ph or pw:
            x = mx.pad(x, [(0, 0), (0, ph), (0, pw), (0, 0)])
        return x

    def __call__(self, inp: mx.array) -> mx.array:
        """inp: [N, H, W, C] in [0, 1]. Returns restored image, same H, W."""
        _, H, W, _ = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = self._run(encoder, x)
            encs.append(x)
            x = down(x)

        x = self._run(self.middle_blks, x)

        for decoder, up, skip in zip(self.decoders, self.ups, encs[::-1]):
            x = pixel_shuffle(up(x), 2)
            x = x + skip
            x = self._run(decoder, x)

        x = self.ending(x)
        x = x + inp
        return x[:, :H, :W, :]
