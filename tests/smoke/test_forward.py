import mlx.core as mx
from nafnet_mlx import NAFNet, NAFNetConfig

def test_sidd_forward_shape_and_padding():
    m = NAFNet(NAFNetConfig.sidd_width64())
    y = m(mx.zeros((1, 64, 96, 3)))   # non-multiple width exercises check_image_size
    mx.eval(y)
    assert y.shape == (1, 64, 96, 3)

def test_reds_config():
    c = NAFNetConfig.reds_width64()
    assert c.width == 64 and c.enc_blk_nums == [1,1,1,28] and c.middle_blk_num == 1 and c.local
    m = NAFNet(c)
    assert len(m.encoders) == 4 and len(m.decoders) == 4
