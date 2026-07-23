"""Unit tests for utils.utils_image.

Covers the pure numpy/torch conversion, geometry, resize and metric helpers
with numerical assertions (round-trips, hand-computed references). File-IO
helpers are exercised against synthetic PNGs from the ``write_rgb_png`` fixture.
"""
import math

import numpy as np
import pytest
import torch

import utils.utils_image as util


# ------------------------------------------------------------------
# deterministic helpers
# ------------------------------------------------------------------
def _rgb(h=32, w=32, seed=0, dtype=np.uint8):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=dtype)


# ------------------------------------------------------------------
# uint <-> single
# ------------------------------------------------------------------
def test_uint2single_is_float01():
    img = _rgb()
    s = util.uint2single(img)
    assert s.dtype == np.float32
    assert s.min() >= 0.0 and s.max() <= 1.0
    # inverse up to single-precision rounding
    assert np.array_equal(util.single2uint(s), img)


def test_single2uint_clips_and_rounds():
    s = np.array([[-0.4, 0.5, 1.4]], dtype=np.float32)
    out = util.single2uint(s)
    assert out.dtype == np.uint8
    # 0.5*255 = 127.5 -> np.round rounds to even -> 128
    assert np.array_equal(out, np.array([[0, 128, 255]], dtype=np.uint8))


def test_uint162single_single2uint16_roundtrip():
    img = (np.random.rand(8, 8, 3) * 65535).astype(np.uint16)
    assert np.array_equal(util.single2uint16(util.uint162single(img)), img)


# ------------------------------------------------------------------
# numpy <-> tensor
# ------------------------------------------------------------------
def test_uint_tensor_roundtrip():
    img = _rgb()
    t = util.uint2tensor3(img)          # 3xHxW float [0,1]
    assert t.shape == (3, 32, 32)
    assert np.array_equal(util.tensor2uint(t), img)
    t4 = util.uint2tensor4(img)         # 1x3xHxW
    assert t4.shape == (1, 3, 32, 32)


def test_single_tensor_roundtrip():
    img = _rgb()
    s = util.uint2single(img)
    t = util.single2tensor3(s)          # 3xHxW
    assert np.allclose(util.tensor2single(t), s)
    out = util.single2tensor4(s)
    assert out.shape == (1, 3, 32, 32)


def test_tensor2single3_keeps_hwc_for_2d():
    t = torch.rand(1, 4, 4)             # single-channel, 1xHxW
    out = util.tensor2single3(t)
    assert out.shape == (4, 4, 1)


# ------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------
@pytest.mark.parametrize("sf", [2, 3, 4])
def test_modcrop_multiples(sf):
    img = np.random.randint(0, 256, (21, 19, 3), dtype=np.uint8)
    out = util.modcrop(img, sf)
    assert out.shape[0] % sf == 0
    assert out.shape[1] % sf == 0


def test_modcrop_2d():
    img = np.arange(25, dtype=np.uint8).reshape(5, 5)
    assert util.modcrop(img, 2).shape == (4, 4)


def test_shave_border():
    img = np.random.rand(10, 10, 3)
    assert util.shave(img, 2).shape == (6, 6, 3)


@pytest.mark.parametrize("mode", [0, 1, 2, 3, 4, 5, 6, 7])
def test_augment_img_all_modes_keep_shape(mode):
    img = _rgb(16, 16)
    out = util.augment_img(img, mode=mode)
    assert out.shape == img.shape


def test_augment_img_mode0_identity():
    img = _rgb(16, 16)
    assert np.array_equal(util.augment_img(img, mode=0), img)


# ------------------------------------------------------------------
# color conversion round-trips
# ------------------------------------------------------------------
def test_rgb_ycbcr_roundtrip():
    img = _rgb(20, 20).astype(np.float32) / 255.0
    # pass copies: the color helpers previously mutated their input in place
    back = util.ycbcr2rgb(util.rgb2ycbcr(img.copy(), only_y=False).copy())
    assert np.allclose(back, img, atol=1e-4)
    y = util.rgb2ycbcr(img.copy(), only_y=True)
    assert y.ndim == 2 or y.shape[-1] == 1


def test_bgr2ycbcr_matches_rgb_on_swapped():
    rgb = _rgb(20, 20).astype(np.float32) / 255.0
    bgr = rgb[:, :, ::-1].copy()
    y_from_rgb = util.rgb2ycbcr(rgb.copy(), only_y=True)
    y_from_bgr = util.bgr2ycbcr(bgr.copy(), only_y=True)
    assert np.allclose(y_from_rgb, y_from_bgr)


# ------------------------------------------------------------------
# resize
# ------------------------------------------------------------------
def test_imresize_np_half_size():
    img = np.random.rand(16, 16, 3).astype(np.float32)
    out = util.imresize_np(img, 0.5)
    assert out.shape[:2] == (8, 8)


def test_imresize_np_matches_tensor_imresize():
    img = np.random.rand(12, 12, 3).astype(np.float32)
    np_out = util.imresize_np(img, 0.5)
    t_out = util.imresize(torch.from_numpy(img).permute(2, 0, 1), 0.5)
    t_out = t_out.permute(1, 2, 0).numpy()
    assert np.allclose(np_out, t_out, atol=1e-5)


def test_imresize_np_and_upscale():
    img = np.random.rand(8, 8, 3).astype(np.float32)
    out = util.imresize_np(img, 2.0)
    assert out.shape[:2] == (16, 16)


# ------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------
def test_calculate_psnr_identical_is_inf():
    img = _rgb(16, 16)
    assert util.calculate_psnr(img, img) == float("inf")


def test_calculate_psnr_known_mse():
    a = np.zeros((1, 1), dtype=np.uint8)
    b = np.full((1, 1), 255, dtype=np.uint8)
    mse = 255.0 ** 2
    expected = 20 * math.log10(255.0 / math.sqrt(mse))
    assert util.calculate_psnr(a, b) == pytest.approx(expected)


def test_calculate_ssim_identical_is_one():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    assert util.calculate_ssim(img, img) == pytest.approx(1.0, abs=1e-4)


def test_calculate_psnrb_runs_and_shape():
    a = _rgb(32, 32)
    b = np.clip(a + 5, 0, 255).astype(np.uint8)
    val = util.calculate_psnrb(a, b)
    assert isinstance(val, float)


# ------------------------------------------------------------------
# file IO
# ------------------------------------------------------------------
def test_read_img_and_imwrite_roundtrip(write_rgb_png, tmp_path):
    p = tmp_path / "x.png"
    write_rgb_png(p, h=24, w=24, seed=1)
    img = util.read_img(str(p))              # HWC float32 BGR [0,1]
    assert img.dtype == np.float32
    assert img.shape[2] == 3
    out = tmp_path / "y.png"
    util.imwrite(img, str(out))
    assert out.exists()


def test_imread_uint_loads_rgb(write_rgb_png, tmp_path):
    p = tmp_path / "z.png"
    write_rgb_png(p, h=20, w=20, seed=2)
    img = util.imread_uint(str(p), n_channels=3)
    assert img.shape == (20, 20, 3)
    assert img.dtype == np.uint8
