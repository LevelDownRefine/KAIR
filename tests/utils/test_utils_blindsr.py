"""Unit tests for utils.utils_blindsr.

Covers the deterministic kernel/geometry/degradation helpers with numerical
assertions, and the random degradation operators with shape/range checks
(seeded where possible). The heavy ``degradation_bsrgan*`` pipelines are checked
for output shape and value range only.
"""
import random

import numpy as np
import pytest
import torch

import utils.utils_blindsr as blindsr
import utils.utils_image as util


def _img(h=64, w=64, seed=0):
    rng = np.random.RandomState(seed)
    return rng.rand(h, w, 3).astype(np.float32)


# ------------------------------------------------------------------
# kernels
# ------------------------------------------------------------------
def test_gm_blur_kernel_normalized():
    k = blindsr.gm_blur_kernel(mean=[0, 0], cov=[[4.0, 0.0], [0.0, 4.0]], size=11)
    assert abs(k.sum() - 1.0) < 1e-6
    assert np.all(k >= 0)


def test_anisotropic_Gaussian_isotropic_normalized_symmetric():
    k = blindsr.anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6)
    assert abs(k.sum() - 1.0) < 1e-6
    assert np.allclose(k, k.T)


def test_analytic_kernel_normalized_shape():
    k = blindsr.fspecial_gaussian(3, 1.0)
    big = blindsr.analytic_kernel(k)
    assert big.shape == (5, 5)
    assert abs(big.sum() - 1.0) < 1e-6


def test_fspecial_gaussian_normalized():
    h = blindsr.fspecial_gaussian(5, 1.0)
    assert abs(h.sum() - 1.0) < 1e-6
    assert np.allclose(h, h.T)


def test_fspecial_laplacian():
    h = blindsr.fspecial_laplacian(0.5)
    expected = np.array([[1/3, 1/3, 1/3], [1/3, -8/3, 1/3], [1/3, 1/3, 1/3]])
    assert np.allclose(h, expected)


def test_fspecial_dispatch():
    assert np.allclose(blindsr.fspecial('gaussian', 5, 1.0),
                       blindsr.fspecial_gaussian(5, 1.0))


# ------------------------------------------------------------------
# geometry / degradation (deterministic)
# ------------------------------------------------------------------
def test_modcrop_np_multiples():
    img = np.random.rand(21, 19, 3)
    out = blindsr.modcrop_np(img, sf=4)
    assert out.shape[0] % 4 == 0
    assert out.shape[1] % 4 == 0


def test_shift_pixel_zero_is_identity():
    x = _img(8, 8)
    assert np.allclose(blindsr.shift_pixel(x, sf=2, upper_left=True), x)


def test_bicubic_degradation_downsamples():
    x = _img(16, 16)
    assert blindsr.bicubic_degradation(x, sf=2).shape[:2] == (8, 8)


def test_srmd_degradation_shape():
    x = _img(16, 16)
    k = blindsr.fspecial_gaussian(5, 1.0)
    assert blindsr.srmd_degradation(x, k, sf=2).shape[:2] == (8, 8)


def test_classical_degradation_shape():
    x = _img(16, 16)
    k = np.ones((3, 3)) / 9.0
    assert blindsr.classical_degradation(x, k, sf=2).shape[:2] == (8, 8)


def test_blur_preserves_shape():
    img = torch.rand(1, 3, 16, 16)
    # blur expects the kernel to share the image dtype (float32 here)
    k = torch.from_numpy(blindsr.fspecial_gaussian(5, 1.0)).to(torch.float32)
    k = k.unsqueeze(0).unsqueeze(0)
    out = blindsr.blur(img, k)
    assert out.shape == (1, 3, 16, 16)


# ------------------------------------------------------------------
# random degradation operators (shape / range only)
# ------------------------------------------------------------------
@pytest.mark.parametrize("fn", [
    blindsr.add_Gaussian_noise,
    blindsr.add_speckle_noise,
])
def test_noise_operators_range_and_shape(fn):
    random.seed(0)
    np.random.seed(0)
    img = _img(32, 32)
    out = fn(img.copy())
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_add_jpeg_noise_range_and_shape():
    img = _img(32, 32)
    out = blindsr.add_JPEG_noise(img.copy())
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_add_poisson_noise_range_and_shape():
    random.seed(0)
    np.random.seed(0)
    img = _img(32, 32)
    out = blindsr.add_Poisson_noise(img.copy())
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_add_blur_range_and_shape():
    random.seed(0)
    np.random.seed(0)
    img = _img(32, 32)
    out = blindsr.add_blur(img.copy(), sf=4)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_random_crop_shapes():
    random.seed(0)
    lq = np.random.rand(100, 100, 3).astype(np.float32)
    hq = np.random.rand(400, 400, 3).astype(np.float32)
    out_lq, out_hq = blindsr.random_crop(lq, hq, sf=4, lq_patchsize=64)
    assert out_lq.shape == (64, 64, 3)
    assert out_hq.shape == (256, 256, 3)


def test_degradation_bsrgan_shapes_and_range():
    random.seed(0)
    np.random.seed(0)
    img = _img(300, 300)
    lq, hq = blindsr.degradation_bsrgan(img.copy(), sf=4, lq_patchsize=72)
    assert lq.shape == (72, 72, 3)
    assert hq.shape == (288, 288, 3)
    assert lq.min() >= 0.0 and lq.max() <= 1.0
    assert hq.min() >= 0.0 and hq.max() <= 1.0
