"""Unit tests for utils.utils_sisr.

Covers the numpy/torch helpers (kernel generation, PCA, pixel shifts, complex
helpers, OTF, up/down sampling, degradation operators) and the migrated
torch.fft wrappers (``rfft``/``irfft``/``fft``/``ifft``/``p2o``/``INVLS_pytorch``)
with numerical assertions. The wrappers were migrated off the removed
top-level ``torch.rfft``/``torch.irfft``/``torch.fft``/``torch.ifft`` onto the
``torch.fft`` namespace (native complex bridged via ``view_as_real`` /
``view_as_complex``) so they run on modern PyTorch.
"""
import numpy as np
import pytest
import torch

import utils.utils_sisr as sisr
import utils.utils_image as util


def _img(h=16, w=16, seed=0):
    rng = np.random.RandomState(seed)
    return rng.rand(h, w, 3).astype(np.float32)


# ------------------------------------------------------------------
# Gaussian kernels
# ------------------------------------------------------------------
def test_anisotropic_Gaussian_isotropic_is_symmetric_and_normalized():
    k = sisr.anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6)
    assert k.shape == (15, 15)
    assert abs(k.sum() - 1.0) < 1e-6
    assert np.all(k >= 0)
    assert np.allclose(k, k.T)


def test_gm_blur_kernel_normalized_and_positive():
    cov = [[4.0, 0.0], [0.0, 4.0]]
    k = sisr.gm_blur_kernel(mean=[0, 0], cov=cov, size=11)
    assert abs(k.sum() - 1.0) < 1e-6
    assert np.all(k >= 0)


# ------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------
def test_get_pca_matrix_shape_and_orthonormal_rows():
    x = np.random.randn(225, 100)
    p = sisr.get_pca_matrix(x, dim_pca=10)
    assert p.shape == (10, 225)
    # rows are orthonormal eigenvectors -> p @ p.T ~ I
    assert np.allclose(p @ p.T, np.eye(10), atol=1e-6)


# ------------------------------------------------------------------
# modcrop
# ------------------------------------------------------------------
def test_modcrop_np_multiples():
    img = np.random.rand(21, 19, 3)
    out = sisr.modcrop_np(img, sf=4)
    assert out.shape[0] % 4 == 0
    assert out.shape[1] % 4 == 0


# ------------------------------------------------------------------
# degradation (numpy)
# ------------------------------------------------------------------
def test_bicubic_degradation_downsamples():
    x = _img(16, 16)
    out = sisr.bicubic_degradation(x, sf=2)
    assert out.shape[:2] == (8, 8)


def test_srmd_degradation_shape():
    x = _img(16, 16)
    k = sisr.anisotropic_Gaussian(ksize=15, theta=np.pi, l1=2, l2=2)
    out = sisr.srmd_degradation(x, k, sf=2)
    assert out.shape[:2] == (8, 8)


def test_classical_degradation_shape():
    x = _img(16, 16)
    k = np.ones((3, 3)) / 9.0
    out = sisr.classical_degradation(x, k, sf=2)
    assert out.shape[:2] == (8, 8)


# ------------------------------------------------------------------
# pixel shift
# ------------------------------------------------------------------
def test_shift_pixel_zero_is_identity():
    x = _img(8, 8)
    out = sisr.shift_pixel(x, sf=2, upper_left=True)
    assert np.allclose(out, x)


# ------------------------------------------------------------------
# complex helpers (torch)
# ------------------------------------------------------------------
def test_r2c_stacks_zero_imag():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = sisr.r2c(t)
    assert out.shape == (2, 2, 2)
    assert torch.allclose(out[..., 0], t)
    assert torch.allclose(out[..., 1], torch.zeros_like(t))


def test_c2c_roundtrip():
    a = np.array([[1 + 2j, 3 - 1j], [0 + 1j, 4 + 0j]], dtype=np.complex128)
    out = sisr.c2c(a)
    assert out.shape == (2, 2, 2)
    assert np.allclose(out[..., 0], a.real)
    assert np.allclose(out[..., 1], a.imag)


def test_cabs_is_magnitude():
    t = torch.tensor([[[3.0, 4.0]]])  # real=3, imag=4
    assert sisr.cabs(t).item() == pytest.approx(5.0)


def test_cmul_matches_complex_mul():
    a = torch.tensor([[[1.0, 2.0]]])   # 1+2j
    b = torch.tensor([[[3.0, -1.0]]])  # 3-1j
    out = sisr.cmul(a, b)
    expected = (1 + 2j) * (3 - 1j)    # 5+5j
    assert out[..., 0].item() == pytest.approx(expected.real)
    assert out[..., 1].item() == pytest.approx(expected.imag)


def test_cconj_negates_imag():
    t = torch.tensor([[[1.0, 2.0]]])
    out = sisr.cconj(t)
    assert out[..., 0].item() == 1.0
    assert out[..., 1].item() == -2.0


# ------------------------------------------------------------------
# OTF / padding
# ------------------------------------------------------------------
def test_zero_pad_corner():
    img = np.ones((2, 2))
    out = sisr.zero_pad(img, (4, 4), position='corner')
    assert out.shape == (4, 4)
    assert np.array_equal(out[:2, :2], np.ones((2, 2)))
    assert out[2:, :].sum() == 0 and out[:, 2:].sum() == 0


def test_psf2otf_all_ones():
    psf = np.ones((3, 3))
    otf = sisr.psf2otf(psf)
    assert otf.shape == (3, 3)
    # fft2 of all-ones is 9 at DC, 0 elsewhere; imaginary part ~ 0
    assert np.allclose(otf, otf.real, atol=1e-9)
    assert otf[0, 0] == pytest.approx(9.0, abs=1e-6)
    assert np.allclose(otf[1:, :], 0, atol=1e-9)
    assert np.allclose(otf[:, 1:], 0, atol=1e-9)


# ------------------------------------------------------------------
# up/down sampling (numpy)
# ------------------------------------------------------------------
def test_upsample_downsample_np_roundtrip():
    x = _img(8, 8)
    up = sisr.upsample_np(x, sf=2)
    assert up.shape[:2] == (16, 16)
    down = sisr.downsample_np(up, sf=2)
    assert np.array_equal(down, x)


# ------------------------------------------------------------------
# G / Gt (numpy)
# ------------------------------------------------------------------
def test_G_np_and_Gt_np_shapes():
    x = _img(16, 16)
    k = np.ones((3, 3)) / 9.0
    low = sisr.G_np(x, k, sf=2)
    assert low.shape[:2] == (8, 8)
    rec = sisr.Gt_np(low, k, sf=2)
    assert rec.shape[:2] == (16, 16)


# ------------------------------------------------------------------
# migrated torch.fft wrappers
# ------------------------------------------------------------------
def test_rfft_shape_and_irfft_roundtrip():
    x = torch.randn(1, 3, 16, 16)
    X = sisr.rfft(x)
    assert X.shape == (1, 3, 16, 16, 2)
    y = sisr.irfft(X)
    assert y.shape == (1, 3, 16, 16)
    assert torch.allclose(y, x, atol=1e-5)


def test_fft_ifft_roundtrip():
    a = torch.randn(1, 1, 16, 16, 2)
    b = sisr.fft(a)
    assert b.shape == (1, 1, 16, 16, 2)
    c = sisr.ifft(b)
    assert torch.allclose(c, a, atol=1e-5)


def test_p2o_matches_numpy_reference():
    psf = torch.randn(1, 1, 5, 5)
    otf = sisr.p2o(psf, (16, 16))
    assert otf.shape == (1, 1, 16, 16, 2)
    ref = np.zeros((1, 1, 16, 16), dtype=np.complex64)
    ref[..., :5, :5] = psf.numpy()
    for ax in (2, 3):
        ref = np.roll(ref, -2, axis=ax)
    err = np.abs(torch.view_as_complex(otf).numpy() - np.fft.fft2(ref)).max()
    assert err < 1e-4


def test_INVLS_pytorch_runs():
    N, C, Up, sf = 1, 3, 32, 2
    shape = (N, C, Up, Up, 2)
    FB = torch.randn(*shape)
    FBC = torch.randn(*shape)
    F2B = torch.randn(*shape)
    FR = torch.randn(*shape)
    out = sisr.INVLS_pytorch(FB, FBC, F2B, FR, torch.tensor(0.05), sf)
    assert out.shape == (N, C, Up, Up)
    assert torch.isfinite(out).all()
