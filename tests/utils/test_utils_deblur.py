"""Unit tests for utils.utils_deblur.

Covers the numpy kernel/FFT/fspecial helpers and the migrated torch.fft
wrappers (``rfft``/``irfft``/``fft``/``ifft``/``p2o``) with numerical
assertions. The wrappers were migrated off the removed top-level
``torch.rfft``/``torch.irfft``/``torch.fft``/``torch.ifft`` onto the
``torch.fft`` namespace (native complex bridged via ``view_as_real`` /
``view_as_complex``) so they run on modern PyTorch.
"""
import numpy as np
import pytest
import torch

import utils.utils_deblur as deblur


# ------------------------------------------------------------------
# OTF / padding (mirrors the sisr copies)
# ------------------------------------------------------------------
def test_zero_pad_corner():
    img = np.ones((2, 2))
    out = deblur.zero_pad(img, (4, 4), position='corner')
    assert out.shape == (4, 4)
    assert np.array_equal(out[:2, :2], np.ones((2, 2)))
    assert out[2:, :].sum() == 0 and out[:, 2:].sum() == 0


def test_psf2otf_all_ones():
    psf = np.ones((3, 3))
    otf = deblur.psf2otf(psf)
    assert otf.shape == (3, 3)
    assert np.allclose(otf, otf.real, atol=1e-9)
    assert otf[0, 0] == pytest.approx(9.0, abs=1e-6)
    assert np.allclose(otf[1:, :], 0, atol=1e-9)
    assert np.allclose(otf[:, 1:], 0, atol=1e-9)


# ------------------------------------------------------------------
# fspecial family
# ------------------------------------------------------------------
def test_fspecial_average():
    h = deblur.fspecial_average(3)
    assert h.shape == (3, 3)
    assert np.allclose(h, 1.0 / 9.0)


def test_fspecial_gaussian_normalized_symmetric():
    h = deblur.fspecial_gaussian(5, 1.0)
    assert h.shape == (5, 5)
    assert abs(h.sum() - 1.0) < 1e-6
    assert np.all(h >= 0)
    assert np.allclose(h, h.T)


def test_fspecial_laplacian():
    h = deblur.fspecial_laplacian(0.5)
    expected = np.array([[1/3, 1/3, 1/3],
                         [1/3, -8/3, 1/3],
                         [1/3, 1/3, 1/3]])
    assert np.allclose(h, expected)


def test_fspecial_prewitt_sobel():
    assert np.array_equal(deblur.fspecial_prewitt(),
                          np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
    assert np.array_equal(deblur.fspecial_sobel(),
                          np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))


def test_fspecial_gauss_normalized():
    g = deblur.fspecial_gauss(5, 1.0)
    assert g.shape == (5, 5)
    assert abs(g.sum() - 1.0) < 1e-6


def test_fspecial_dispatch():
    assert np.allclose(deblur.fspecial('gaussian', 5, 1.0),
                       deblur.fspecial_gaussian(5, 1.0))


# ------------------------------------------------------------------
# opt_fft_size
# ------------------------------------------------------------------
def test_opt_fft_size_next_good():
    # 111 is not a 2/3/5/7*11/13 product; the next good size is 112 = 16*7
    assert int(deblur.opt_fft_size([111])[0]) == 112


# ------------------------------------------------------------------
# get_uperleft_denominator (numpy path)
# ------------------------------------------------------------------
def test_get_uperleft_denominator_shapes():
    img = np.random.rand(16, 16, 3).astype(np.float32)
    kernel = deblur.fspecial_gaussian(5, 1.0)
    upperleft, denom = deblur.get_uperleft_denominator(img, kernel)
    assert upperleft.shape == (16, 16, 3)
    assert denom.shape == (16, 16, 1)
    assert np.all(denom >= 0)


# ------------------------------------------------------------------
# complex helpers (duplicated from sisr)
# ------------------------------------------------------------------
def test_cabs_is_magnitude():
    t = torch.tensor([[[3.0, 4.0]]])
    assert deblur.cabs(t).item() == pytest.approx(5.0)


def test_cmul_matches_complex_mul():
    a = torch.tensor([[[1.0, 2.0]]])   # 1+2j
    b = torch.tensor([[[3.0, -1.0]]])  # 3-1j
    out = deblur.cmul(a, b)
    expected = (1 + 2j) * (3 - 1j)     # 5+5j
    assert out[..., 0].item() == pytest.approx(expected.real)
    assert out[..., 1].item() == pytest.approx(expected.imag)


# ------------------------------------------------------------------
# migrated torch.fft wrappers
# ------------------------------------------------------------------
def test_rfft_shape_and_irfft_roundtrip():
    x = torch.randn(1, 3, 16, 16)
    X = deblur.rfft(x)
    assert X.shape == (1, 3, 16, 16, 2)
    y = deblur.irfft(X)
    assert y.shape == (1, 3, 16, 16)
    assert torch.allclose(y, x, atol=1e-5)


def test_fft_ifft_roundtrip():
    a = torch.randn(1, 1, 16, 16, 2)
    b = deblur.fft(a)
    assert b.shape == (1, 1, 16, 16, 2)
    c = deblur.ifft(b)
    assert torch.allclose(c, a, atol=1e-5)


def test_p2o_matches_numpy_psf2otf():
    psf = torch.randn(1, 1, 5, 5)
    otf = deblur.p2o(psf, (16, 16))
    assert otf.shape == (1, 1, 16, 16, 2)
    ref = deblur.psf2otf(psf[0, 0].numpy(), (16, 16))
    err = np.abs(torch.view_as_complex(otf)[0, 0].numpy() - ref).max()
    assert err < 1e-4
