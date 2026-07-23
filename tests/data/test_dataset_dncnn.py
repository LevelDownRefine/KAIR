"""Unit tests for ``data/dataset_dncnn.py`` (DatasetDnCNN).

Covers:
  * ``_make_noisy`` -- seeded RNG reproduces exact noise; unseeded call adds
                       AWGN (std == sigma/255) and differs from the input.
  * ``_make_sample`` -- test mode exact noise (seed=0); train mode patch pair.
  * ``__init__``     -- H_size / sigma / sigma_test defaults and overrides.
  * ``__len__``      -- via a real image directory.
  * ``__getitem__``  -- end-to-end, tensor dtype / shape / values, L_path == H_path.
"""
import numpy as np
import torch

import utils.utils_image as util
from data.dataset_dncnn import DatasetDnCNN


def _img_H(h=64, w=64, seed=0):
    """Deterministic synthetic RGB uint8 image of shape (h, w, 3) (seeded)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _single(h=64, w=64, seed=0):
    """Deterministic float32 single-precision image in [0,1]."""
    return util.uint2single(_img_H(h, w, seed)).astype(np.float32)


def _opt(**kw):
    """Default opt dict for DatasetDnCNN; sigma/sigma_test required (no .get())."""
    opt = {"phase": "test", "n_channels": 3, "H_size": 64, "sigma": 25, "sigma_test": 25}
    opt.update(kw)
    return opt


# ------------------------------------------------------------
# _make_noisy
# ------------------------------------------------------------
def test_dncnn_make_noisy_seed_reproducible():
    ds = DatasetDnCNN(_opt())
    img = _single(32, 32, seed=1)
    a = ds._make_noisy(img, 25, seed=5)
    b = ds._make_noisy(img, 25, seed=5)
    assert np.array_equal(a, b)  # same seed -> identical

    # matches the exact RNG draw the method performs internally
    np.random.seed(5)
    expected = img + np.random.normal(0, 25 / 255.0, img.shape)
    assert np.array_equal(a, expected)
    # numpy promotes float32 + float64 -> float64; the dataset converts to a
    # float32 *tensor* downstream, so the in-memory dtype is floating either way.
    assert np.issubdtype(a.dtype, np.floating)


def test_dncnn_make_noisy_unseeded_adds_noise():
    ds = DatasetDnCNN(_opt())
    img = _single(32, 32, seed=2)
    out = ds._make_noisy(img, 50)  # no seed -> global RNG, fresh noise
    assert out.shape == img.shape
    assert np.issubdtype(out.dtype, np.floating)
    # AWGN std must equal sigma/255; with ~3k samples the sample-std has
    # sub-percent error, so allow 2%.
    expected_std = 50 / 255.0
    assert abs(np.std(out - img) - expected_std) / expected_std < 0.02
    assert abs(np.mean(out - img)) < 0.02  # ~zero mean


# ------------------------------------------------------------
# _make_sample
# ------------------------------------------------------------
def test_dncnn_make_sample_test_exact_noise():
    sigma_test = 30
    ds = DatasetDnCNN(_opt(H_size=64, sigma=25, sigma_test=sigma_test))
    img_H = _img_H(64, 64, seed=7)

    img_H_out, img_L = ds._make_sample(img_H, 0)

    # H is converted to single precision, no noise added.
    expected_H = util.uint2single(img_H)
    assert np.array_equal(img_H_out, expected_H)

    # L = H + AWGN(sigma_test/255) with np.random.seed(0) inside _make_noisy.
    np.random.seed(0)
    expected_noise = np.random.normal(0, sigma_test / 255.0, img_H.shape)
    expected_L = util.uint2single(img_H) + expected_noise
    assert np.array_equal(img_L, expected_L)
    expected_std = sigma_test / 255.0
    assert abs(np.std(img_L - expected_H) - expected_std) / expected_std < 0.02


def test_dncnn_make_sample_train_is_patch_pair():
    patch = 48
    ds = DatasetDnCNN(_opt(phase="train", H_size=patch, sigma=25, sigma_test=25))
    img_H = _img_H(80, 80, seed=3)

    img_H_out, img_L = ds._make_sample(img_H, 0)
    assert img_H_out.shape == (patch, patch, 3)
    assert img_L.shape == (patch, patch, 3)
    assert img_H_out.dtype == np.float32


# ------------------------------------------------------------
# __init__
# ------------------------------------------------------------
def test_dncnn_init_defaults():
    ds = DatasetDnCNN(_opt())
    assert ds.patch_size == 64
    assert ds.sigma == 25
    assert ds.sigma_test == 25


def test_dncnn_init_explicit():
    ds = DatasetDnCNN(_opt(H_size=48, sigma=10, sigma_test=40))
    assert ds.patch_size == 48
    assert ds.sigma == 10
    assert ds.sigma_test == 40


# ------------------------------------------------------------
# __len__
# ------------------------------------------------------------
def test_dncnn_len(make_image_dir):
    d = make_image_dir(n=3)
    ds = DatasetDnCNN(_opt(dataroot_H=str(d)))
    assert len(ds) == 3


# ------------------------------------------------------------
# __getitem__
# ------------------------------------------------------------
def test_dncnn_getitem(make_image_dir):
    d = make_image_dir(n=1, h=64, w=64)
    ds = DatasetDnCNN(_opt(dataroot_H=str(d), sigma_test=30))
    img_H = ds._load_img_H(0)
    exp_H, exp_L = ds._make_sample(img_H, 0)
    out = ds[0]

    assert set(out.keys()) == {"L", "H", "H_path", "L_path"}
    assert out["H"].dtype == torch.float32
    assert torch.equal(out["H"], util.single2tensor3(exp_H))
    assert torch.equal(out["L"], util.single2tensor3(exp_L))
    assert out["L_path"] == out["H_path"]
