"""Unit tests for ``data/dataset_fdncnn.py`` (DatasetFDnCNN).

Covers:
  * ``_make_sample`` -- test mode exact noise (seed=0) + noise-level map M;
                        train mode patch pair (H, L with appended M).
  * ``__init__``     -- H_size / sigma / sigma_test defaults and overrides.
  * ``__len__``      -- via a real image directory.
  * ``__getitem__``  -- end-to-end, tensor dtype / shape / values, L_path == H_path.
"""
import numpy as np
import torch

import utils.utils_image as util
from data.dataset_fdncnn import DatasetFDnCNN


def _img_H(h=64, w=64, seed=0):
    """Deterministic synthetic RGB uint8 image of shape (h, w, 3) (seeded)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _opt(**kw):
    """Default opt dict for DatasetFDnCNN; H_size/sigma/sigma_test required (no .get())."""
    opt = {"phase": "test", "n_channels": 3, "H_size": 64, "sigma": [0, 75], "sigma_test": 25}
    opt.update(kw)
    return opt


# ------------------------------------------------------------
# _make_sample
# ------------------------------------------------------------
def test_fdncnn_make_sample_test_exact_noise_and_map():
    sigma_test = 30
    ds = DatasetFDnCNN(_opt(H_size=64, sigma=[0, 75], sigma_test=sigma_test))
    img_H = _img_H(64, 64, seed=7)

    img_H_out, img_L = ds._make_sample(img_H, 0)

    # H is converted to single precision, no noise added (3 channels).
    expected_H = util.uint2single(img_H)
    assert np.array_equal(img_H_out, expected_H)
    assert img_H_out.shape == (64, 64, 3)

    # L = H + AWGN(sigma_test/255) with np.random.seed(0) inside _make_sample,
    # plus a constant noise-level map M appended as the 4th channel.
    img_L_exp = np.copy(expected_H)
    np.random.seed(0)
    img_L_exp += np.random.normal(0, sigma_test / 255.0, img_L_exp.shape)  # in-place -> float32
    H, W, _ = img_L_exp.shape
    map_exp = (np.ones((H, W, 1), dtype=np.float32) * np.float32(sigma_test / 255.0))
    expected_L = np.concatenate([img_L_exp, map_exp], axis=2)

    assert img_L.shape == (64, 64, 4)
    assert np.array_equal(img_L, expected_L)

    # the map is constant sigma_test/255 over all spatial positions
    assert np.allclose(img_L[:, :, 3], sigma_test / 255.0)
    # first 3 channels equal H + noise
    assert np.array_equal(img_L[:, :, :3], img_L_exp)
    assert abs(np.std(img_L[:, :, 0] - expected_H[:, :, 0]) - sigma_test / 255.0) / (sigma_test / 255.0) < 0.02


def test_fdncnn_make_sample_train_is_patch_pair_with_map():
    patch = 48
    ds = DatasetFDnCNN(_opt(phase="train", H_size=patch, sigma=[0, 75], sigma_test=25))
    img_H = _img_H(80, 80, seed=3)

    img_H_out, img_L = ds._make_sample(img_H, 0)
    assert img_H_out.shape == (patch, patch, 3)
    assert img_L.shape == (patch, patch, 4)  # 3 channels + noise-level map
    assert img_H_out.dtype == np.float32
    assert img_L.dtype == np.float32
    # map is constant
    assert np.allclose(img_L[:, :, 3], img_L[0, 0, 3])


# ------------------------------------------------------------
# __init__
# ------------------------------------------------------------
def test_fdncnn_init_defaults():
    ds = DatasetFDnCNN(_opt())
    assert ds.patch_size == 64
    assert ds.sigma == [0, 75]
    assert ds.sigma_min == 0
    assert ds.sigma_max == 75
    assert ds.sigma_test == 25


def test_fdncnn_init_explicit():
    ds = DatasetFDnCNN(_opt(H_size=48, sigma=[10, 50], sigma_test=40))
    assert ds.patch_size == 48
    assert ds.sigma == [10, 50]
    assert ds.sigma_min == 10
    assert ds.sigma_max == 50
    assert ds.sigma_test == 40


# ------------------------------------------------------------
# __len__
# ------------------------------------------------------------
def test_fdncnn_len(make_image_dir):
    d = make_image_dir(n=3)
    ds = DatasetFDnCNN(_opt(dataroot_H=str(d)))
    assert len(ds) == 3


# ------------------------------------------------------------
# __getitem__
# ------------------------------------------------------------
def test_fdncnn_getitem(make_image_dir):
    d = make_image_dir(n=1, h=64, w=64)
    ds = DatasetFDnCNN(_opt(dataroot_H=str(d), sigma_test=30))
    img_H = ds._load_img_H(0)
    exp_H, exp_L = ds._make_sample(img_H, 0)
    out = ds[0]

    assert set(out.keys()) == {"L", "H", "L_path", "H_path"}
    assert out["H"].dtype == torch.float32
    assert out["L"].dtype == torch.float32
    assert out["L"].shape == torch.Size([4, 64, 64])
    assert torch.equal(out["H"], util.single2tensor3(exp_H))
    assert torch.equal(out["L"], util.single2tensor3(exp_L))
    assert out["L_path"] == out["H_path"]
