"""Unit tests for ``data/dataset_ffdnet.py`` (DatasetFFDNet).

Covers:
  * ``_make_sample`` -- test mode exact noise (seed=0) + returned noise level;
                        train mode patch pair (H, L) plus per-sample noise level.
  * ``__init__``     -- H_size / sigma / sigma_test defaults and overrides.
  * ``__len__``      -- via a real image directory.
  * ``__getitem__``  -- end-to-end, tensor dtype / shape / values, 'C' noise level.
"""
import numpy as np
import torch

import utils.utils_image as util
from data.dataset_ffdnet import DatasetFFDNet


def _img_H(h=64, w=64, seed=0):
    """Deterministic synthetic RGB uint8 image of shape (h, w, 3) (seeded)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _opt(**kw):
    """Default opt dict for DatasetFFDNet; H_size/sigma/sigma_test required (no .get())."""
    opt = {"phase": "test", "n_channels": 3, "H_size": 64, "sigma": [0, 75], "sigma_test": 25}
    opt.update(kw)
    return opt


# ------------------------------------------------------------
# _make_sample
# ------------------------------------------------------------
def test_ffdnet_make_sample_test_exact_noise_and_level():
    sigma_test = 30
    ds = DatasetFFDNet(_opt(H_size=64, sigma=[0, 75], sigma_test=sigma_test))
    img_H = _img_H(64, 64, seed=7)

    img_H_out, img_L, noise_level = ds._make_sample(img_H, 0)

    # H is converted to single precision, no noise added.
    expected_H = util.uint2single(img_H)
    assert np.array_equal(img_H_out, expected_H)

    # L = H + AWGN(sigma_test/255) with np.random.seed(0); noise_level = sigma_test/255.
    img_L_exp = np.copy(expected_H)
    np.random.seed(0)
    img_L_exp += np.random.normal(0, sigma_test / 255.0, img_L_exp.shape)  # in-place -> float32
    assert np.array_equal(img_L, img_L_exp)
    assert img_L.shape == (64, 64, 3)

    assert abs(noise_level - sigma_test / 255.0) < 1e-9
    expected_std = sigma_test / 255.0
    assert abs(np.std(img_L[:, :, 0] - expected_H[:, :, 0]) - expected_std) / expected_std < 0.02


def test_ffdnet_make_sample_train_is_patch_pair_with_level():
    patch = 48
    ds = DatasetFFDNet(_opt(phase="train", H_size=patch, sigma=[0, 75], sigma_test=25))
    img_H = _img_H(80, 80, seed=3)

    img_H_out, img_L, noise_level = ds._make_sample(img_H, 0)
    assert img_H_out.shape == (patch, patch, 3)
    assert img_L.shape == (patch, patch, 3)
    assert img_H_out.dtype == np.float32
    assert img_L.dtype == np.float32
    assert noise_level >= 0.0


# ------------------------------------------------------------
# __init__
# ------------------------------------------------------------
def test_ffdnet_init_defaults():
    ds = DatasetFFDNet(_opt())
    assert ds.patch_size == 64
    assert ds.sigma == [0, 75]
    assert ds.sigma_min == 0
    assert ds.sigma_max == 75
    assert ds.sigma_test == 25


def test_ffdnet_init_explicit():
    ds = DatasetFFDNet(_opt(H_size=48, sigma=[10, 50], sigma_test=40))
    assert ds.patch_size == 48
    assert ds.sigma == [10, 50]
    assert ds.sigma_min == 10
    assert ds.sigma_max == 50
    assert ds.sigma_test == 40


# ------------------------------------------------------------
# __len__
# ------------------------------------------------------------
def test_ffdnet_len(make_image_dir):
    d = make_image_dir(n=3)
    ds = DatasetFFDNet(_opt(dataroot_H=str(d)))
    assert len(ds) == 3


# ------------------------------------------------------------
# __getitem__
# ------------------------------------------------------------
def test_ffdnet_getitem(make_image_dir):
    d = make_image_dir(n=1, h=64, w=64)
    ds = DatasetFFDNet(_opt(dataroot_H=str(d), sigma_test=30))
    img_H = ds._load_img_H(0)
    exp_H, exp_L, noise_level = ds._make_sample(img_H, 0)
    out = ds[0]

    assert set(out.keys()) == {"L", "H", "C", "L_path", "H_path"}
    assert out["H"].dtype == torch.float32
    assert out["L"].dtype == torch.float32
    assert out["L"].shape == torch.Size([3, 64, 64])
    assert torch.equal(out["H"], util.single2tensor3(exp_H))
    assert torch.equal(out["L"], util.single2tensor3(exp_L))
    # 'C' carries the per-sample noise level as a [1,1,1] tensor
    assert out["C"].shape == torch.Size([1, 1, 1])
    assert out["C"].dtype == torch.float32
    assert abs(float(out["C"]) - noise_level) < 1e-9
    assert out["L_path"] == out["H_path"]
