"""Unit tests for ``data/dataset_dnpatch.py`` (DatasetDnPatch).

Covers ``__init__`` / ``__len__`` / ``_make_sample`` / ``__getitem__``.

NUMERICAL VERIFICATION:
  * test mode: ``_make_sample`` adds exactly seeded AWGN (``sigma_test/255``,
    ``np.random.seed(0)``); recomputed independently and asserted equal.
  * train mode: patch is augmented (``random.randint``) then corrupted with
    ``torch.randn(sigma/255)``; reseeding ``random`` + ``torch`` identically
    reproduces the exact (H, L) pair.
"""
import random
import numpy as np
import torch

import utils.utils_image as util
from data.dataset_dnpatch import DatasetDnPatch


def _img_H(h=64, w=64, seed=0):
    """Deterministic synthetic RGB uint8 image of shape (h, w, 3) (seeded)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _opt(**kw):
    """Default opt dict for DatasetDnPatch; H_size/sigma/sigma_test/... are required (no .get())."""
    opt = {"phase": "test", "n_channels": 3, "H_size": 64, "sigma": 25, "sigma_test": 25,
           "num_patches_per_image": 1, "num_sampled": 1}
    opt.update(kw)
    return opt


# ------------------------------------------------------------
# __init__ / __len__
# ------------------------------------------------------------
def test_dnpatch_init_and_len(make_image_dir):
    d = make_image_dir(n=3, h=64, w=64)
    ds = DatasetDnPatch(_opt(dataroot_H=str(d), num_sampled=2, num_patches_per_image=2))
    assert ds.patch_size == 64
    assert ds.sigma == 25
    assert ds.sigma_test == 25
    assert ds.num_sampled == 2
    assert len(ds) == 2 * 2  # num_sampled * num_patches_per_image


# ------------------------------------------------------------
# _make_sample
# ------------------------------------------------------------
def test_dnpatch_make_sample_test_exact_noise(make_image_dir):
    sigma_test = 30
    d = make_image_dir(n=1, h=64, w=64)
    ds = DatasetDnPatch(_opt(dataroot_H=str(d), num_sampled=1, num_patches_per_image=1,
                             phase='test', H_size=64, sigma=25, sigma_test=sigma_test))
    known_H = _img_H(64, 64, seed=7)

    img_H_out, img_L = ds._make_sample(known_H, 0)

    exp_H = util.uint2single(known_H)
    assert np.array_equal(img_H_out, exp_H)

    exp_L = np.copy(exp_H)
    np.random.seed(0)
    exp_L += np.random.normal(0, sigma_test / 255.0, exp_L.shape)
    assert np.array_equal(img_L, exp_L)


def test_dnpatch_make_sample_train_is_noisy_patch(make_image_dir):
    d = make_image_dir(n=1, h=64, w=64)
    ds = DatasetDnPatch(_opt(dataroot_H=str(d), num_sampled=1, num_patches_per_image=1,
                             phase='train', H_size=48, sigma=25, sigma_test=25))
    known_H = _img_H(48, 48, seed=3)

    random.seed(11)
    torch.manual_seed(11)
    img_H_out, img_L = ds._make_sample(known_H, 0)

    random.seed(11)
    torch.manual_seed(11)
    mode = random.randint(0, 7)
    aug = util.augment_img(known_H, mode=mode)
    exp_H = util.uint2single(aug)
    exp_L = np.copy(exp_H) + torch.randn(exp_H.shape).mul_(25 / 255.0).numpy()

    assert np.array_equal(img_H_out, exp_H)
    assert np.array_equal(img_L, exp_L)
    assert img_L.shape == (48, 48, 3)
    assert img_L.dtype == np.float32


# ------------------------------------------------------------
# __getitem__ (end-to-end, test mode)
# ------------------------------------------------------------
def test_dnpatch_getitem(make_image_dir):
    d = make_image_dir(n=3, h=64, w=64)
    ds = DatasetDnPatch(_opt(dataroot_H=str(d), num_sampled=2, num_patches_per_image=1, phase='test', H_size=64))
    out = ds[0]

    assert set(out.keys()) == {"L", "H", "L_path", "H_path"}
    assert out["H"].dtype == torch.float32 and out["L"].dtype == torch.float32
    assert out["L"].shape == (3, 64, 64)
    assert out["H"].shape == (3, 64, 64)
    assert out["L_path"] == out["H_path"]
