"""Unit tests for ``data/dataset_dpsr.py`` (DatasetDPSR).

Covers ``__init__`` / ``__len__`` / ``_make_sample`` / ``__getitem__``.

NUMERICAL VERIFICATION: in test mode ``_make_sample`` computes
``L = imresize_np(modcrop(H, sf), 1/sf)`` then adds ``torch.randn(sigma_test)``
AWGN and concatenates the constant noise-level map M. Reseeding ``torch`` and
recomputing with the same ``util`` primitives reproduces the exact L (the
bicubic and modcrop use no RNG, so only ``torch.randn`` must be reseeded).
"""
import random
import numpy as np
import torch

import utils.utils_image as util
from data.dataset_dpsr import DatasetDPSR


def _img_H(h=64, w=64, seed=0):
    """Deterministic synthetic RGB uint8 image of shape (h, w, 3) (seeded)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _opt(**kw):
    """Default opt dict for DatasetDPSR; scale/H_size/sigma/sigma_test are required (no .get())."""
    opt = {"phase": "test", "n_channels": 3, "scale": 4, "H_size": 96,
           "sigma": [0, 50], "sigma_test": 0}
    opt.update(kw)
    return opt


# ------------------------------------------------------------
# __init__
# ------------------------------------------------------------
def test_dpsr_init_defaults(make_image_dir):
    d = make_image_dir(n=2)
    ds = DatasetDPSR(_opt(dataroot_H=str(d)))
    assert ds.sf == 4
    assert ds.patch_size == 96
    assert ds.L_size == 96 // 4
    assert ds.sigma == [0, 50]
    assert ds.sigma_min == 0 and ds.sigma_max == 50
    assert ds.sigma_test == 0
    assert len(ds.paths_H) == 2


# ------------------------------------------------------------
# __len__
# ------------------------------------------------------------
def test_dpsr_len(make_image_dir):
    d = make_image_dir(n=5)
    ds = DatasetDPSR(_opt(dataroot_H=str(d)))
    assert len(ds) == 5


# ------------------------------------------------------------
# _make_sample (test mode): exact bicubic L + noise + M
# ------------------------------------------------------------
def test_dpsr_make_sample_test_exact():
    sf = 2
    sigma_test = 10
    ds = DatasetDPSR(_opt(scale=sf, H_size=96, sigma=[0, 50], sigma_test=sigma_test, paths_H=['dummy.png']))
    known_H = _img_H(64, 64, seed=11)

    torch.manual_seed(0)
    img_H_out, img_L = ds._make_sample(known_H, 0)

    exp_H = util.modcrop(util.uint2single(known_H), sf)
    assert np.array_equal(img_H_out, exp_H)

    exp_L_bic = util.imresize_np(exp_H, 1 / sf, True)
    torch.manual_seed(0)
    exp_noise = torch.randn(exp_L_bic.shape).mul_(sigma_test).numpy()
    exp_M = np.full((exp_L_bic.shape[0], exp_L_bic.shape[1], 1), sigma_test, dtype=np.float32)
    exp_L = np.concatenate((exp_L_bic + exp_noise, exp_M), axis=2)

    assert np.array_equal(img_L, exp_L)
    assert img_L.shape[2] == 4  # 3 RGB channels + 1 noise-level map


# ------------------------------------------------------------
# _make_sample (train mode): augmented patch pair, M constant
# ------------------------------------------------------------
def test_dpsr_make_sample_train_is_patch_pair():
    sf = 2
    ds = DatasetDPSR(_opt(scale=sf, H_size=96, sigma=[0, 50], sigma_test=0,
                          paths_H=['dummy.png'], phase='train'))
    known_H = _img_H(96, 96, seed=9)
    img_H_out, img_L = ds._make_sample(known_H, 0)

    assert img_L.shape == (96 // sf, 96 // sf, 4)   # 3 RGB + 1 M
    assert img_H_out.shape == (96, 96, 3)
    assert img_L.dtype == np.float32
    # M (4th channel) is a constant noise-level map
    assert np.allclose(img_L[..., 3], img_L[0, 0, 3])


# ------------------------------------------------------------
# __getitem__ (end-to-end, test mode)
# ------------------------------------------------------------
def test_dpsr_getitem(make_image_dir):
    d = make_image_dir(n=1, h=128, w=128)
    ds = DatasetDPSR(_opt(scale=2, H_size=96, sigma=[0, 50], sigma_test=5, dataroot_H=str(d), phase='test'))
    out = ds[0]

    assert set(out.keys()) == {"L", "H", "L_path", "H_path"}
    assert out["H"].dtype == torch.float32 and out["L"].dtype == torch.float32
    assert out["H"].shape == (3, 128, 128)   # modcrop(128, 2) -> 128
    assert out["L"].shape == (4, 64, 64)     # bicubic(128/2) + M
    assert out["L_path"] == out["H_path"]
