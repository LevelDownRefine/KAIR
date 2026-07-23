"""Unit tests for ``data/dataset_plain.py`` (DatasetPlain).

Loads both L and H from disk; no synthesis. Covers ``__init__`` / ``__len__`` /
``_make_sample`` (exact: test returns H unchanged and L equal to the on-disk L;
train returns a cropped+augmented paired patch) / ``__getitem__``.
"""
import numpy as np
import torch

import utils.utils_image as util
from data.dataset_plain import DatasetPlain


def _img(h=64, w=64, seed=0):
    """Deterministic synthetic RGB uint8 image of shape (h, w, 3) (seeded)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _opt(dataroot_H, dataroot_L, phase="test", **kw):
    """Default opt dict for DatasetPlain; both dataroot_H and dataroot_L are required."""
    opt = {
        "n_channels": 3,
        "H_size": 64,
        "dataroot_H": dataroot_H,
        "dataroot_L": dataroot_L,
        "phase": phase,
    }
    opt.update(kw)
    return opt


# ------------------------------------------------------------
# __init__
# ------------------------------------------------------------
def test_plain_init_defaults(make_image_dir):
    h = make_image_dir(n=2, name="h")
    l = make_image_dir(n=2, name="l")
    ds = DatasetPlain(_opt(str(h), str(l)))
    assert ds.patch_size == 64
    assert len(ds.paths_H) == 2
    assert len(ds.paths_L) == 2


def test_plain_init_paired_length_mismatch_raises(make_image_dir):
    import pytest
    h = make_image_dir(n=2, name="h")
    l = make_image_dir(n=3, name="l")
    with pytest.raises(AssertionError):
        DatasetPlain(_opt(str(h), str(l)))


# ------------------------------------------------------------
# __len__
# ------------------------------------------------------------
def test_plain_len(make_image_dir):
    h = make_image_dir(n=4, name="h")
    l = make_image_dir(n=4, name="l")
    ds = DatasetPlain(_opt(str(h), str(l)))
    assert len(ds) == 4


# ------------------------------------------------------------
# _make_sample (test mode): exact recompute
# ------------------------------------------------------------
def test_plain_make_sample_test_exact(make_image_dir):
    h = make_image_dir(n=2, name="h", h=64, w=64)
    l = make_image_dir(n=2, name="l", h=64, w=64)
    ds = DatasetPlain(_opt(str(h), str(l), phase="test"))
    known_H = _img(64, 64, seed=9)

    out_H, out_L, L_path = ds._make_sample(known_H, 0)

    # test mode returns H unchanged and L equal to the on-disk L image
    assert np.array_equal(out_H, known_H)
    expected_L = util.imread_uint(ds.paths_L[0], ds.n_channels)
    assert np.array_equal(out_L, expected_L)
    assert L_path == ds.paths_L[0]


# ------------------------------------------------------------
# _make_sample (train mode): paired patch crop
# ------------------------------------------------------------
def test_plain_make_sample_train_patch_shape(make_image_dir):
    h = make_image_dir(n=2, name="h", h=128, w=128)
    l = make_image_dir(n=2, name="l", h=128, w=128)
    ds = DatasetPlain(_opt(str(h), str(l), phase="train", H_size=32))
    known_H = _img(128, 128, seed=9)
    out_H, out_L, _ = ds._make_sample(known_H, 0)
    assert out_H.shape == (32, 32, 3)
    assert out_L.shape == (32, 32, 3)
    assert out_H.dtype == np.uint8 and out_L.dtype == np.uint8


# ------------------------------------------------------------
# __getitem__
# ------------------------------------------------------------
def test_plain_getitem_test(make_image_dir):
    h = make_image_dir(n=1, name="h", h=64, w=64)
    l = make_image_dir(n=1, name="l", h=64, w=64)
    ds = DatasetPlain(_opt(str(h), str(l), phase="test"))
    img_H = ds._load_img_H(0)
    exp_H, exp_L, L_path = ds._make_sample(img_H, 0)
    out = ds[0]
    assert set(out.keys()) == {"L", "H", "L_path", "H_path"}
    assert out["L"].dtype == torch.float32 and out["H"].dtype == torch.float32
    assert torch.equal(out["H"], util.uint2tensor3(exp_H))
    assert torch.equal(out["L"], util.uint2tensor3(exp_L))
    assert out["L_path"] == L_path
