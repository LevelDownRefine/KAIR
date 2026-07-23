"""Unit tests for ``data/dataset_sr.py`` (DatasetSR).

Covers:
  * ``_make_sample`` (test mode) -- exact bicubic-downsample L of a modcropped H.
  * ``__init__``       -- scale / patch / L_size defaults and the paired L/H
                          length assertion.
  * ``__len__``        -- via a real image directory.
  * ``__getitem__``    -- end-to-end (synthetic L and paired L), tensor dtype /
                          shape / values, and correct L_path handling.
"""
import numpy as np
import pytest
import torch

import utils.utils_image as util
from data.dataset_sr import DatasetSR


def _img_H(h=64, w=64, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _opt(**kw):
    # DatasetSR.__init__ reads opt['scale'] with no .get(), so it is required
    # (mirrors the real dataloader config).
    opt = {"phase": "test", "n_channels": 3, "H_size": 96, "scale": 4}
    opt.update(kw)
    return opt


# ------------------------------------------------------------
# _make_sample (test mode): synthesized bicubic L
# ------------------------------------------------------------
def test_sr_make_sample_test_synthesized_l_exact():
    sf = 2
    ds = DatasetSR(_opt(scale=sf, paths_H=["dummy.png"]))
    img_H = _img_H(64, 64, seed=11)

    img_H_out, img_L, L_path = ds._make_sample(img_H, 0)

    assert L_path is None  # synthesized, not loaded from disk

    expected_H = util.modcrop(util.uint2single(img_H), sf)
    assert np.array_equal(img_H_out, expected_H)

    expected_L = util.imresize_np(expected_H, 1 / sf, True)
    assert np.array_equal(img_L, expected_L)
    assert img_L.shape[:2] == (expected_H.shape[0] // sf, expected_H.shape[1] // sf)


# ------------------------------------------------------------
# __init__
# ------------------------------------------------------------
def test_sr_init_defaults(make_image_dir):
    d = make_image_dir(n=2)
    ds = DatasetSR(_opt(dataroot_H=str(d)))
    assert ds.sf == 4
    assert ds.patch_size == 96
    assert ds.L_size == 96 // 4
    assert len(ds.paths_H) == 2


def test_sr_init_paired_length_mismatch_raises(make_image_dir):
    h = make_image_dir(n=2, name="h")
    l = make_image_dir(n=3, name="l")
    with pytest.raises(AssertionError):
        DatasetSR(_opt(dataroot_H=str(h), dataroot_L=str(l)))


# ------------------------------------------------------------
# __len__
# ------------------------------------------------------------
def test_sr_len(make_image_dir):
    d = make_image_dir(n=4)
    ds = DatasetSR(_opt(dataroot_H=str(d)))
    assert len(ds) == 4


# ------------------------------------------------------------
# __getitem__
# ------------------------------------------------------------
def test_sr_getitem_synthetic(make_image_dir):
    d = make_image_dir(n=1, h=80, w=80)
    ds = DatasetSR(_opt(scale=2, dataroot_H=str(d)))
    img_H = ds._load_img_H(0)
    exp_H, exp_L, _ = ds._make_sample(img_H, 0)
    out = ds[0]

    assert set(out.keys()) == {"L", "H", "L_path", "H_path"}
    assert out["H"].dtype == torch.float32 and out["L"].dtype == torch.float32
    assert torch.equal(out["H"], util.single2tensor3(exp_H))
    assert torch.equal(out["L"], util.single2tensor3(exp_L))
    assert out["L_path"] == out["H_path"]  # synthesized


def test_sr_getitem_paired(make_image_dir):
    h = make_image_dir(n=2, name="h", h=80, w=80)
    l = make_image_dir(n=2, name="l", h=80, w=80)
    ds = DatasetSR(_opt(scale=2, dataroot_H=str(h), dataroot_L=str(l)))
    img_H = ds._load_img_H(0)
    exp_H, exp_L, L_path = ds._make_sample(img_H, 0)
    out = ds[0]

    assert out["L_path"] != out["H_path"]
    assert out["L_path"] == L_path
    assert torch.equal(out["H"], util.single2tensor3(exp_H))
    assert torch.equal(out["L"], util.single2tensor3(exp_L))
