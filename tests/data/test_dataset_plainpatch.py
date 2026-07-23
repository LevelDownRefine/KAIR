"""Unit tests for ``data/dataset_plainpatch.py`` (DatasetPlainPatch).

Pre-extracts L/H patch pairs into ``H_data``/``L_data`` buffers at ``__init__``;
``__getitem__`` augments a buffered pair in train mode and loads the full
image pair in test mode. Covers ``__init__`` (buffer shape via ``len`` and
``H_data``/``L_data``), ``_make_sample`` (exact: test is identity; train
applies the same augment mode to H and L), and ``__getitem__`` end-to-end.

Guards the historical ``self.path_size`` -> ``self.patch_size`` typo that
raised ``AttributeError`` during buffer allocation and ``get_patches``.
"""
import random
import numpy as np
import torch

import utils.utils_image as util
from data.dataset_plainpatch import DatasetPlainPatch


def _make_roots(tmp_path, write_rgb_png, size=100, count=3):
    roots = {}
    for key in ("dataroot_H", "dataroot_L"):
        d = tmp_path / key
        d.mkdir(parents=True)
        for i in range(count):
            write_rgb_png(d / f"img_{i:03d}.png", size, size, seed=i * 7 + (0 if key == "dataroot_H" else 3))
        roots[key] = str(d)
    return roots


def _opt(roots, phase="train", num_sampled=2, num_patches=3, H_size=64):
    """Default opt dict for DatasetPlainPatch; both dataroots are required."""
    return {
        "n_channels": 3,
        "H_size": H_size,
        "num_patches_per_image": num_patches,
        "num_sampled": num_sampled,
        "dataroot_H": roots["dataroot_H"],
        "dataroot_L": roots["dataroot_L"],
        "phase": phase,
    }


# ------------------------------------------------------------
# __init__ / buffers
# ------------------------------------------------------------
def test_plainpatch_builds_patch_buffers(tmp_path, write_rgb_png):
    roots = _make_roots(tmp_path, write_rgb_png)
    ds = DatasetPlainPatch(_opt(roots))
    expected = 2 * 3  # num_sampled * num_patches_per_image
    assert len(ds) == expected
    assert ds.H_data.shape == (expected, 64, 64, 3)
    assert ds.L_data.shape == (expected, 64, 64, 3)


# ------------------------------------------------------------
# _make_sample (test mode): identity
# ------------------------------------------------------------
def test_plainpatch_make_sample_test_identity(tmp_path, write_rgb_png):
    roots = _make_roots(tmp_path, write_rgb_png)
    ds = DatasetPlainPatch(_opt(roots, phase="test"))
    img_H = ds._load_img_H(0)
    img_L = ds._load_img_L(0)
    out_H, out_L = ds._make_sample(img_H, img_L, 0)
    assert np.array_equal(out_H, img_H)
    assert np.array_equal(out_L, img_L)


# ------------------------------------------------------------
# _make_sample (train mode): same augment mode on H and L
# ------------------------------------------------------------
def test_plainpatch_make_sample_train_augment_exact(tmp_path, write_rgb_png):
    roots = _make_roots(tmp_path, write_rgb_png)
    ds = DatasetPlainPatch(_opt(roots, phase="train"))
    img_H = ds._load_img_H(0)
    img_L = ds._load_img_L(0)

    # Predict the single random draw inside _make_sample without calling it.
    expected_mode = random.Random(42).randint(0, 7)
    random.seed(42)
    out_H, out_L = ds._make_sample(img_H, img_L, 0)

    exp_H = util.augment_img(img_H, expected_mode)
    exp_L = util.augment_img(img_L, expected_mode)
    assert np.array_equal(out_H, exp_H)
    assert np.array_equal(out_L, exp_L)


# ------------------------------------------------------------
# __getitem__
# ------------------------------------------------------------
def test_plainpatch_getitem_train_returns_paired(tmp_path, write_rgb_png):
    roots = _make_roots(tmp_path, write_rgb_png)
    ds = DatasetPlainPatch(_opt(roots, phase="train"))
    sample = ds[0]
    assert set(sample.keys()) >= {"L", "H"}
    assert isinstance(sample["L"], torch.Tensor)
    assert isinstance(sample["H"], torch.Tensor)
    assert sample["L"].shape == (3, 64, 64)
    assert sample["H"].shape == (3, 64, 64)


def test_plainpatch_getitem_test_returns_full(tmp_path, write_rgb_png):
    roots = _make_roots(tmp_path, write_rgb_png, size=64)
    ds = DatasetPlainPatch(_opt(roots, phase="test"))
    sample = ds[0]
    assert set(sample.keys()) >= {"L", "H"}
    assert sample["L"].shape == (3, 64, 64)
    assert sample["H"].shape == (3, 64, 64)
