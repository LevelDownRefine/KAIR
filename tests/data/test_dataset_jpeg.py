"""Unit tests for ``data/dataset_jpeg.py`` (DatasetJPEG).

Covers:
  * ``_make_sample`` (test mode) -- exact JPEG round-trip of a known H,
    recomputed with cv2.imencode/imdecode (color and grayscale).
  * ``__init__``       -- quality-factor / patch-size defaults.
  * ``__len__``        -- via a real image directory.
  * ``__getitem__``    -- end-to-end (regression for the test-mode grayscale
                         branch and the train patch shape).

JPEG is lossy, so the random ``train`` branch (crop + flip/rotate + compress)
is checked only for output shape/dtype; the deterministic ``test`` branch is
verified bit-exactly by re-deriving the compressed L with the same cv2
primitives (independent recomputation, not copied class code).
"""
import numpy as np
import torch

import cv2
import utils.utils_image as util
from data.dataset_jpeg import DatasetJPEG


def _img_H(h=64, w=64, seed=0):
    """Deterministic synthetic RGB uint8 image of shape (h, w, 3) (seeded)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _opt(dataroot_H, phase="test", is_color=False, **kw):
    """Default opt dict for DatasetJPEG; quality factors / patch size are required keys."""
    opt = {
        "n_channels": 3,
        "H_size": 64,
        "quality_factor": 40,
        "quality_factor_test": 40,
        "is_color": is_color,
        "dataroot_H": dataroot_H,
        "phase": phase,
    }
    opt.update(kw)
    return opt


# ------------------------------------------------------------
# _make_sample (test mode, color): exact JPEG round-trip
# ------------------------------------------------------------
def test_jpeg_make_sample_test_color_exact():
    ds = DatasetJPEG(_opt("dummy", phase="test", is_color=True, paths_H=["dummy.png"]))
    img_H = _img_H(64, 64, seed=3)
    qf = ds.quality_factor_test

    out_H, out_L = ds._make_sample(img_H, 0)

    # H is unchanged; L is the JPEG-compressed color form
    assert np.array_equal(out_H, img_H)

    expected_L = cv2.cvtColor(img_H, cv2.COLOR_RGB2BGR)
    _, enc = cv2.imencode('.jpg', expected_L, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
    expected_L = cv2.imdecode(enc, 1)
    expected_L = cv2.cvtColor(expected_L, cv2.COLOR_BGR2RGB)
    assert np.array_equal(out_L, expected_L)


# ------------------------------------------------------------
# _make_sample (test mode, grayscale): exact JPEG round-trip
# ------------------------------------------------------------
def test_jpeg_make_sample_test_gray_exact():
    ds = DatasetJPEG(_opt("dummy", phase="test", is_color=False, paths_H=["dummy.png"]))
    img_H = _img_H(64, 64, seed=3)
    qf = ds.quality_factor_test

    out_H, out_L = ds._make_sample(img_H, 0)

    expected_H = util.rgb2ycbcr(img_H)  # only Y channel (2D)
    assert np.array_equal(out_H, expected_H)

    _, enc = cv2.imencode('.jpg', expected_H, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
    expected_L = cv2.imdecode(enc, 0)
    assert np.array_equal(out_L, expected_L)


# ------------------------------------------------------------
# __init__
# ------------------------------------------------------------
def test_jpeg_init_defaults(make_image_dir):
    d = make_image_dir(n=2)
    ds = DatasetJPEG(_opt(str(d), phase="test", is_color=False))
    assert ds.patch_size == 64
    assert ds.quality_factor == 40
    assert ds.quality_factor_test == 40
    assert ds.is_color is False
    assert len(ds.paths_H) == 2


# ------------------------------------------------------------
# __len__
# ------------------------------------------------------------
def test_jpeg_len(make_image_dir):
    d = make_image_dir(n=5)
    ds = DatasetJPEG(_opt(str(d), phase="test"))
    assert len(ds) == 5


# ------------------------------------------------------------
# __getitem__ (regression guards)
# ------------------------------------------------------------
def test_jpeg_getitem_gray_test_mode(make_image_dir):
    d = make_image_dir(n=1, h=64, w=64)
    ds = DatasetJPEG(_opt(str(d), phase="test", is_color=False))
    sample = ds[0]
    assert set(sample.keys()) >= {"L", "H", "L_path", "H_path"}
    assert isinstance(sample["L"], torch.Tensor)
    assert isinstance(sample["H"], torch.Tensor)


def test_jpeg_getitem_train_patch_shape(make_image_dir):
    d = make_image_dir(n=1, h=80, w=80)
    ds = DatasetJPEG(_opt(str(d), phase="train", is_color=False))
    sample = ds[0]
    assert sample["L"].shape[1:] == (64, 64)
    assert sample["H"].shape[1:] == (64, 64)
