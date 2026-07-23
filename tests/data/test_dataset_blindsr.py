"""Unit tests for ``data/dataset_blindsr.py`` (DatasetBlindSR).

Covers ``__init__`` / ``__len__`` / ``_make_sample`` / ``__getitem__``.

NUMERICAL VERIFICATION: in test mode ``_make_sample`` applies exactly
``degradation_bsrgan`` / ``degradation_bsrgan_plus`` to a ``uint2single`` H. We
recompute the expected result independently with the same ``blindsr`` primitive
and identically seeded RNGs (``random`` + ``np.random``) and assert equality,
which confirms the wiring (correct sf / lq_patchsize / shuffle_prob / use_sharp)
and the degradation order.
"""
import random
import numpy as np
import torch

import utils.utils_image as util
from utils import utils_blindsr as blindsr
from data.dataset_blindsr import DatasetBlindSR


def _img_H(h=64, w=64, seed=0):
    """Deterministic synthetic RGB uint8 image of shape (h, w, 3) (seeded)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _opt(**kw):
    """Default opt dict for DatasetBlindSR; scale/shuffle_prob/... are required (read with no .get())."""
    opt = {"phase": "test", "n_channels": 3, "scale": 4, "shuffle_prob": 0.1,
           "use_sharp": False, "degradation_type": "bsrgan", "lq_patchsize": 16, "H_size": 64}
    opt.update(kw)
    return opt


# ------------------------------------------------------------
# _make_sample (test mode): exact bsrgan degradation
# ------------------------------------------------------------
def test_blindsr_make_sample_test_bsrgan_exact():
    sf, lq = 4, 16
    ds = DatasetBlindSR(_opt(scale=sf, lq_patchsize=lq, H_size=lq * sf,
                             degradation_type='bsrgan', paths_H=['dummy.png']))
    known_H = _img_H(64, 64, seed=0)

    random.seed(123)
    np.random.seed(123)
    img_H_out, img_L = ds._make_sample(known_H, 0)

    random.seed(123)
    np.random.seed(123)
    exp_H = util.uint2single(known_H)
    exp_L, exp_H2 = blindsr.degradation_bsrgan(exp_H, sf, lq_patchsize=lq, isp_model=None)

    assert np.array_equal(img_L, exp_L)
    assert np.array_equal(img_H_out, exp_H2)


def test_blindsr_make_sample_test_bsrgan_plus_exact():
    sf, lq = 4, 16
    ds = DatasetBlindSR(_opt(scale=sf, lq_patchsize=lq, H_size=lq * sf,
                             degradation_type='bsrgan_plus', shuffle_prob=0.5,
                             use_sharp=False, paths_H=['dummy.png']))
    known_H = _img_H(80, 80, seed=5)

    random.seed(7)
    np.random.seed(7)
    img_H_out, img_L = ds._make_sample(known_H, 0)

    random.seed(7)
    np.random.seed(7)
    exp_H = util.uint2single(known_H)
    exp_L, exp_H2 = blindsr.degradation_bsrgan_plus(exp_H, sf, shuffle_prob=0.5,
                                                    use_sharp=False, lq_patchsize=lq)

    assert np.array_equal(img_L, exp_L)
    assert np.array_equal(img_H_out, exp_H2)


# ------------------------------------------------------------
# __init__
# ------------------------------------------------------------
def test_blindsr_init_defaults(make_image_dir):
    d = make_image_dir(n=2)
    ds = DatasetBlindSR(_opt(dataroot_H=str(d)))
    assert ds.sf == 4
    assert ds.shuffle_prob == 0.1
    assert ds.use_sharp is False
    assert ds.degradation_type == 'bsrgan'
    assert ds.lq_patchsize == 16
    assert ds.patch_size == 16 * 4
    assert len(ds.paths_H) == 2


# ------------------------------------------------------------
# __len__
# ------------------------------------------------------------
def test_blindsr_len(make_image_dir):
    d = make_image_dir(n=4)
    ds = DatasetBlindSR(_opt(dataroot_H=str(d)))
    assert len(ds) == 4


# ------------------------------------------------------------
# __getitem__ (end-to-end, test mode)
# ------------------------------------------------------------
def test_blindsr_getitem(make_image_dir):
    d = make_image_dir(n=1, h=128, w=128)
    ds = DatasetBlindSR(_opt(scale=4, lq_patchsize=16, H_size=64, dataroot_H=str(d), phase='test'))
    out = ds[0]

    assert set(out.keys()) == {"L", "H", "L_path", "H_path"}
    assert out["H"].dtype == torch.float32 and out["L"].dtype == torch.float32
    assert out["L"].shape == (3, 16, 16)   # lq_patchsize
    assert out["H"].shape == (3, 64, 64)   # lq_patchsize * sf
    assert out["L_path"] == out["H_path"]
