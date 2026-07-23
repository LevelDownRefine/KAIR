"""Unit tests for ``data/dataset_l.py`` (DatasetL).

DatasetL only loads L (no H, no degradation). Covers ``__init__`` /
``__len__`` / ``__getitem__``.
"""
import numpy as np
import torch

import utils.utils_image as util
from data.dataset_l import DatasetL


def _img_L(h=64, w=64, seed=0):
    """Deterministic synthetic RGB uint8 image of shape (h, w, 3) (seeded)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _opt(dataroot_L, **kw):
    """Default opt dict for DatasetL; only dataroot_L is required."""
    opt = {"n_channels": 3, "dataroot_L": dataroot_L}
    opt.update(kw)
    return opt


# ------------------------------------------------------------
# __init__
# ------------------------------------------------------------
def test_l_init(make_image_dir):
    d = make_image_dir(n=3)
    ds = DatasetL(_opt(str(d)))
    assert ds.n_channels == 3
    assert len(ds.paths_L) == 3


def test_l_init_empty_raises(make_image_dir):
    import pytest
    d = make_image_dir(n=0, name="empty")
    with pytest.raises(AssertionError):
        DatasetL(_opt(str(d)))


# ------------------------------------------------------------
# __len__
# ------------------------------------------------------------
def test_l_len(make_image_dir):
    d = make_image_dir(n=4)
    ds = DatasetL(_opt(str(d)))
    assert len(ds) == 4


# ------------------------------------------------------------
# __getitem__
# ------------------------------------------------------------
def test_l_getitem(make_image_dir):
    d = make_image_dir(n=2, h=64, w=64)
    ds = DatasetL(_opt(str(d)))
    out = ds[0]
    assert set(out.keys()) == {"L", "L_path"}
    assert out["L"].dtype == torch.float32
    assert out["L"].shape == (3, 64, 64)
    # value matches the on-disk load
    expected = util.uint2tensor3(util.imread_uint(ds.paths_L[0], 3))
    assert torch.equal(out["L"], expected)
