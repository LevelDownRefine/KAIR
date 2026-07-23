"""Tests for the shared ``BaseDataset`` methods used by all four refactored
dataset classes (SR / DnCNN / USRNet / SRMD).

These cover the base-class contract only:
  * ``_make_sample`` is abstract (raises NotImplementedError),
  * ``__init__`` stores opt / n_channels / paths_H / paths_L,
  * ``__len__`` reflects the number of H images,
  * ``_load_img_H`` / ``_load_img_L`` read a real image from disk as uint8 HWC.

A tiny stub subclass satisfies the abstract method so we can instantiate the base.
"""
import cv2
import numpy as np
import pytest

from data.base_dataset import BaseDataset


class _Stub(BaseDataset):
    """Minimal BaseDataset subclass that implements the abstract _make_sample."""
    def _make_sample(self, img_H, index):
        return img_H


def test_base_make_sample_is_abstract():
    # BaseDataset does not implement _make_sample; call it on a bare instance.
    ds = BaseDataset({"phase": "test", "n_channels": 3, "paths_H": ["x.png"]})
    with pytest.raises(NotImplementedError):
        ds._make_sample(np.zeros((4, 4, 3), np.uint8), 0)


def test_base_stores_opt_and_paths(make_image_dir):
    d = make_image_dir(n=2)
    ds = _Stub({"phase": "test", "n_channels": 3, "dataroot_H": str(d)})
    assert ds.opt["n_channels"] == 3
    assert isinstance(ds.paths_H, list) and len(ds.paths_H) == 2
    assert ds.paths_L is None  # no dataroot_L was given


def test_base_n_channels_default_and_override():
    ds_def = _Stub({"phase": "test", "paths_H": ["x.png"]})
    assert ds_def.n_channels == 3  # default
    ds_1 = _Stub({"phase": "test", "n_channels": 1, "paths_H": ["x.png"]})
    assert ds_1.n_channels == 1


def test_base_len(make_image_dir):
    d = make_image_dir(n=5)
    ds = _Stub({"phase": "test", "n_channels": 3, "dataroot_H": str(d)})
    assert len(ds) == 5
    # empty list -> 0
    ds2 = _Stub({"phase": "test", "n_channels": 3, "paths_H": []})
    assert len(ds2) == 0


def test_base_load_img_H_roundtrip(make_image_dir):
    d = make_image_dir(n=1, h=40, w=50)
    ds = _Stub({"phase": "test", "n_channels": 3, "dataroot_H": str(d)})
    img = ds._load_img_H(0)
    assert img.dtype == np.uint8
    assert img.shape == (40, 50, 3)


def test_base_load_img_H_known_array(tmp_path):
    # Write a known RGB array (cv2 stores BGR on disk; imread_uint returns
    # RGB), then confirm _load_img_H round-trips it exactly.
    arr = np.zeros((20, 30, 3), np.uint8)
    arr[5, 5] = [10, 20, 30]
    cv2.imwrite(str(tmp_path / "k.png"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    ds = _Stub({"phase": "test", "n_channels": 3, "paths_H": [str(tmp_path / "k.png")]})
    assert np.array_equal(ds._load_img_H(0), arr)


def test_base_load_img_L(tmp_path, write_rgb_png):
    d = tmp_path / "l"
    d.mkdir()
    write_rgb_png(d / "l0.png", h=32, w=32, seed=0)
    ds = _Stub({"phase": "test", "n_channels": 3, "paths_L": [str(d / "l0.png")]})
    img = ds._load_img_L(0)
    assert img.dtype == np.uint8
    assert img.shape == (32, 32, 3)
