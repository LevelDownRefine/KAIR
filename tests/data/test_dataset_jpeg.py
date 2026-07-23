"""Regression tests for ``data.dataset_jpeg``.

Guards the ``img_L.ndim`` -> ``img_H.ndim`` bug that raised
``UnboundLocalError`` in the non-color, test-phase branch, where ``img_L`` was
referenced before it was assigned (found during the data/ code review).
"""
import torch

from data.dataset_jpeg import DatasetJPEG


def _make_dir(tmp_path, write_rgb_png, count=1, size=120):
    d = tmp_path / "jpeg"
    d.mkdir()
    for i in range(count):
        write_rgb_png(d / f"img_{i:03d}.png", size, size, seed=7 + i)
    return str(d)


def _opt(dataroot_H, phase="train", is_color=False):
    return {
        "n_channels": 3,
        "H_size": 64,
        "quality_factor": 40,
        "quality_factor_test": 40,
        "is_color": is_color,
        "dataroot_H": dataroot_H,
        "phase": phase,
    }


def test_jpeg_grayscale_test_mode_no_unboundlocalerror(tmp_path, write_rgb_png):
    dataroot = _make_dir(tmp_path, write_rgb_png)
    ds = DatasetJPEG(_opt(dataroot, phase="test", is_color=False))
    sample = ds[0]
    assert set(sample.keys()) >= {"L", "H", "L_path", "H_path"}
    assert isinstance(sample["L"], torch.Tensor)
    assert isinstance(sample["H"], torch.Tensor)


def test_jpeg_grayscale_train_mode_patch_shape(tmp_path, write_rgb_png):
    dataroot = _make_dir(tmp_path, write_rgb_png)
    ds = DatasetJPEG(_opt(dataroot, phase="train", is_color=False))
    sample = ds[0]
    assert sample["L"].shape[1:] == (64, 64)
    assert sample["H"].shape[1:] == (64, 64)
