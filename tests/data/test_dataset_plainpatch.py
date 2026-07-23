"""Regression tests for ``data.dataset_plainpatch``.

Guards the ``self.path_size`` -> ``self.patch_size`` typo that raised
``AttributeError`` during ``__init__`` (buffer allocation) and ``get_patches``
(found during the data/ code review).
"""
import torch

from data.dataset_plainpatch import DatasetPlainPatch


def _opt(roots, num_sampled=2, num_patches=3, H_size=64):
    return {
        "n_channels": 3,
        "H_size": H_size,
        "num_patches_per_image": num_patches,
        "num_sampled": num_sampled,
        "dataroot_H": roots["dataroot_H"],
        "dataroot_L": roots["dataroot_L"],
        "phase": "train",
    }


def test_plainpatch_builds_patch_buffers(tmp_path, write_rgb_png):
    roots = {}
    for key in ("dataroot_H", "dataroot_L"):
        d = tmp_path / key
        d.mkdir(parents=True)
        for i in range(3):
            write_rgb_png(d / f"img_{i:03d}.png", 100, 100, seed=i * 7 + (0 if key == "dataroot_H" else 3))
        roots[key] = str(d)

    ds = DatasetPlainPatch(_opt(roots))
    expected = 2 * 3  # num_sampled * num_patches_per_image
    assert len(ds) == expected
    assert ds.H_data.shape == (expected, 64, 64, 3)
    assert ds.L_data.shape == (expected, 64, 64, 3)


def test_plainpatch_getitem_returns_paired_tensors(tmp_path, write_rgb_png):
    roots = {}
    for key in ("dataroot_H", "dataroot_L"):
        d = tmp_path / key
        d.mkdir(parents=True)
        for i in range(3):
            write_rgb_png(d / f"img_{i:03d}.png", 100, 100, seed=i * 7 + (0 if key == "dataroot_H" else 3))
        roots[key] = str(d)

    ds = DatasetPlainPatch(_opt(roots))
    sample = ds[0]
    assert set(sample.keys()) >= {"L", "H"}
    assert isinstance(sample["L"], torch.Tensor)
    assert isinstance(sample["H"], torch.Tensor)
    assert sample["L"].shape == (3, 64, 64)
    assert sample["H"].shape == (3, 64, 64)
