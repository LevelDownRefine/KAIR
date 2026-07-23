"""Unit tests for ``data/dataset_usrnet.py`` (DatasetUSRNet).

Covers:
  * ``_make_sample`` (test mode) -- fixed validation kernel + zero noise ->
    blur+subsample of a modcropped H, plus kernel / noise_level / sf tensors.
  * ``__init__``     -- patch_size / sigma_max / scales / sf_validation defaults
    and that the validation kernel file is loaded.
  * ``__len__``      -- via a real image directory.
  * ``__getitem__``  -- end-to-end, tensor dtype / shape / values, L_path == H_path.

The validation kernel is loaded relative to CWD (``kernels/kernels_12.mat``), so
every test that constructs the dataset ``chdir``s to the project root.
"""
import os
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import filters

import utils.utils_image as util
from data.dataset_usrnet import DatasetUSRNet

ROOT = Path(__file__).resolve().parents[2]


def _img_H(h=96, w=96, seed=0):
    """Deterministic synthetic RGB uint8 image of shape (h, w, 3) (seeded)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _opt(**kw):
    """Default opt dict for DatasetUSRNet; sigma_max/scales/sf_validation required (no .get())."""
    opt = {"phase": "test", "n_channels": 3, "H_size": 96,
           "sigma_max": 25, "scales": [1, 2, 3, 4], "sf_validation": 3}
    opt.update(kw)
    return opt


# ------------------------------------------------------------
# _make_sample (test mode)
# ------------------------------------------------------------
def test_usrnet_make_sample_test_exact_blur_downsample(monkeypatch):
    monkeypatch.chdir(ROOT)  # kernel mat is loaded relative to CWD
    sf = 3
    ds = DatasetUSRNet(_opt(scales=[1, 2, 3, 4], sf_validation=sf))
    img_H = _img_H(96, 96, seed=5)

    img_L, img_H_out, k, noise_level, sf_out = ds._make_sample(img_H, 0)

    assert sf_out == sf
    # validation kernel: kernels[0,0] normalized to sum 1.
    k0 = loadmat(os.path.join("kernels", "kernels_12.mat"))["kernels"][0, 0].astype(np.float64)
    k0 /= np.sum(k0)
    assert np.allclose(k.numpy().squeeze(), k0, atol=1e-9)

    # H is modcropped then converted to single (float32).
    expected_H = util.uint2single(util.modcrop(img_H, sf))
    assert np.array_equal(img_H_out, expected_H)

    # L = uint2single( convolve(modcrop(H), k_expanded, 'wrap')[::sf, ::sf] ) + 0 noise.
    blurred = filters.convolve(util.modcrop(img_H, sf), np.expand_dims(k0, axis=2), mode="wrap")
    expected_L = util.uint2single(blurred[0::sf, 0::sf, ...])
    assert np.allclose(img_L, expected_L, atol=1e-6)

    # zero validation noise level -> tensor of shape (1,1,1), value 0.
    assert noise_level.shape == (1, 1, 1)
    assert float(noise_level) == 0.0
    # kernel tensor: single2tensor3(expand_dims(float32(k), 2)) -> (1, ksize, ksize)
    assert k.shape == (1, k0.shape[0], k0.shape[1])


# ------------------------------------------------------------
# __init__
# ------------------------------------------------------------
def test_usrnet_init_defaults(monkeypatch, make_image_dir):
    monkeypatch.chdir(ROOT)
    d = make_image_dir(n=2)
    ds = DatasetUSRNet(_opt(dataroot_H=str(d)))
    assert ds.patch_size == 96
    assert ds.sigma_max == 25
    assert ds.scales == [1, 2, 3, 4]
    assert ds.sf_validation == 3
    assert ds.kernels is not None
    assert len(ds.paths_H) == 2


# ------------------------------------------------------------
# __len__
# ------------------------------------------------------------
def test_usrnet_len(monkeypatch, make_image_dir):
    monkeypatch.chdir(ROOT)
    d = make_image_dir(n=3)
    ds = DatasetUSRNet(_opt(dataroot_H=str(d)))
    assert len(ds) == 3


# ------------------------------------------------------------
# __getitem__
# ------------------------------------------------------------
def test_usrnet_getitem(monkeypatch, make_image_dir):
    monkeypatch.chdir(ROOT)
    d = make_image_dir(n=1, h=96, w=96)
    ds = DatasetUSRNet(_opt(dataroot_H=str(d), sf_validation=3, sigma_max=25))
    img_H = ds._load_img_H(0)
    exp_L, exp_H, k, noise_level, sf = ds._make_sample(img_H, 0)
    out = ds[0]

    assert set(out.keys()) == {"L", "H", "k", "sigma", "sf", "L_path", "H_path"}
    assert out["H"].dtype == torch.float32
    assert torch.equal(out["H"], util.single2tensor3(exp_H))
    assert torch.equal(out["L"], util.single2tensor3(exp_L))  # zero noise -> exact
    assert out["sf"] == 3
    assert float(out["sigma"]) == 0.0
    # kernel tensor is single2tensor3(expand_dims(float32(k0), 2)) -> (1, ksize, ksize)
    assert out["k"].ndim == 3 and out["k"].shape[0] == 1 and out["k"].shape[1] == out["k"].shape[2]
    assert out["L_path"] == out["H_path"]
