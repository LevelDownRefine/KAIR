"""Unit tests for ``data/dataset_srmd.py`` (DatasetSRMD).

Covers:
  * ``_make_sample`` -- test mode exact PCA-reduced kernel + degradation L;
                        train mode patch-pair structure.
  * ``__init__``     -- sf / patch / sigma / sigma_test defaults and that the
    PCA projection matrix is loaded.
  * ``__len__``      -- via a real image directory.
  * ``__getitem__``  -- end-to-end (sigma_test=0 -> no noise), tensor dtype /
    shape / values, degradation-map channel count, L_path == H_path.

The PCA matrix is loaded relative to CWD (``kernels/srmd_pca_pytorch.mat``), so
every test that constructs the dataset ``chdir``s to the project root.
"""
import os
from pathlib import Path

import numpy as np
import torch
import hdf5storage

import utils.utils_image as util
from utils import utils_sisr
from data.dataset_srmd import DatasetSRMD

ROOT = Path(__file__).resolve().parents[2]


def _img_H(h=64, w=64, seed=0):
    """Deterministic synthetic RGB uint8 image of shape (h, w, 3) (seeded)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _opt(**kw):
    """Default opt dict for DatasetSRMD; scale/sigma/sigma_test required (no .get())."""
    opt = {"phase": "test", "n_channels": 3, "H_size": 96,
           "scale": 4, "sigma": [0, 50], "sigma_test": 0}
    opt.update(kw)
    return opt


# ------------------------------------------------------------
# _make_sample
# ------------------------------------------------------------
def test_srmd_make_sample_test_exact_pca_and_degradation(monkeypatch):
    monkeypatch.chdir(ROOT)  # srmd_pca_pytorch.mat is loaded relative to CWD
    sf = 2
    ds = DatasetSRMD(_opt(scale=sf, sigma=[0, 50], sigma_test=0))
    img_H = _img_H(64, 64, seed=9)

    img_H_out, img_L, k_reduced = ds._make_sample(img_H, 0)

    # fixed test-mode kernel: anisotropic_Gaussian(theta=pi, l1=l2=0.1)
    kernel = utils_sisr.anisotropic_Gaussian(ksize=ds.ksize, theta=np.pi, l1=0.1, l2=0.1)
    k = np.reshape(kernel, (-1), order="F")
    p = hdf5storage.loadmat(os.path.join("kernels", "srmd_pca_pytorch.mat"))["p"]
    expected_kr = np.dot(p, k)
    assert np.allclose(k_reduced.numpy(), expected_kr, atol=1e-6)
    assert k_reduced.shape == (expected_kr.shape[0],)

    # H is modcropped + single.
    expected_H = util.uint2single(util.modcrop(img_H, sf))
    assert np.array_equal(img_H_out, expected_H)

    # L = srmd_degradation(H, kernel, sf) = convolve then bicubic downsample.
    expected_L = utils_sisr.srmd_degradation(expected_H, kernel, sf).astype(np.float32)
    assert np.allclose(img_L, expected_L, atol=1e-6)


def test_srmd_make_sample_train_is_patch_pair(monkeypatch):
    monkeypatch.chdir(ROOT)
    sf = 2
    ds = DatasetSRMD(_opt(phase="train", scale=sf, H_size=48, sigma=[0, 50], sigma_test=0))
    img_H = _img_H(80, 80, seed=4)

    img_H_out, img_L, k_reduced = ds._make_sample(img_H, 0)
    assert img_H_out.shape == (48, 48, 3)
    assert img_L.shape == (24, 24, 3)
    assert k_reduced.ndim == 1


# ------------------------------------------------------------
# __init__
# ------------------------------------------------------------
def test_srmd_init_defaults(monkeypatch, make_image_dir):
    monkeypatch.chdir(ROOT)
    d = make_image_dir(n=2)
    ds = DatasetSRMD(_opt(dataroot_H=str(d)))
    assert ds.sf == 4
    assert ds.patch_size == 96
    assert ds.L_size == 96 // 4
    assert ds.sigma_min == 0 and ds.sigma_max == 50
    assert ds.sigma_test == 0
    assert ds.p is not None and ds.ksize == int(np.sqrt(ds.p.shape[-1]))
    assert len(ds.paths_H) == 2


# ------------------------------------------------------------
# __len__
# ------------------------------------------------------------
def test_srmd_len(monkeypatch, make_image_dir):
    monkeypatch.chdir(ROOT)
    d = make_image_dir(n=3)
    ds = DatasetSRMD(_opt(dataroot_H=str(d)))
    assert len(ds) == 3


# ------------------------------------------------------------
# __getitem__
# ------------------------------------------------------------
def test_srmd_getitem(monkeypatch, make_image_dir):
    monkeypatch.chdir(ROOT)
    d = make_image_dir(n=1, h=64, w=64)
    ds = DatasetSRMD(_opt(dataroot_H=str(d), scale=2, sigma=[0, 50], sigma_test=0))
    img_H = ds._load_img_H(0)
    exp_H, exp_L, k_reduced = ds._make_sample(img_H, 0)
    out = ds[0]

    assert set(out.keys()) == {"L", "H", "L_path", "H_path"}
    assert out["H"].dtype == torch.float32
    assert torch.equal(out["H"], util.single2tensor3(exp_H))

    # sigma_test=0 -> no noise added; L[0:3] equals the degraded image channels.
    assert torch.equal(out["L"][:3], util.single2tensor3(exp_L))
    # L = degraded image (3ch) concatenated with the degradation map M
    #     (k_reduced channels + 1 noise-level channel).
    n_k = k_reduced.shape[0]
    assert out["L"].shape[0] == 3 + (n_k + 1)
    assert out["L_path"] == out["H_path"]
