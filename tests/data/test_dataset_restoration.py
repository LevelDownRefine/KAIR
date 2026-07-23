"""Correctness smoke tests for the image-restoration ``Dataset`` classes in ``data/``.

Each test synthesizes a few RGB PNGs into a temp dir (via the ``write_rgb_png``
fixture), instantiates the dataset, fetches one sample, and asserts the
returned dict / tensor shapes are sane. No real training data or GPU required.
"""
import torch


def _image_dir(tmp_path, write_rgb_png, name, count=2, size=128):
    d = tmp_path / name
    d.mkdir(parents=True)
    for i in range(count):
        write_rgb_png(d / f"{i:04d}.png", size, size, seed=11 + i)
    return str(d)


def test_dncnn_train_returns_paired_tensors(tmp_path, write_rgb_png):
    from data.dataset_dncnn import DatasetDnCNN

    opt = {
        "dataroot_H": _image_dir(tmp_path, write_rgb_png, "h"),
        "phase": "train", "n_channels": 3, "H_size": 64,
        "sigma": 25, "sigma_test": 25,
    }
    ds = DatasetDnCNN(opt)
    s = ds[0]
    assert set(s) >= {"L", "H", "H_path", "L_path"}
    assert isinstance(s["L"], torch.Tensor) and s["L"].dim() == 3
    assert s["L"].shape == s["H"].shape


def test_dnpatch_builds_buffers_and_getitem(tmp_path, write_rgb_png):
    from data.dataset_dnpatch import DatasetDnPatch

    opt = {
        "dataroot_H": _image_dir(tmp_path, write_rgb_png, "h", count=1, size=128),
        "phase": "train", "n_channels": 3, "H_size": 32,
        "sigma": 25, "sigma_test": 25,
        "num_patches_per_image": 2, "num_sampled": 1,
    }
    ds = DatasetDnPatch(opt)
    # total_patches = num_sampled * num_patches_per_image
    assert len(ds) == 2
    s = ds[0]
    assert isinstance(s["L"], torch.Tensor) and s["L"].dim() == 3
    assert s["L"].shape == s["H"].shape


def test_dpsr_concats_noise_level_map(tmp_path, write_rgb_png):
    from data.dataset_dpsr import DatasetDPSR

    d = _image_dir(tmp_path, write_rgb_png, "x", count=2, size=128)
    opt = {
        "dataroot_H": d, "dataroot_L": d, "phase": "test", "n_channels": 3,
        "scale": 2, "H_size": 64, "sigma": [0, 50], "sigma_test": 0,
    }
    ds = DatasetDPSR(opt)
    s = ds[0]
    assert set(s) >= {"L", "H", "L_path", "H_path"}
    # L = cat(img_L (3ch), noise-level map (1ch)) -> 4 channels
    assert s["L"].shape[0] == 4
    assert s["H"].dim() == 3


def test_fdcnn_concats_noise_level_map(tmp_path, write_rgb_png):
    from data.dataset_fdncnn import DatasetFDnCNN

    opt = {
        "dataroot_H": _image_dir(tmp_path, write_rgb_png, "h"),
        "phase": "test", "n_channels": 3, "H_size": 64,
        "sigma": [0, 75], "sigma_test": 25,
    }
    ds = DatasetFDnCNN(opt)
    s = ds[0]
    assert set(s) >= {"L", "H", "L_path", "H_path"}
    # L = cat(img_L (3ch), noise-level map (1ch)) -> 4 channels
    assert s["L"].shape[0] == 4


def test_ffdnet_returns_sigma_scalar(tmp_path, write_rgb_png):
    from data.dataset_ffdnet import DatasetFFDNet

    opt = {
        "dataroot_H": _image_dir(tmp_path, write_rgb_png, "h"),
        "phase": "test", "n_channels": 3, "H_size": 64,
        "sigma": [0, 75], "sigma_test": 25,
    }
    ds = DatasetFFDNet(opt)
    s = ds[0]
    assert set(s) >= {"L", "H", "C", "L_path", "H_path"}
    assert s["L"].dim() == 3 and s["H"].dim() == 3
    assert s["C"].dim() == 3  # (1, 1, 1) noise-level scalar


def test_dataset_l_test_mode_returns_only_l(tmp_path, write_rgb_png):
    from data.dataset_l import DatasetL

    opt = {"dataroot_L": _image_dir(tmp_path, write_rgb_png, "l"), "n_channels": 3}
    ds = DatasetL(opt)
    s = ds[0]
    assert set(s) >= {"L", "L_path"}
    assert "H" not in s
    assert s["L"].dim() == 3


def test_plain_train_returns_paired(tmp_path, write_rgb_png):
    from data.dataset_plain import DatasetPlain

    d = _image_dir(tmp_path, write_rgb_png, "x", count=2)
    opt = {
        "dataroot_H": d, "dataroot_L": d, "phase": "train",
        "n_channels": 3, "H_size": 64,
    }
    ds = DatasetPlain(opt)
    s = ds[0]
    assert set(s) >= {"L", "H", "L_path", "H_path"}
    assert s["L"].shape == s["H"].shape


def test_sr_train_returns_paired_with_scale(tmp_path, write_rgb_png):
    from data.dataset_sr import DatasetSR

    # SRResNet with an explicit, properly-downsampled L (H=128 -> L=64 for
    # sf=2) so the LR/GT crops stay spatially aligned.
    d_h = _image_dir(tmp_path, write_rgb_png, "h", count=2, size=128)
    d_l = _image_dir(tmp_path, write_rgb_png, "l", count=2, size=64)
    opt = {
        "dataroot_H": d_h, "dataroot_L": d_l, "phase": "train",
        "n_channels": 3, "scale": 2, "H_size": 64,
    }
    ds = DatasetSR(opt)
    s = ds[0]
    assert set(s) >= {"L", "H", "L_path", "H_path"}
    # H patch == H_size, L patch == H_size // scale
    assert s["H"].shape[1:] == (64, 64)
    assert s["L"].shape[1:] == (32, 32)


def test_srmd_uses_rebuilt_kernel_and_concats_map(tmp_path, write_rgb_png):
    from data.dataset_srmd import DatasetSRMD

    d = _image_dir(tmp_path, write_rgb_png, "x", count=2, size=128)
    opt = {
        "dataroot_H": d, "dataroot_L": d, "phase": "test", "n_channels": 3,
        "scale": 2, "H_size": 64, "sigma": [0, 50], "sigma_test": 0,
    }
    ds = DatasetSRMD(opt)  # requires kernels/srmd_pca_pytorch.mat
    s = ds[0]
    assert set(s) >= {"L", "H", "L_path", "H_path"}
    # L = cat(img_L (3ch), degradation map (k_reduced 15ch + noise 1ch = 16ch))
    assert s["L"].shape[0] == 19


def test_usrnet_test_mode_returns_k_and_sf(tmp_path, write_rgb_png):
    from data.dataset_usrnet import DatasetUSRNet

    opt = {
        "dataroot_H": _image_dir(tmp_path, write_rgb_png, "h", count=2, size=96),
        "phase": "test", "n_channels": 3, "H_size": 96, "sigma_max": 25,
        "scales": [1, 2, 3, 4], "sf_validation": 3,
    }
    ds = DatasetUSRNet(opt)  # requires kernels/kernels_12.mat
    s = ds[0]
    assert set(s) >= {"L", "H", "k", "sigma", "sf", "L_path", "H_path"}
    assert isinstance(s["k"], torch.Tensor) and s["k"].dim() == 3
    assert isinstance(s["sf"], int)


def test_blindsr_train_returns_paired(tmp_path, write_rgb_png):
    import random
    import numpy as np

    # degradation_bsrgan is stochastic; seed for a deterministic test run.
    random.seed(1234)
    np.random.seed(1234)

    from data.dataset_blindsr import DatasetBlindSR

    opt = {
        "dataroot_H": _image_dir(tmp_path, write_rgb_png, "h", count=2, size=256),
        "phase": "train", "n_channels": 3, "scale": 4, "H_size": 256,
        "degradation_type": "bsrgan", "lq_patchsize": 64,
        "shuffle_prob": 0.1, "use_sharp": False,
    }
    ds = DatasetBlindSR(opt)
    s = ds[0]
    assert set(s) >= {"L", "H", "L_path", "H_path"}
    assert s["L"].dim() == 3 and s["H"].dim() == 3
