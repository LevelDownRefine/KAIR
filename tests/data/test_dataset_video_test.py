"""Unit tests for ``data/dataset_video_test.py`` (video recurrent test datasets).

Covers:
  * ``VideoRecurrentTestDataset`` -- primary paired LQ/GT clip test dataset.
    - ``_make_sample`` non-blind denoising branch: deterministic (``torch.manual_seed(0)``)
      AWGN with an extra concatenated noise-level channel -> exact value check.
    - ``_make_sample`` sr/deblur branch: identity (returns the on-disk LQ sequence).
    - ``__init__`` / ``__len__`` / ``__getitem__`` via a synthetic on-disk corpus.
  * ``SingleVideoRecurrentTestDataset`` and ``VideoTestVimeo90KDataset`` -- lighter
    structure checks (full-sequence / center-frame test sets).

NOTE on numerical verification: for the sr/deblur branch ``_make_sample`` is the
identity, so exact value equality holds. For the non-blind denoising branch the
noise is synthesized with a fixed seed inside ``_make_sample``, so the LQ
tensor is recomputed exactly in the test.
"""
import torch

import utils.utils_image as util
from data.dataset_video_test import (
    VideoRecurrentTestDataset,
    SingleVideoRecurrentTestDataset,
    VideoTestVimeo90KDataset,
)


def _opt(**kw):
    """Default opt dict for the test datasets (all keys required by ``__init__``)."""
    opt = {
        'phase': 'test',
        'cache_data': False,
        'dataroot_gt': 'x',
        'dataroot_lq': 'y',
        'num_frame': 3,
    }
    opt.update(kw)
    return opt


def _make_corpus(tmp_path, write_rgb_png, n_frames=4, h=16, w=16, clip='clip0'):
    """Write a tiny gt/ + lq/ corpus; one subfolder with ``n_frames`` PNGs each.
    Returns ``(gt_root, lq_root)`` (the roots, not the subfolder)."""
    gt_root = tmp_path / 'gt'
    lq_root = tmp_path / 'lq'
    gt = gt_root / clip
    lq = lq_root / clip
    gt.mkdir(parents=True)
    lq.mkdir(parents=True)
    for i in range(n_frames):
        write_rgb_png(gt / f'{i:08d}.png', h=h, w=w, seed=i)
        write_rgb_png(lq / f'{i:08d}.png', h=h, w=w, seed=i + 100)
    return gt_root, lq_root


# ------------------------------------------------------------
# VideoRecurrentTestDataset -- _make_sample (non-blind denoising)
# ------------------------------------------------------------
def test_test_make_sample_sigma_exact():
    sigma = 10
    ds = VideoRecurrentTestDataset(_opt(dataroot_gt='g', dataroot_lq='l', sigma=sigma))
    assert abs(ds.sigma - sigma / 255.) < 1e-9

    t, c, h, w = 4, 3, 8, 8
    imgs_H = torch.rand((t, c, h, w), dtype=torch.float32)
    g, l = ds._make_sample(imgs_H, None, 0)

    # recompute the exact (seed=0) noise that _make_sample produces internally
    noise_level = torch.ones((1, 1, 1, 1)) * ds.sigma
    torch.manual_seed(0)
    expected_noise = torch.normal(mean=0, std=noise_level.expand_as(imgs_H))
    expected_L = torch.cat([imgs_H + expected_noise, noise_level.expand(t, 1, h, w)], 1)

    assert g.shape == (t, c, h, w)
    assert l.shape == (t, c + 1, h, w)
    assert torch.equal(g, imgs_H)            # H is passed through unchanged
    assert torch.equal(l, expected_L)        # L == H + seeded AWGN, + noise-level channel
    assert l.dtype == torch.float32


def test_test_make_sample_sr_identity():
    ds = VideoRecurrentTestDataset(_opt(dataroot_gt='g', dataroot_lq='l'))  # no sigma
    assert ds.sigma == 0
    g0 = torch.rand((4, 3, 8, 8), dtype=torch.float32)
    l0 = torch.rand((4, 3, 8, 8), dtype=torch.float32)
    g, l = ds._make_sample(g0, l0, 0)
    assert torch.equal(g, g0)               # sr/deblur branch returns LQ unchanged
    assert torch.equal(l, l0)


# ------------------------------------------------------------
# VideoRecurrentTestDataset -- __init__ / __len__ / __getitem__
# ------------------------------------------------------------
def test_test_init_and_len(tmp_path, write_rgb_png):
    gt_root, lq_root = _make_corpus(tmp_path, write_rgb_png, n_frames=4)
    ds = VideoRecurrentTestDataset(_opt(dataroot_gt=str(gt_root), dataroot_lq=str(lq_root)))
    # one subfolder -> one clip sample
    assert len(ds) == 1
    assert ds.folders == ['clip0']


def test_test_getitem(tmp_path, write_rgb_png):
    gt_root, lq_root = _make_corpus(tmp_path, write_rgb_png, n_frames=4, h=16, w=16)
    ds = VideoRecurrentTestDataset(_opt(dataroot_gt=str(gt_root), dataroot_lq=str(lq_root)))
    out = ds[0]
    assert set(out.keys()) == {'L', 'H', 'folder', 'lq_path'}
    # full sequence is returned (no windowing for the test set)
    assert out['L'].shape == (4, 3, 16, 16)
    assert out['H'].shape == (4, 3, 16, 16)
    assert out['L'].dtype == torch.float32
    assert out['H'].dtype == torch.float32
    assert out['folder'] == 'clip0'
    assert isinstance(out['lq_path'], list) and len(out['lq_path']) == 4


# ------------------------------------------------------------
# SingleVideoRecurrentTestDataset
# ------------------------------------------------------------
def test_single_getitem(tmp_path, write_rgb_png):
    _, lq_root = _make_corpus(tmp_path, write_rgb_png, n_frames=4, h=16, w=16, clip='clipA')
    ds = SingleVideoRecurrentTestDataset(_opt(dataroot_lq=str(lq_root)))
    assert len(ds) == 1
    out = ds[0]
    assert set(out.keys()) == {'L', 'folder', 'lq_path'}
    assert out['L'].shape == (4, 3, 16, 16)
    assert out['L'].dtype == torch.float32
    assert out['folder'] == 'clipA'


# ------------------------------------------------------------
# VideoTestVimeo90KDataset (center-frame GT + LQ window)
# ------------------------------------------------------------
def test_vimeo90k_getitem(tmp_path, write_rgb_png):
    sub = '00001/0001'
    gt_root = tmp_path / 'gt'
    lq_root = tmp_path / 'lq'
    gt = gt_root / sub
    lq = lq_root / sub
    gt.mkdir(parents=True)
    lq.mkdir(parents=True)
    write_rgb_png(gt / 'im4.png', h=16, w=16, seed=4)
    for i in (3, 4, 5):
        write_rgb_png(lq / f'im{i}.png', h=16, w=16, seed=i)
    meta = tmp_path / 'meta.txt'
    meta.write_text(f'{sub} 7 (256,448,3)\n')

    ds = VideoTestVimeo90KDataset(_opt(
        dataroot_gt=str(gt_root), dataroot_lq=str(lq_root), meta_info_file=str(meta),
        num_frame=3, temporal_scale=1, pad_sequence=False, mirror_sequence=False))
    assert len(ds) == 1
    out = ds[0]
    # 3 LQ neighbors (t=3) and a single center GT (t=1)
    assert out['L'].shape == (3, 3, 16, 16)
    assert out['H'].shape == (1, 3, 16, 16)
    assert out['L'].dtype == torch.float32
    assert out['H'].dtype == torch.float32
    assert out['folder'] == sub
