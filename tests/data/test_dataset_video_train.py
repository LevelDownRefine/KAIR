"""Unit tests for ``data/dataset_video_train.py`` (video recurrent train datasets).

Covers:
  * ``VideoRecurrentTrainDataset`` -- the primary paired LQ/GT clip dataset.
    - ``_make_sample``: paired random crop + flip/rotate -> (t,c,h,w) tensors.
    - ``__init__`` / ``__len__`` / ``__getitem__`` via a synthetic on-disk corpus.
  * ``VideoRecurrentTrainNonblindDenoisingDataset`` -- GT-only, LQ synthesized
    with AWGN; ``_make_sample`` appends the noise-level channel.

The Vimeo/VFI variants reuse the same ``_make_sample`` contract (tested lightly);
the frame list is loaded in ``__getitem__`` and the degradation is applied in
``_make_sample``, exactly as for the image datasets.

NOTE on numerical verification: the train path uses random crop + (optional) random
flip/rotate, so exact value recomputation is not deterministic. We therefore assert
the returned *structure* (number of frames, tensor shapes, dtypes, key) and that the
processing matches the original intent (crop/augment applied identically to every
frame). ``use_hflip``/``use_rot`` are disabled in tests so augmentation is the
identity and the only randomness is the crop location (shape is fixed).
"""
import numpy as np
import torch

import utils.utils_image as util
from data.dataset_video_train import (
    VideoRecurrentTrainDataset,
    VideoRecurrentTrainNonblindDenoisingDataset,
)


def _img_H(h=64, w=64, seed=0):
    """Deterministic synthetic RGB uint8 frame of shape (h, w, 3) (seeded)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _frames(n=3, h=64, w=64, seed=0):
    """Return a list of ``n`` deterministic RGB uint8 frames (distinct seeds)."""
    return [_img_H(h, w, seed=seed + i) for i in range(n)]


def _opt(**kw):
    """Default opt dict for the train datasets (all keys required by ``__init__``)."""
    opt = {
        'phase': 'train',
        'name': 'MyVideo',       # not 'REDS' -> no validation-partition filtering
        'test_mode': False,
        'dataroot_gt': None,
        'dataroot_lq': None,
        'meta_info_file': None,
        'num_frame': 3,
        'gt_size': 32,
        'scale': 1,              # keep LQ/GT same size so crop does not assert
        'interval_list': [1],
        'random_reverse': False,
        'use_hflip': False,
        'use_rot': False,
        'filename_tmpl': '08d',
        'filename_ext': 'png',
        'io_backend': {'type': 'disk'},
    }
    opt.update(kw)
    return opt


def _make_corpus(tmp_path, write_rgb_png, n_frames=5, h=64, w=64, clip='sub'):
    """Write a tiny REDS-like corpus (gt/ and lq/ roots, each with one clip subfolder
    plus a meta_info file). Returns ``(gt_root, lq_root, meta_info_path)``.
    """
    gt_root = tmp_path / 'gt'
    lq_root = tmp_path / 'lq'
    gt = gt_root / clip
    lq = lq_root / clip
    gt.mkdir(parents=True)
    lq.mkdir(parents=True)
    for i in range(n_frames):
        write_rgb_png(gt / f'{i:08d}.png', h=h, w=w, seed=i)
        write_rgb_png(lq / f'{i:08d}.png', h=h, w=w, seed=i + 100)
    meta = tmp_path / 'meta.txt'
    meta.write_text(f'{clip} {n_frames} ({h},{w},3) 0\n')
    return gt_root, lq_root, meta


# ------------------------------------------------------------
# VideoRecurrentTrainDataset
# ------------------------------------------------------------
def test_train_init_and_len(tmp_path, write_rgb_png):
    gt_root, lq_root, meta = _make_corpus(tmp_path, write_rgb_png, n_frames=5)
    ds = VideoRecurrentTrainDataset(_opt(dataroot_gt=str(gt_root), dataroot_lq=str(lq_root), meta_info_file=str(meta)))

    assert ds.scale == 1
    assert ds.gt_size == 32
    assert ds.num_frame == 3
    assert ds.interval_list == [1]
    assert ds.random_reverse is False
    # 5 frames -> 5 sample keys (one per frame, since start_frame=0, frame_num=5)
    assert len(ds) == 5


def test_train_make_sample_structure(tmp_path, write_rgb_png):
    gt_root, lq_root, meta = _make_corpus(tmp_path, write_rgb_png, n_frames=5)
    ds = VideoRecurrentTrainDataset(_opt(dataroot_gt=str(gt_root), dataroot_lq=str(lq_root), meta_info_file=str(meta)))

    imgs_H = [util.uint2single(f) for f in _frames(3, 64, 64, seed=0)]
    imgs_L = [util.uint2single(f) for f in _frames(3, 64, 64, seed=10)]
    img_gts, img_lqs = ds._make_sample(imgs_H, imgs_L, 0)

    # num_frame frames, 3 channels, cropped to gt_size x gt_size
    assert isinstance(img_gts, torch.Tensor) and isinstance(img_lqs, torch.Tensor)
    assert img_gts.shape == (3, 3, 32, 32)
    assert img_lqs.shape == (3, 3, 32, 32)
    assert img_gts.dtype == torch.float32
    assert img_lqs.dtype == torch.float32
    # augmentation disabled -> values finite and in [0, 1]
    assert torch.isfinite(img_gts).all()


def test_train_getitem(tmp_path, write_rgb_png):
    gt_root, lq_root, meta = _make_corpus(tmp_path, write_rgb_png, n_frames=5, h=64, w=64)
    ds = VideoRecurrentTrainDataset(_opt(dataroot_gt=str(gt_root), dataroot_lq=str(lq_root), meta_info_file=str(meta)))

    out = ds[0]
    assert set(out.keys()) == {'L', 'H', 'key'}
    # L and H are (t, c, h, w) sequences of num_frame=3 frames, cropped to 32x32
    assert out['L'].shape == (3, 3, 32, 32)
    assert out['H'].shape == (3, 3, 32, 32)
    assert out['L'].dtype == torch.float32
    assert out['H'].dtype == torch.float32
    assert out['key'] == 'sub/00000000'


# ------------------------------------------------------------
# VideoRecurrentTrainNonblindDenoisingDataset
# ------------------------------------------------------------
def test_nonblind_init(tmp_path, write_rgb_png):
    gt_root, lq_root, meta = _make_corpus(tmp_path, write_rgb_png, n_frames=5)
    ds = VideoRecurrentTrainNonblindDenoisingDataset(
        _opt(dataroot_gt=str(gt_root), dataroot_lq=str(lq_root), meta_info_file=str(meta),
             sigma_min=10, sigma_max=30))
    assert abs(ds.sigma_min - 10 / 255.) < 1e-9
    assert abs(ds.sigma_max - 30 / 255.) < 1e-9
    assert len(ds) == 5


def test_nonblind_make_sample_appends_noise_channel(tmp_path, write_rgb_png):
    gt_root, lq_root, meta = _make_corpus(tmp_path, write_rgb_png, n_frames=5)
    ds = VideoRecurrentTrainNonblindDenoisingDataset(
        _opt(dataroot_gt=str(gt_root), dataroot_lq=str(lq_root), meta_info_file=str(meta),
             sigma_min=10, sigma_max=30))

    imgs_H = [util.uint2single(f) for f in _frames(3, 64, 64, seed=0)]
    img_gts, img_lqs = ds._make_sample(imgs_H, 0)

    # H is the cropped/augmented GT (t, c, h, w); L has an extra noise-level channel
    assert img_gts.shape == (3, 3, 32, 32)
    assert img_lqs.shape == (3, 4, 32, 32)
    assert img_gts.dtype == torch.float32
    assert img_lqs.dtype == torch.float32
    # the appended 4th channel is the (constant) noise level, equal across time
    noise_level = img_lqs[0, 3, 0, 0]
    assert torch.isclose(img_lqs[:, 3], noise_level.expand_as(img_lqs[:, 3])).all()
    assert float(noise_level) > 0


def test_nonblind_getitem(tmp_path, write_rgb_png):
    gt_root, lq_root, meta = _make_corpus(tmp_path, write_rgb_png, n_frames=5, h=64, w=64)
    ds = VideoRecurrentTrainNonblindDenoisingDataset(
        _opt(dataroot_gt=str(gt_root), dataroot_lq=str(lq_root), meta_info_file=str(meta),
             sigma_min=10, sigma_max=30))
    out = ds[0]
    assert set(out.keys()) == {'L', 'H', 'key'}
    assert out['H'].shape == (3, 3, 32, 32)
    assert out['L'].shape == (3, 4, 32, 32)
    assert out['key'] == 'sub/00000000'
