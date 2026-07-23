"""Regression tests for ``data.dataset_video_train``.

Guards the hardcoded ``img_results[:7]`` / ``img_results[7:]`` split in
``VideoRecurrentTrainVimeoDataset.__getitem__``, which is only correct when
``num_frame == 7`` (found during the data/ code review).
"""
import torch

from data.dataset_video_train import (
    VideoRecurrentTrainDataset,
    VideoRecurrentTrainNonblindDenoisingDataset,
    VideoRecurrentTrainVimeoDataset,
    VideoRecurrentTrainVimeoVFIDataset,
)


def _make_vimeo_clip(tmp_path, write_rgb_png, num_frame=5):
    # For num_frame=5 the Vimeo neighbor list is [2,3,4,5,6]
    # (formula: i + (9 - num_frame)//2 for i in range(num_frame)).
    gt_root = tmp_path / "gt"
    lq_root = tmp_path / "lq"
    clip, seq = "00001", "0001"
    (gt_root / clip / seq).mkdir(parents=True)
    (lq_root / clip / seq).mkdir(parents=True)
    for n in range(2, 2 + num_frame):
        write_rgb_png(gt_root / clip / seq / f"im{n}.png", 128, 128, seed=10 + n)
        write_rgb_png(lq_root / clip / seq / f"im{n}.png", 128, 128, seed=20 + n)
    meta = tmp_path / "meta.txt"
    meta.write_text(f"{clip}/{seq} 7 (128,128,3) 00000\n")
    return str(gt_root), str(lq_root), str(meta)


def _opt(gt_root, lq_root, meta, num_frame=5):
    return {
        "dataroot_gt": gt_root,
        "dataroot_lq": lq_root,
        "meta_info_file": meta,
        "io_backend": {"type": "disk"},
        "num_frame": num_frame,
        "gt_size": 64,
        "scale": 1,
        "random_reverse": False,
        "use_hflip": False,
        "use_rot": False,
        "temporal_scale": 1,
    }


def test_vimeo_frame_split_respects_num_frame(tmp_path, write_rgb_png):
    gt_root, lq_root, meta = _make_vimeo_clip(tmp_path, write_rgb_png, num_frame=5)
    ds = VideoRecurrentTrainVimeoDataset(_opt(gt_root, lq_root, meta, num_frame=5))
    sample = ds[0]
    assert sample["L"].shape[0] == 5
    assert sample["H"].shape[0] == 5
    assert "key" in sample


def _make_reds_tree(tmp_path, write_rgb_png, clip="000", n_frames=7, size=64):
    gt_root = tmp_path / "gt"
    lq_root = tmp_path / "lq"
    for root in (gt_root, lq_root):
        c = root / clip
        c.mkdir(parents=True)
        for i in range(n_frames):
            write_rgb_png(c / f"{i:08d}.png", size, size, seed=200 + i)
    meta = tmp_path / "meta_reds.txt"
    # REDS-style meta line: "folder frame_num _ start_frame"
    meta.write_text(f"{clip} {n_frames} (64,64,3) 0\n")
    return str(gt_root), str(lq_root), str(meta)


def _train_opt(gt_root, lq_root, meta, num_frame=3, name="generic"):
    return {
        "dataroot_gt": gt_root,
        "dataroot_lq": lq_root,
        "meta_info_file": meta,
        "io_backend": {"type": "disk"},
        "num_frame": num_frame,
        "gt_size": 64,
        "scale": 1,
        "name": name,
        "val_partition": "REDS4",
        "test_mode": False,
        "interval_list": [1],
        "random_reverse": False,
        "use_hflip": False,
        "use_rot": False,
    }


def test_video_recurrent_train_dataset_basic(tmp_path, write_rgb_png):
    gt_root, lq_root, meta = _make_reds_tree(tmp_path, write_rgb_png)
    ds = VideoRecurrentTrainDataset(_train_opt(gt_root, lq_root, meta, num_frame=3))
    assert len(ds) == 7  # 7 GT frames -> 7 clip keys
    s = ds[0]
    assert set(s) >= {"L", "H", "key"}
    # L and H are (num_frame, c, h, w) tensors
    assert s["L"].dim() == 4 and s["L"].shape[0] == 3
    assert s["H"].shape[0] == 3


def test_video_recurrent_train_nonblind_denoising(tmp_path, write_rgb_png):
    gt_root, lq_root, meta = _make_reds_tree(tmp_path, write_rgb_png)
    opt = _train_opt(gt_root, lq_root, meta, num_frame=3)
    opt.update({"sigma_min": 0, "sigma_max": 50})
    ds = VideoRecurrentTrainNonblindDenoisingDataset(opt)
    s = ds[0]
    assert set(s) >= {"L", "H", "key"}
    # L = cat(img_lqs (3ch), noise-level map (1ch)) -> 4 channels
    assert s["L"].shape[1] == 4
    assert s["H"].shape[0] == 3


def test_video_recurrent_train_vimeo_vfi(tmp_path, write_rgb_png):
    gt_root, lq_root, meta = _make_vimeo_clip(tmp_path, write_rgb_png, num_frame=5)
    opt = _opt(gt_root, lq_root, meta, num_frame=5)
    ds = VideoRecurrentTrainVimeoVFIDataset(opt)
    s = ds[0]
    assert set(s) >= {"L", "H", "key"}
    # VFI: many LQ input frames, single middle GT frame (im4.png)
    assert s["L"].shape[0] == 5
    assert s["H"].shape[0] == 1

