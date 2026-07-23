"""Correctness smoke tests for the *video test* ``Dataset`` classes in ``data/``.

Synthetic frame sequences are written into temp dirs (via the ``write_rgb_png``
fixture); each dataset is instantiated, one sample is fetched, and the returned
tensors / dict keys are asserted. No real Vimeo/REDS/DAVIS data or GPU needed.
"""
import torch


def _make_recurrent_test_tree(tmp_path, write_rgb_png):
    gt = tmp_path / "gt"
    lq = tmp_path / "lq"
    for root in (gt, lq):
        sub = root / "sub1"
        sub.mkdir(parents=True)
        for i in range(2):
            write_rgb_png(sub / f"{i:04d}.png", 64, 64, seed=60 + i)
    return str(gt), str(lq)


def _make_single_test_tree(tmp_path, write_rgb_png):
    lq = tmp_path / "lq"
    sub = lq / "sub1"
    sub.mkdir(parents=True)
    for i in range(2):
        write_rgb_png(sub / f"{i:04d}.png", 64, 64, seed=70 + i)
    return str(lq)


def _make_vimeo_tree(tmp_path, write_rgb_png):
    gt = tmp_path / "gt_v"
    lq = tmp_path / "lq_v"
    clip, seq = "00001", "0001"
    (gt / clip / seq).mkdir(parents=True)
    (lq / clip / seq).mkdir(parents=True)
    write_rgb_png(gt / clip / seq / "im4.png", 64, 64, seed=80)
    for i in (3, 4, 5):
        write_rgb_png(lq / clip / seq / f"im{i}.png", 64, 64, seed=80 + i)
    meta = tmp_path / "meta_vimeo.txt"
    meta.write_text(f"{clip}/{seq} 7 (64,64,3)\n")
    return str(gt), str(lq), str(meta)


def test_video_recurrent_test_dataset(tmp_path, write_rgb_png):
    from data.dataset_video_test import VideoRecurrentTestDataset

    gt, lq = _make_recurrent_test_tree(tmp_path, write_rgb_png)
    opt = {"dataroot_gt": gt, "dataroot_lq": lq, "cache_data": False, "num_frame": 2}
    ds = VideoRecurrentTestDataset(opt)
    assert len(ds) == 1
    s = ds[0]
    assert set(s) >= {"L", "H", "folder", "lq_path"}
    assert isinstance(s["L"], torch.Tensor) and s["L"].dim() == 4
    assert s["L"].shape[0] == 2  # two frames in the folder


def test_single_video_recurrent_test_dataset(tmp_path, write_rgb_png):
    from data.dataset_video_test import SingleVideoRecurrentTestDataset

    lq = _make_single_test_tree(tmp_path, write_rgb_png)
    opt = {"dataroot_lq": lq, "cache_data": False, "num_frame": 2}
    ds = SingleVideoRecurrentTestDataset(opt)
    assert len(ds) == 1
    s = ds[0]
    # only LQ is provided -> no 'H' key
    assert set(s) >= {"L", "folder", "lq_path"}
    assert "H" not in s
    assert isinstance(s["L"], torch.Tensor) and s["L"].dim() == 4
    assert s["L"].shape[0] == 2


def test_vimeo90k_test_dataset(tmp_path, write_rgb_png):
    from data.dataset_video_test import VideoTestVimeo90KDataset

    gt, lq, meta = _make_vimeo_tree(tmp_path, write_rgb_png)
    opt = {
        "dataroot_gt": gt, "dataroot_lq": lq, "cache_data": False,
        "num_frame": 3, "meta_info_file": meta,
    }
    ds = VideoTestVimeo90KDataset(opt)  # num_frame=3 -> neighbor_list [3,4,5]
    assert len(ds) == 1
    s = ds[0]
    assert set(s) >= {"L", "H", "folder", "idx", "border", "lq_path", "gt_path"}
    assert isinstance(s["L"], torch.Tensor) and s["L"].dim() == 4
    assert s["L"].shape[0] == 3  # 3 LQ frames
    assert s["H"].shape[0] == 1  # 1 GT (center) frame


def _make_davis_tree(tmp_path, write_rgb_png):
    root = tmp_path / "davis"
    label = root / "cat"
    label.mkdir(parents=True)
    for i in range(7):
        write_rgb_png(label / f"{i:05d}.png", 480, 840, seed=30 + i)
    return str(root)


def _make_ucf101_tree(tmp_path, write_rgb_png):
    root = tmp_path / "ucf"
    clip = root / "clip1"
    clip.mkdir(parents=True)
    for n, seed in (("frame0", 1), ("frame1", 2), ("frame2", 3), ("frame3", 4), ("framet", 5)):
        write_rgb_png(clip / f"{n}.png", 224, 224, seed=40 + seed)
    return str(root)


def _make_vid4_tree(tmp_path, write_rgb_png):
    root = tmp_path / "vid4"
    label = root / "city"
    label.mkdir(parents=True)
    for i in range(7):
        write_rgb_png(label / f"{i:04d}.png", 64, 64, seed=50 + i)
    return str(root)


def test_vfi_davis(tmp_path, write_rgb_png):
    from data.dataset_video_test import VFI_DAVIS

    ds = VFI_DAVIS(_make_davis_tree(tmp_path, write_rgb_png))
    assert len(ds) >= 1
    s = ds[0]
    # L = 4 input frames, H = 1 middle frame
    assert s["L"].shape[0] == 4
    assert s["H"].shape[0] == 1


def test_vfi_ucf101(tmp_path, write_rgb_png):
    from data.dataset_video_test import VFI_UCF101

    ds = VFI_UCF101(_make_ucf101_tree(tmp_path, write_rgb_png))
    assert len(ds) >= 1
    s = ds[0]
    assert s["L"].shape[0] == 4
    assert s["H"].shape[0] == 1


def test_vfi_vid4(tmp_path, write_rgb_png):
    from data.dataset_video_test import VFI_Vid4

    ds = VFI_Vid4(_make_vid4_tree(tmp_path, write_rgb_png))
    assert len(ds) >= 1
    s = ds[0]
    # Vid4: 7 input frames -> first window yields 4 LQ frames + 1 middle GT
    assert s["L"].shape[0] == 4
    assert s["H"].shape[0] == 1
