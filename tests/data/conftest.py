"""Shared fixtures for the data/ unit tests.

``write_rgb_png`` writes a small random RGB PNG (stored as BGR on disk, the
way ``cv2.imread`` expects) so the datasets can be exercised against real
image files without committing any binary test data.
"""
import numpy as np
import cv2
import pytest


@pytest.fixture
def write_rgb_png():
    def _write(path, h=128, w=128, seed=0):
        rng = np.random.RandomState(seed)
        img = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return _write


@pytest.fixture
def make_image_dir(tmp_path, write_rgb_png):
    """Factory that writes ``n`` synthetic RGB PNGs into a fresh tmp dir.

    Returns the directory Path. Use it to exercise disk-backed dataset methods
    (``__init__`` path scanning, ``_load_img_H``, ``__getitem__``) without
    committing any binary test assets.
    """
    def _make(n=3, h=64, w=64, name="imgs"):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            write_rgb_png(d / "{:03d}.png".format(i), h=h, w=w, seed=i)
        return d

    return _make
