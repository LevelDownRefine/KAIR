import os
import yaml

import pytest

from utils import utils_benchmark


@pytest.fixture
def bench_yml(tmp_path, monkeypatch):
    p = tmp_path / 'benchmark.yml'
    monkeypatch.setattr(utils_benchmark, 'BENCHMARK_YML', str(p))
    return str(p)


def _load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def test_writes_new_entry(bench_yml):
    """A fresh weight/test-set/noise triple is written as the third level."""
    utils_benchmark.save_benchmark('w', 'ts', 29.22, 0.8278, noise_level=25)
    data = _load(bench_yml)
    assert data['w']['ts'][25] == {'psnr': 29.22, 'ssim': 0.8278}


def test_multiple_noise_levels_under_same_testset(bench_yml):
    """One weight/test set can carry several noise levels side by side."""
    utils_benchmark.save_benchmark('w', 'ts', 30.0, 0.8, noise_level=15)
    utils_benchmark.save_benchmark('w', 'ts', 29.0, 0.78, noise_level=25)
    data = _load(bench_yml)
    assert data['w']['ts'][15]['psnr'] == 30.0
    assert data['w']['ts'][25]['psnr'] == 29.0


def test_real_noise_string_key(bench_yml):
    """Real-noise sets use the 'real' string key alongside numeric noise keys."""
    utils_benchmark.save_benchmark('w', 'real_faces', 32.2, 0.8978, noise_level='real')
    data = _load(bench_yml)
    assert data['w']['real_faces']['real'] == {'psnr': 32.2, 'ssim': 0.8978}


def test_existing_entry_verified_when_close(bench_yml):
    """Re-running the same triple within tolerance verifies instead of overwriting."""
    utils_benchmark.save_benchmark('w', 'ts', 29.22, 0.8278, noise_level=25)
    utils_benchmark.save_benchmark('w', 'ts', 29.2201, 0.82781, noise_level=25)  # must not raise
    data = _load(bench_yml)
    # verified path does not overwrite the stored value
    assert data['w']['ts'][25]['psnr'] == 29.22


def test_noise_levels_sorted_ascending_on_write(bench_yml):
    """Noise-level keys are stored sorted ascending, with string keys like 'real' last."""
    utils_benchmark.save_benchmark('w', 'ts', 29.0, 0.78, noise_level=25)
    utils_benchmark.save_benchmark('w', 'ts', 32.2, 0.90, noise_level='real')
    utils_benchmark.save_benchmark('w', 'ts', 30.0, 0.80, noise_level=15)
    utils_benchmark.save_benchmark('w', 'ts', 27.0, 0.75, noise_level=50)
    data = _load(bench_yml)
    assert list(data['w']['ts'].keys()) == [15, 25, 50, 'real']


def test_existing_entry_asserts_on_mismatch(bench_yml):
    """A conflicting re-run fails loudly rather than silently overwriting."""
    utils_benchmark.save_benchmark('w', 'ts', 29.22, 0.8278, noise_level=25)
    with pytest.raises(AssertionError):
        utils_benchmark.save_benchmark('w', 'ts', 99.99, 0.8278, noise_level=25)
