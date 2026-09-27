"""Avoid treating half-written checkpoints as reusable completed fits."""

import pytest

from scripts.marble.complete_unfrozen import complete_method


def test_completion_requires_all_three_artifacts(tmp_path):
    assert not complete_method(tmp_path / "missing")
    assert not complete_method(tmp_path)
    for name in ("diagnostics.json", "embeddings.pt", "model.pt"):
        (tmp_path / name).write_bytes(b"saved")
    assert complete_method(tmp_path)


def test_half_written_fit_is_rejected_instead_of_retried(tmp_path):
    (tmp_path / "embeddings.pt").write_bytes(b"partial serialization")
    with pytest.raises(AssertionError, match="Partial serialized"):
        complete_method(tmp_path)
