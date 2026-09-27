"""Avoid treating half-written checkpoints as reusable completed fits."""

import pytest

from scripts.marble import finish_unfrozen
from scripts.marble.complete_unfrozen import complete_method
from utils.utils_marble import read_json, save_json, sha256


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


def test_seal_reuses_original_partial_and_new_artifacts_without_overwrite(
    tmp_path, monkeypatch
):
    original, partial, output = [
        tmp_path / name for name in ("original", "partial", "output")
    ]
    for name, path in (
        ("ORIGINAL", original),
        ("PARTIAL", partial),
        ("OUTPUT", output),
    ):
        monkeypatch.setattr(finish_unfrozen, name, path)
    monkeypatch.setattr(finish_unfrozen.experiment, "SEEDS", (0,))
    monkeypatch.setattr(finish_unfrozen.experiment, "RATS", ())
    monkeypatch.setattr(finish_unfrozen, "verify_sources", lambda protocol: None)
    methods = ("initial", *finish_unfrozen.experiment.METHODS)
    sources = (original, original, partial / "recovery_fits", output)
    for method, source in zip(methods, sources, strict=True):
        folder = source / "seed-0/decoding" / method
        folder.mkdir(parents=True)
        save_json(folder / "diagnostics.json", {"fit_seconds": 1.0})
        (folder / "embeddings.pt").write_bytes(method.encode())
        (folder / "model.pt").write_bytes(method.encode())
    (original / "seed-0/decoding/pairs.pt").write_bytes(b"fixed pairs")
    save_json(output / "protocol.json", {"fixed": True})
    before = {
        str(p.relative_to(original)): sha256(p)
        for p in original.rglob("*")
        if p.is_file()
    }
    save_json(partial / "interruption.json", {"original_files_sha256": before})
    finish_unfrozen.seal()
    assert len(read_json(output / "fit_complete.json")["outputs_sha256"]) == 4
    for method in methods:
        assert (
            output / "seed-0/decoding" / method / "embeddings.pt"
        ).read_bytes() == method.encode()
    assert all(sha256(original / name) == digest for name, digest in before.items())
    with pytest.raises(AssertionError):
        finish_unfrozen.seal()
