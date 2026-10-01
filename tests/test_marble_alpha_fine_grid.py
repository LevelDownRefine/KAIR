"""Protect local-grid reuse, complete reporting and exclusion of old candidates."""

import pytest
from scripts.marble import run_alpha_grid as runner
from scripts.marble.score_alpha_grid import summarize_grid
from utils.utils_marble import save_json, sha256


def prior_manifest(tmp_path):
    prior = tmp_path / "coarse"
    prior.mkdir()
    work = runner.jobs(prior, tmp_path / "multirat", tmp_path / "discovery")
    save_json(prior / "frozen_protocol.json", {"jobs": work})
    save_json(
        prior / "completed.json",
        {"jobs": len(work), "receipts": {j["folder"]: {"digest": "ok"} for j in work}},
    )
    return prior, work


def test_fine_grid_reuses_24_bundles_without_mutating_coarse_results(
    tmp_path, monkeypatch
):
    prior, old = prior_manifest(tmp_path)
    before = {p.name: sha256(p) for p in prior.iterdir()}
    checked = []

    def receipt(job):
        checked.append(job)
        return {"digest": "ok"}

    monkeypatch.setattr(runner, "bundle_receipt", receipt)
    output = tmp_path / "fine"
    work = runner.fine_jobs(output, prior)
    assert runner.grid_alphas(work) == (0, 0.01, 0.02, 0.03, 0.04, 0.05)
    assert len(work) == 48 and len(checked) == 24
    assert {j["alpha"] for j in work if not j["reused"]} == {0.02, 0.04, 0.05}
    assert 3 * sum(not j["reused"] for j in work) == 72
    for job in work:
        if job["reused"]:
            source = runner.find_job(old, job["protocol"], job["animal"], job["alpha"])
            assert job["folder"] == source["folder"]
        else:
            assert str(output) in job["folder"]
    assert not output.exists()
    assert before == {p.name: sha256(p) for p in prior.iterdir()}


def test_fine_grid_rejects_changed_historical_receipt(tmp_path, monkeypatch):
    prior, _ = prior_manifest(tmp_path)
    monkeypatch.setattr(runner, "bundle_receipt", lambda job: {"digest": "changed"})
    with pytest.raises(AssertionError):
        runner.fine_jobs(tmp_path / "fine", prior)
    assert not (tmp_path / "fine").exists()


def test_incomplete_duplicate_or_colliding_grid_is_rejected(tmp_path):
    work = runner.jobs(tmp_path / "new", tmp_path / "prior", tmp_path / "discovery")
    assert runner.grid_alphas(work) == runner.ALPHAS
    for invalid in (work[:-1], work + [work[0]]):
        with pytest.raises(AssertionError):
            runner.grid_alphas(invalid)
    work[-1]["folder"] = work[0]["folder"]
    with pytest.raises(AssertionError):
        runner.grid_alphas(work)


def test_local_scores_use_new_points_all_seeds_and_no_excluded_alphas():
    rows = [
        {
            "alpha": alpha,
            "animal": animal,
            "mode": mode,
            "seed": seed,
            "mae_cm": {0.04: (8, 9, 10), 0.05: (1, 15, 20)}[alpha][seed]
            if alpha in (0.04, 0.05)
            else 10,
            "position_r2": 0.8,
        }
        for alpha in runner.FINE_ALPHAS
        for mode in ("eval", "notebook")
        for animal in runner.RATS
        for seed in runner.SEEDS
    ]
    result = summarize_grid(rows, runner.FINE_ALPHAS)
    assert [r["alpha"] for r in result["common_alpha"]] == list(runner.FINE_ALPHAS)
    assert result["best_common"]["alpha"] == 0.04
    assert result["best_common"]["macro_mean_cm"] == 9
    assert result["best_common"]["paired_seeds_improved"] == 8
    assert all(f["selected_alpha"] == 0.04 for f in result["loao"]["folds"])
    with pytest.raises(AssertionError):
        summarize_grid(rows[:-1], runner.FINE_ALPHAS)
    with pytest.raises(AssertionError):
        summarize_grid(rows + [{**rows[0], "alpha": 0.1}], runner.FINE_ALPHAS)


def test_resume_rejects_changed_parent_grid(tmp_path, monkeypatch):
    prior, _ = prior_manifest(tmp_path)
    save_json(prior / "aggregate.json", {"status": "original"})
    monkeypatch.setattr(runner, "bundle_receipt", lambda job: {"digest": "ok"})
    monkeypatch.setattr(runner, "check_pair", lambda *args: None)
    monkeypatch.setattr(runner, "source_manifest", lambda: {})
    output = tmp_path / "fine"
    work = runner.fine_jobs(output, prior)
    runner.freeze(output, work, False, prior)
    runner.freeze(output, work, True, prior)
    save_json(prior / "aggregate.json", {"status": "changed"})
    with pytest.raises(AssertionError):
        runner.freeze(output, work, True, prior)
