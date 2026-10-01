"""Protect the fixed search scope, reuse provenance and parameter selection."""

import pytest
from scripts.marble.multirat_reference import prepare
from scripts.marble.run_alpha_grid import (
    ALPHAS,
    RATS,
    bundle_receipt,
    find_job,
    freeze,
    jobs,
)
from scripts.marble.score_alpha_grid import check_parameters, summarize_grid
from utils.utils_marble import save_json


def test_grid_contains_all_animals_protocols_without_duplicate_outputs(tmp_path):
    work = jobs(tmp_path / "new", tmp_path / "prior", tmp_path / "discovery")
    assert len(work) == 48
    assert sum(j["reused"] for j in work) == 17
    assert len({j["folder"] for j in work}) == 48
    for alpha in ALPHAS:
        for animal in RATS:
            for protocol in ("decoding", "consistency"):
                item = find_job(work, protocol, animal, alpha)
                assert item["alpha"] == alpha
    assert "discovery" in find_job(work, "decoding", "achilles", 1.0)["folder"]
    assert not find_job(work, "consistency", "achilles", 1.0)["reused"]
    with pytest.raises(AssertionError):
        find_job(work + [work[0]], "decoding", "achilles", 0.0)


@pytest.mark.parametrize("alpha", [-0.01, float("inf"), float("nan")])
def test_exporter_rejects_invalid_alpha_before_writing(tmp_path, alpha):
    with pytest.raises(AssertionError):
        prepare(None, None, tmp_path / "input", "achilles", "decoding", alpha)
    assert not (tmp_path / "input").exists()


def test_reuse_rejects_wrong_alpha_and_short_training(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "run").mkdir()
    diagnostic = tmp_path / "input/projection_diagnostics.json"
    job = {"folder": str(tmp_path), "alpha": 0.1, "protocol": "decoding"}
    save_json(diagnostic, {"alpha": 0.0, "dimensions": 20})
    with pytest.raises(AssertionError):
        bundle_receipt(job)
    save_json(diagnostic, {"alpha": 0.1, "dimensions": 20})
    save_json(
        tmp_path / "run/summary.json",
        {
            "seeds": [0, 1, 2],
            "runs": [{"seed": s, "training_epochs": 99} for s in (0, 1, 2)],
        },
    )
    with pytest.raises(AssertionError):
        bundle_receipt(job)


def test_resume_rejects_changed_execution_code(tmp_path):
    work = jobs(tmp_path / "new", tmp_path / "prior", tmp_path / "discovery")
    save_json(tmp_path / "frozen_protocol.json", {"jobs": work, "source_sha256": {}})
    with pytest.raises(AssertionError, match="Execution code changed"):
        freeze(tmp_path, work, resume=True)


def grid_rows():
    rows = []
    for mode in ("eval", "notebook"):
        for alpha in ALPHAS:
            for animal in RATS:
                values = [10.0] * 3
                if alpha == 0.01:
                    values = [4.0 if animal == "achilles" else 20.0] * 3
                elif alpha == 0.03:
                    values = [8.0, 9.0, 10.0]
                elif alpha == 0.1:
                    values = [1.0, 15.0, 20.0]
                for seed, value in enumerate(values):
                    rows.append(
                        {
                            "alpha": alpha,
                            "animal": animal,
                            "mode": mode,
                            "seed": seed,
                            "mae_cm": value,
                            "position_r2": 0.8,
                        }
                    )
    return rows


def test_selection_uses_seed_mean_and_equal_animal_weight():
    rows = grid_rows()
    result = summarize_grid(list(reversed(rows)))
    assert result["best_common"]["alpha"] == 0.03
    assert result["best_common"]["macro_mean_cm"] == 9.0
    assert result["best_common"]["animals_improved"] == 4
    assert result["best_common"]["paired_seeds_improved"] == 8
    assert result["best_per_animal"][0]["alpha"] == 0.01
    assert result["loao"]["macro_mean_cm"] == 9.0
    with pytest.raises(AssertionError):
        summarize_grid(rows[:-1])


def test_loao_parameter_does_not_use_excluded_animals_scores():
    rows = grid_rows()
    before = summarize_grid(rows)["loao"]["folds"][0]
    for row in rows:
        if row["animal"] == "achilles" and row["alpha"] == 1.0:
            row["mae_cm"] = 0.0
    after = summarize_grid(rows)["loao"]["folds"][0]
    assert before["excluded_animal"] == after["excluded_animal"] == "achilles"
    assert before["selected_alpha"] == after["selected_alpha"] == 0.03
    assert before["selection_mean_cm"] == after["selection_mean_cm"]


def test_exact_tie_prefers_smaller_alpha():
    rows = grid_rows()
    for row in rows:
        row["mae_cm"] = 10.0
    result = summarize_grid(rows)
    assert result["best_common"]["alpha"] == 0.0
    assert all(f["selected_alpha"] == 0.0 for f in result["loao"]["folds"])


def test_grid_rejects_a_hidden_network_hyperparameter_change(tmp_path):
    left, right = tmp_path / "candidate", tmp_path / "pca"
    for folder in (left, right):
        (folder / "run").mkdir(parents=True)
        save_json(folder / "run/protocol.json", {"model_parameters": {"dropout": 0.0}})
    check_parameters(left, right)
    save_json(left / "run/protocol.json", {"model_parameters": {"dropout": 0.5}})
    with pytest.raises(AssertionError, match="changed model hyperparameters"):
        check_parameters(left, right)
