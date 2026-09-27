"""Search boundaries and eligibility must dominate apparent validation gains."""

import pytest
import torch

from main_search_marble_theory import (
    BUDGETS,
    LAGS,
    PURGE,
    RATS,
    evaluate_candidate,
    lagged,
    split_ranges,
    summarize,
)


def records(passed):
    return [
        {
            "lag": lag,
            "condition_budget": budget,
            "seed": seed,
            "task": task,
            "validation_error_ratio": 1.0,
            "certificates": [{"nontrivial_certificate": passed} for _ in range(2)],
        }
        for lag in LAGS
        for budget in BUDGETS
        for seed in (0, 1, 2)
        for task in ("decoding", *RATS)
    ]


def test_no_candidate_selected_when_theory_fails():
    result = summarize(records(False))
    assert "selected" in result and result["selected"] is None
    assert (
        "behavioral_benchmark_run" in result and not result["behavioral_benchmark_run"]
    )


def test_ineligible_best_validation_score_cannot_win():
    data = records(True)
    for row in data:
        assert "lag" in row and "condition_budget" in row
        if row["lag"] == LAGS[0] and row["condition_budget"] == BUDGETS[0]:
            row["validation_error_ratio"] = 0.0001
            row["certificates"] = [{"nontrivial_certificate": False} for _ in range(2)]
    result = summarize(data)
    assert "selected" in result
    selected = result["selected"]
    assert selected is not None and "lag" in selected and "condition_budget" in selected
    assert selected["lag"] == LAGS[0] and selected["condition_budget"] == BUDGETS[1]


def test_temporal_pairs_stay_inside_purged_windows():
    stop, start = split_ranges(500)
    assert start - stop == PURGE
    times = torch.arange(500)
    for lag in LAGS:
        x, y = lagged(times[:stop], lag)
        u, v = lagged(times[start:], lag)
        assert torch.all(y - x == lag) and torch.all(v - u == lag)
        assert int(y.max()) < stop and int(u.min()) >= start


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_validation_cannot_change_fitted_ridge_or_certificates(device):
    torch.manual_seed(7)
    train = torch.randn(200, 6, device=device, dtype=torch.float64)
    validation = torch.randn(80, 6, device=device, dtype=torch.float64)
    first = evaluate_candidate(train, validation, lag=4, budget=30, rank=2)
    second = evaluate_candidate(train, 20 * validation + 7, lag=4, budget=30, rank=2)
    for key in ("ridge", "condition_number", "certificates"):
        assert key in first and key in second
        assert first[key] == second[key]
    assert "validation_neural_mse" in first and "validation_neural_mse" in second
    assert first["validation_neural_mse"] != second["validation_neural_mse"]
