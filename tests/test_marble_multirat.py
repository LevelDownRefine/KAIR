"""Prevent unpaired controls and posthoc selection in multi-rat summaries."""

import pytest
from scripts.marble.run_multirat_projection import check_pair
from scripts.marble.score_multirat_projection import (
    RATS,
    novel_pair_mean,
    summarize_decoding,
)
from utils.utils_marble import save_json


@pytest.mark.parametrize("mismatch", ["initialization", "split"])
def test_paired_conditions_reject_confound(tmp_path, mismatch):
    paths = [tmp_path / name for name in ("control", "candidate")]
    for directory in paths:
        directory.mkdir()
        for seed in (0, 1, 2):
            (directory / f"seed-{seed}").mkdir()
            save_json(
                directory / f"seed-{seed}/initialization.json",
                {"initial_state_sha256": str(seed)},
            )
        save_json(
            directory / "projection_diagnostics.json",
            {
                "graphs": {
                    split: {"split_masks_sha256": split} for split in ("train", "test")
                }
            },
        )
    check_pair(*paths)
    if mismatch == "initialization":
        save_json(
            paths[0] / "seed-1/initialization.json",
            {"initial_state_sha256": "different"},
        )
    else:
        save_json(
            paths[0] / "projection_diagnostics.json",
            {
                "graphs": {
                    split: {"split_masks_sha256": "different"}
                    for split in ("train", "test")
                }
            },
        )
    with pytest.raises(AssertionError):
        check_pair(*paths)


def test_summary_uses_every_seed_including_unfavorable_results():
    rows = [
        {
            "animal": animal,
            "mode": mode,
            "condition": condition,
            "seed": seed,
            "mae_cm": value,
            "position_r2": 0.5,
        }
        for animal in RATS
        for mode in ("eval", "notebook")
        for condition, values in (("pca", [9, 10, 11]), ("joint-01", [1, 15, 20]))
        for seed, value in enumerate(values)
    ]
    result = summarize_decoding(list(reversed(rows)))
    for value in result:
        assert value["pca_mean_cm"] == 10
        assert value["joint_mean_cm"] == 12
        assert value["delta_cm"] == 2
        assert value["seeds_improved"] == 1
        assert value["discovery"] == (value["animal"] == "achilles")
    with pytest.raises(AssertionError):
        summarize_decoding(rows[:-1])


def test_novel_pairs_exclude_discovery_in_both_directions():
    pairs = [[a, b] for a in RATS for b in RATS if a != b]
    scores = [0.9 if "achilles" in pair else 0.2 for pair in pairs]
    assert novel_pair_mean({"pairs": pairs, "scores": scores}) == pytest.approx(0.2)
