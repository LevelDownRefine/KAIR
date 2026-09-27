"""Compare a KAIR run with an earlier MARBLE reproduction on identical inputs."""

import argparse
from pathlib import Path

import numpy as np

from utils.utils_marble import read_json, save_json


def compare(current, reference):
    protocol = read_json(current / "protocol.json")
    reference_protocol = read_json(reference / "protocol.json")
    for key in ("input_fingerprints", "model_parameters", "decoder"):
        assert key in protocol and key in reference_protocol
        assert protocol[key] == reference_protocol[key], f"Different protocol: {key}"
    summary = read_json(current / "summary.json")
    old_summary = read_json(reference / "summary.json")
    assert "seeds" in summary and "seeds" in old_summary
    assert summary["seeds"] == old_summary["seeds"]
    results = []
    for seed in summary["seeds"]:
        new = read_json(current / f"seed-{seed}/metrics.json")
        old = read_json(reference / f"seed-{seed}/metrics.json")
        assert "initial_state_sha256" in new and "initial_state_sha256" in old
        assert new["initial_state_sha256"] == old["initial_state_sha256"]
        receipt = read_json(reference / f"seed-{seed}/sampling_receipt.json")
        assert "sampling_sha256" in new and "sha256" in receipt
        assert new["sampling_sha256"] == receipt["sha256"]
        new_history = read_json(current / f"seed-{seed}/loss_history.json")
        old_history = read_json(reference / f"seed-{seed}/loss_history.json")
        differences = {}
        for key in ("train_loss", "val_loss", "lr"):
            assert key in new_history and key in old_history
            assert len(new_history[key]) == len(old_history[key])
            differences[key] = float(
                np.max(np.abs(np.array(new_history[key]) - np.array(old_history[key])))
            )
        results.append(
            {
                "seed": seed,
                "identical_graphs_initialization_sampling": True,
                "best_epoch_zero_based": new["best_epoch_zero_based"],
                "reference_best_epoch_zero_based": old["best_epoch_zero_based"],
                "mae_cm": 100 * new["results"]["eval"]["mean_absolute_error_m"],
                "reference_mae_cm": 100
                * old["results"]["eval"]["mean_absolute_error_m"],
                "max_absolute_history_differences": differences,
            }
        )
    report = {
        "reference": reference.name,
        "scope": "Identical graph, initialization and sampling; independent GPU training",
        "bitwise_training_equivalence_claimed": False,
        "runs": results,
    }
    save_json(current / "previous_run_comparison.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    arguments = parser.parse_args()
    compare(arguments.run, arguments.reference)
