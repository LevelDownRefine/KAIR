"""Preserve an interrupted run and finish missing fits without changing settings.

Run with python -m scripts.marble.complete_unfrozen. Original files are read-only;
the combined result records both the interruption and this execution amendment.
"""

import argparse
import copy
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import main_explore_marble_unfrozen as experiment
from utils.utils_marble import read_json, save_json, sha256

logger = logging.getLogger(__name__)


def complete_method(path):
    required = ("diagnostics.json", "embeddings.pt", "model.pt")
    present = [(path / name).is_file() for name in required]
    if any(present):
        assert all(present), f"Partial serialized artifacts need inspection: {path}"
    return all(present)


def run(original, output):
    assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    assert not output.exists() and not (original / "fit_complete.json").exists()
    assert not (original / "summary.json").exists(), "No recovery after evaluation"
    protocol = read_json(original / "protocol.json")
    for name, digest in protocol["source_sha256"].items():
        assert sha256(experiment.ROOT / name) == digest
    assert tuple(protocol["methods"]) == experiment.METHODS
    input_path, rat_input, baseline = map(
        Path, (protocol["input"], protocol["rat_input"], protocol["baseline"])
    )
    assert sha256(input_path / "training_input.pt") == protocol["input_sha256"]
    assert sha256(rat_input / "rat_consistency.pt") == protocol["rat_input_sha256"]
    for name, digest in protocol["baseline_sha256"].items():
        assert sha256(baseline / name) == digest
    output.mkdir(parents=True)
    stage = output / "recovery_fits"
    stage.mkdir()
    completed = sorted(original.glob("seed-*/*/*/diagnostics.json"))
    manifest = {
        "observed_exit_code": 1,
        "exception_output": None,
        "cause": "Unconfirmed; a roughly 15-minute session limit is suspected",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "original": str(original.resolve()),
        "original_protocol_sha256": sha256(original / "protocol.json"),
        "completed_learned_fits": sum(p.parent.name != "initial" for p in completed),
        "interrupted_fit": "seed-2/decoding/full_no_prior",
        "retry_scope": "Restart missing fits with fixed settings; reuse completed fits",
        "original_files_sha256": {
            str(p.relative_to(original)): sha256(p)
            for p in sorted(original.glob("seed-*/*/*/*"))
            if p.is_file()
        },
    }
    assert manifest["completed_learned_fits"] == 31
    interrupted = original / manifest["interrupted_fit"]
    assert interrupted.is_dir() and not any(interrupted.iterdir())
    save_json(output / "interruption.json", manifest)
    protocol = copy.deepcopy(protocol)
    protocol["recovery"] = {
        "original_protocol_sha256": manifest["original_protocol_sha256"],
        "interruption_sha256": sha256(output / "interruption.json"),
        "learning_protocol_changed": False,
    }
    for name in (
        "scripts/marble/complete_unfrozen.py",
        "docs/MARBLE_UNFROZEN_RECOVERY.md",
    ):
        protocol["source_sha256"][name] = sha256(experiment.ROOT / name)
    save_json(output / "protocol.json", protocol)
    pack = torch.load(input_path / "training_input.pt", weights_only=True)
    rats = torch.load(rat_input / "rat_consistency.pt", weights_only=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    methods = experiment.METHODS
    for seed in experiment.SEEDS:
        checkpoint = torch.load(
            baseline / f"seed-{seed}/best_model.pth", weights_only=True
        )
        reference = torch.load(
            baseline / f"seed-{seed}/embeddings.pt", weights_only=True
        )["eval"]
        for task in ("decoding", *experiment.RATS):
            relative = Path(f"seed-{seed}") / task
            prior = original / relative
            missing = tuple(
                method for method in methods if not complete_method(prior / method)
            )
            if missing:
                logger.info("Complete %s methods=%s", relative, missing)
                destination = stage / relative
                destination.mkdir(parents=True)
                experiment.METHODS = missing
                if task == "decoding":
                    experiment.fit_task(
                        pack["params"],
                        checkpoint["model_state_dict"],
                        pack["train_graph"],
                        reference["train"],
                        pack["test_graph"],
                        reference["test"],
                        destination,
                        seed,
                    )
                else:
                    animal = rats["animals"][task]
                    experiment.fit_task(
                        animal["params"],
                        animal["state"],
                        animal["graph"],
                        animal["modes"]["eval"]["embedding"],
                        None,
                        None,
                        destination,
                        seed,
                    )
                if complete_method(prior / "initial"):
                    first = torch.load(
                        prior / "initial/embeddings.pt", weights_only=True
                    )
                    second = torch.load(
                        destination / "initial/embeddings.pt", weights_only=True
                    )
                    for name in first:
                        torch.testing.assert_close(
                            first[name], second[name], rtol=1e-10, atol=1e-11
                        )
                if (prior / "pairs.pt").exists():
                    first = torch.load(prior / "pairs.pt", weights_only=True)
                    second = torch.load(destination / "pairs.pt", weights_only=True)
                    for name in first:
                        assert torch.equal(first[name], second[name])
            target = output / relative
            target.mkdir(parents=True)
            pair_source = (
                prior / "pairs.pt"
                if (prior / "pairs.pt").exists()
                else stage / relative / "pairs.pt"
            )
            shutil.copyfile(pair_source, target / "pairs.pt")
            for method in ("initial", *methods):
                source = (
                    prior / method
                    if complete_method(prior / method)
                    else stage / relative / method
                )
                assert complete_method(source)
                shutil.copytree(source, target / method)
            save_json(
                output / "recovery_progress.json",
                {
                    "completed_task": str(relative),
                    "seconds": time.perf_counter() - started,
                },
            )
    experiment.METHODS = methods
    for name, digest in manifest["original_files_sha256"].items():
        assert sha256(original / name) == digest
    outputs = sorted(output.glob("seed-*/*/*/embeddings.pt"))
    assert len(outputs) == 60
    diagnostics = [read_json(p) for p in output.glob("seed-*/*/*/diagnostics.json")]
    save_json(
        output / "fit_complete.json",
        {
            "seconds": None,
            "completed_method_seconds": sum(d["fit_seconds"] for d in diagnostics),
            "recovery_wall_seconds": time.perf_counter() - started,
            "peak_allocated_mib": None,
            "recovery_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "original_process_peak_unavailable": True,
            "protocol_sha256": sha256(output / "protocol.json"),
            "outputs_sha256": {str(p.relative_to(output)): sha256(p) for p in outputs},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(arguments.original, arguments.output)
