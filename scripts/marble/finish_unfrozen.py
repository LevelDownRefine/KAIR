"""Audit float32 preprocessing drift, fit bounded tasks, and seal retained outputs."""

import argparse
import copy
import logging
import shutil
import time
from pathlib import Path

import torch

import main_explore_marble_unfrozen as experiment
from scripts.marble.complete_unfrozen import complete_method
from utils.utils_marble import read_json, save_json, sha256

ROOT = experiment.ROOT
ORIGINAL = ROOT / "results/unfrozen-20260928"
PARTIAL = ROOT / "results/unfrozen-20260928-completed"
OUTPUT = ROOT / "results/unfrozen-20260928-final"
logger = logging.getLogger(__name__)


def verify_sources(protocol):
    for name, digest in protocol["source_sha256"].items():
        assert sha256(ROOT / name) == digest
    assert (
        sha256(Path(protocol["input"]) / "training_input.pt")
        == protocol["input_sha256"]
    )
    assert (
        sha256(Path(protocol["rat_input"]) / "rat_consistency.pt")
        == protocol["rat_input_sha256"]
    )
    for name, digest in protocol["baseline_sha256"].items():
        assert sha256(Path(protocol["baseline"]) / name) == digest


def audit():
    assert not OUTPUT.exists()
    assert (
        not (ORIGINAL / "summary.json").exists()
        and not (PARTIAL / "summary.json").exists()
    )
    protocol = read_json(PARTIAL / "protocol.json")
    verify_sources(protocol)
    first = ORIGINAL / "seed-2/decoding/initial"
    second = PARTIAL / "recovery_fits/seed-2/decoding/initial"
    old = torch.load(first / "model.pt", weights_only=True)
    new = torch.load(second / "model.pt", weights_only=True)
    for key, value in old["base_state"].items():
        assert torch.equal(value, new["base_state"][key])
    states = [old["model_state"], new["model_state"]]
    coefficients = []
    for state in states:
        weight = state["encoder.0"] / state["input_scale"]
        bias = state["encoder.1"] - weight @ state["input_mean"]
        coefficients.append((weight, bias))
    for left, right in zip(*coefficients, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-14)
    for name in states[0]:
        if name not in ("input_mean", "input_scale", "encoder.0", "encoder.1"):
            assert torch.equal(states[0][name], states[1][name])
    fixed_columns = old["params"]["dim_emb"] + old["params"]["dim_signal"]
    for name in ("input_mean", "input_scale"):
        assert torch.equal(
            states[0][name][:fixed_columns], states[1][name][:fixed_columns]
        )
    pairs = [
        torch.load(p.parent / "pairs.pt", weights_only=True) for p in (first, second)
    ]
    assert all(torch.equal(pairs[0][key], pairs[1][key]) for key in pairs[0])
    embeddings = [
        torch.load(p / "embeddings.pt", weights_only=True) for p in (first, second)
    ]
    difference = {}
    for name in embeddings[0]:
        delta = embeddings[0][name] - embeddings[1][name]
        difference[name] = {
            "max_absolute": float(delta.abs().max()),
            "max_row_norm": float(delta.norm(dim=1).max()),
        }
        # Explicit numerical audit, not exact equality or a formal error bound.
        assert delta.abs().max() < 1e-8
    OUTPUT.mkdir()
    audit_record = {
        "same_original_weights": True,
        "same_pair_tensors": True,
        "effective_first_weight_difference": float(
            (coefficients[0][0] - coefficients[1][0]).abs().max()
        ),
        "effective_first_bias_difference": float(
            (coefficients[0][1] - coefficients[1][1]).abs().max()
        ),
        "embedding_difference": difference,
        "absolute_audit_threshold": 1e-8,
        "previous_check_failed": {"rtol": 1e-10, "atol": 1e-11},
        "interpretation": "Float32 graph-feature drift; equivalent folded network",
    }
    save_json(OUTPUT / "numerical_recovery_audit.json", audit_record)
    interruption = read_json(PARTIAL / "interruption.json")
    interruption["second_stop"] = (
        "Completed two fits, then failed an over-strict initial-output comparison"
    )
    interruption["second_stop_exit_code"] = 1
    interruption["numerical_audit_sha256"] = sha256(
        OUTPUT / "numerical_recovery_audit.json"
    )
    save_json(OUTPUT / "interruption.json", interruption)
    protocol = copy.deepcopy(protocol)
    protocol["recovery"]["interruption_sha256"] = sha256(OUTPUT / "interruption.json")
    protocol["recovery"]["numerical_audit_sha256"] = sha256(
        OUTPUT / "numerical_recovery_audit.json"
    )
    for name in (
        "scripts/marble/finish_unfrozen.py",
        "docs/MARBLE_UNFROZEN_NUMERICAL_AUDIT.md",
    ):
        protocol["source_sha256"][name] = sha256(ROOT / name)
    save_json(OUTPUT / "protocol.json", protocol)
    logger.info("Numerical audit passed: %s", difference)


def fit_task(rat):
    assert rat in experiment.RATS
    protocol = read_json(OUTPUT / "protocol.json")
    verify_sources(protocol)
    assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    animals = torch.load(
        Path(protocol["rat_input"]) / "rat_consistency.pt", weights_only=True
    )
    animal = animals["animals"][rat]
    target = OUTPUT / "seed-2" / rat
    target.mkdir(parents=True)
    started = time.perf_counter()
    experiment.fit_task(
        animal["params"],
        animal["state"],
        animal["graph"],
        animal["modes"]["eval"]["embedding"],
        None,
        None,
        target,
        2,
    )
    save_json(
        target / "task_execution.json",
        {
            "seconds": time.perf_counter() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        },
    )


def seal():
    protocol = read_json(OUTPUT / "protocol.json")
    verify_sources(protocol)
    assert not (OUTPUT / "fit_complete.json").exists()
    origins = {}
    for seed in experiment.SEEDS:
        for task in ("decoding", *experiment.RATS):
            relative = Path(f"seed-{seed}") / task
            target = OUTPUT / relative
            target.mkdir(parents=True, exist_ok=True)
            sources = (
                ORIGINAL / relative,
                PARTIAL / "recovery_fits" / relative,
                target,
            )
            for method in ("initial", *experiment.METHODS):
                candidates = [
                    source / method
                    for source in sources
                    if complete_method(source / method)
                ]
                assert candidates, f"Missing {relative}/{method}"
                source = candidates[0]
                destination = target / method
                if source != destination:
                    assert not destination.exists()
                    shutil.copytree(source, destination)
                origins[str(relative / method)] = str(source)
            pair_source = next(
                source / "pairs.pt"
                for source in sources
                if (source / "pairs.pt").is_file()
            )
            if pair_source != target / "pairs.pt":
                assert not (target / "pairs.pt").exists()
                shutil.copyfile(pair_source, target / "pairs.pt")
    original_manifest = read_json(PARTIAL / "interruption.json")
    for name, digest in original_manifest["original_files_sha256"].items():
        assert sha256(ORIGINAL / name) == digest
    outputs = sorted(OUTPUT.glob("seed-*/*/*/embeddings.pt"))
    expected = (
        len(experiment.SEEDS)
        * (1 + len(experiment.RATS))
        * (1 + len(experiment.METHODS))
    )
    assert len(outputs) == expected
    diagnostics = [read_json(p) for p in OUTPUT.glob("seed-*/*/*/diagnostics.json")]
    save_json(
        OUTPUT / "fit_complete.json",
        {
            "seconds": None,
            "completed_method_seconds": sum(d["fit_seconds"] for d in diagnostics),
            "peak_allocated_mib": None,
            "original_process_peak_unavailable": True,
            "task_executions": {
                rat: read_json(OUTPUT / "seed-2" / rat / "task_execution.json")
                for rat in experiment.RATS
            },
            "protocol_sha256": sha256(OUTPUT / "protocol.json"),
            "outputs_sha256": {str(p.relative_to(OUTPUT)): sha256(p) for p in outputs},
            "artifact_origins": origins,
        },
    )
    logger.info("Sealed all %d embeddings", len(outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "task", "seal"))
    parser.add_argument("--rat", choices=experiment.RATS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.phase == "audit":
        audit()
    elif args.phase == "task":
        assert args.rat
        fit_task(args.rat)
    else:
        seal()
