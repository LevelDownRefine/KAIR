"""Run the prespecified four-condition MARBLE projection experiment and audits."""

import argparse
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from utils.utils_marble import read_json, save_json, sha256

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = (("pca", 0.0), ("joint-01", 0.1), ("joint-1", 1.0), ("joint-10", 10.0))


def run(output, repository, reference_python):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new experiment directory: {output}")
    output.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(ROOT),
        PYTHONIOENCODING="utf-8",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(ROOT / "results/.matplotlib"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        NUMBA_NUM_THREADS="4",
    )
    save_json(
        output / "design.json",
        {
            "conditions": CONDITIONS,
            "primary_alpha": 1.0,
            "dimensions": 20,
            "seeds": [0, 1, 2],
            "epochs": 100,
            "selection": "No alpha selection on test scores; report all conditions",
            "protocol_sha256": sha256(ROOT / "docs/MARBLE_DYNAMICS_PROJECTION.md"),
            "runner_sha256": sha256(Path(__file__)),
        },
    )
    initializations, masks, records = None, None, []
    for name, alpha in CONDITIONS:
        source, destination = output / name / "input", output / name / "run"
        source.parent.mkdir()
        commands = [
            [
                str(reference_python),
                "-u",
                "-m",
                "scripts.marble.dynamics_reference",
                "--marble-repo",
                str(repository),
                "--output",
                str(source),
                "--alpha",
                str(alpha),
            ],
            [
                sys.executable,
                "-u",
                str(ROOT / "main_train_marble.py"),
                "--input",
                str(source),
                "--output",
                str(destination),
            ],
            [
                str(reference_python),
                "-u",
                "-m",
                "scripts.marble.check_original",
                "--marble-repo",
                str(repository),
                "--input",
                str(source),
                "--run",
                str(destination),
            ],
        ]
        for phase, command in zip(("prepare", "train", "audit"), commands, strict=True):
            logger.info("%s: %s", name, phase)
            with (source.parent / f"{phase}.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1800,
                )
            if phase == "prepare":
                hashes = [
                    read_json(source / f"seed-{seed}/initialization.json")[
                        "initial_state_sha256"
                    ]
                    for seed in (0, 1, 2)
                ]
                diagnostics = read_json(source / "projection_diagnostics.json")
                current_masks = [
                    diagnostics["graphs"][split]["split_masks_sha256"]
                    for split in ("train", "test")
                ]
                if initializations is None:
                    initializations, masks = hashes, current_masks
                assert initializations == hashes, (
                    "Initialization changed across methods"
                )
                assert masks == current_masks, "Train graph validation split changed"
        summary = read_json(destination / "summary.json")
        assert "runs" in summary
        for result in summary["runs"]:
            assert "results" in result and "eval" in result["results"]
            metrics = result["results"]["eval"]
            records.append(
                {
                    "condition": name,
                    "alpha": alpha,
                    "seed": result["seed"],
                    "mae_cm": 100 * metrics["mean_absolute_error_m"],
                    "r2": metrics["position_r2"],
                    "best_epoch": result["best_epoch_zero_based"] + 1,
                    "training_seconds": result["training_seconds"],
                }
            )
        save_json(output / "completed_results.json", records)
        logger.info("%s complete: %s", name, records[-3:])
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    aggregate = []
    for name, alpha in CONDITIONS:
        subset = [r for r in records if r["condition"] == name]
        aggregate.append(
            {
                "condition": name,
                "alpha": alpha,
                "mae_mean_cm": float(np.mean([r["mae_cm"] for r in subset])),
                "mae_sd_cm": float(np.std([r["mae_cm"] for r in subset], ddof=1)),
                "r2_mean": float(np.mean([r["r2"] for r in subset])),
                "projection": read_json(
                    output / name / "input/projection_diagnostics.json"
                ),
            }
        )
    save_json(output / "summary.json", aggregate)
    logger.info("All four conditions and original-code audits completed: %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marble-repo", type=Path, default=ROOT.parent / "MARBLE")
    parser.add_argument(
        "--reference-python",
        type=Path,
        default=ROOT.parent / "MARBLE/reproduction/.venv/Scripts/python.exe",
    )
    args = parser.parse_args()
    run(args.output, args.marble_repo.resolve(), args.reference_python.resolve())
