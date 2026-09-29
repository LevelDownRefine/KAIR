"""Run the fixed-alpha, paired multi-rat validation without retuning."""

import argparse
import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from utils.utils_marble import read_json, save_json, sha256

ROOT = Path(__file__).resolve().parents[2]
RATS = ("achilles", "buddy", "cicero", "gatsby")
CONDITIONS = (("pca", 0.0), ("joint-01", 0.1))
logger = logging.getLogger(__name__)


def jobs():
    return [
        (protocol, animal, condition, alpha)
        for protocol in ("decoding", "consistency")
        for animal in (RATS[1:] if protocol == "decoding" else RATS)
        for condition, alpha in CONDITIONS
    ]


def check_pair(source, control):
    """Interventions must share initialization and node split assignments."""
    for seed in (0, 1, 2):
        left = read_json(source / f"seed-{seed}/initialization.json")
        right = read_json(control / f"seed-{seed}/initialization.json")
        key = "initial_state_sha256"
        assert key in left and key in right and left[key] == right[key]
    left = read_json(source / "projection_diagnostics.json")
    right = read_json(control / "projection_diagnostics.json")
    assert "graphs" in left and "graphs" in right
    for split in ("train", "test"):
        key = "split_masks_sha256"
        assert key in left["graphs"][split] and key in right["graphs"][split]
        assert left["graphs"][split][key] == right["graphs"][split][key]


def execute(command, log):
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(ROOT),
        PYTHONUNBUFFERED="1",
        PYTHONUTF8="1",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(ROOT / "results/.matplotlib"),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    logger.info("Starting %s", log)
    with log.open("x", encoding="utf-8") as handle:
        subprocess.run(
            [str(part) for part in command],
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1800,
        )
    logger.info("Finished %s", log)


def run(output, repository, data, resume=False):
    cpu = repository / "reproduction/.venv/Scripts/python.exe"
    gpu = ROOT / ".venv/Scripts/python.exe"
    assert cpu.is_file() and gpu.is_file()
    if output.exists():
        assert resume, "Use --resume only for already completed stages"
        frozen = read_json(output / "frozen_protocol.json")
        assert "jobs" in frozen and frozen["jobs"] == [list(job) for job in jobs()]
    else:
        output.mkdir(parents=True)
        save_json(
            output / "frozen_protocol.json",
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "jobs": jobs(),
                "seeds": [0, 1, 2],
                "epochs": 100,
                "decoding": "q=20, out=32, chronological 80/20 split, frozen BN primary",
                "recording_rows": {
                    animal: 6577 if animal == "buddy" else 10000 for animal in RATS
                },
                "consistency": "q=10, out=3, dropout=0.5, whole-recording training",
                "alpha_selection": "Fixed 0.1 from Achilles discovery; no new-animal tuning",
                "achilles_decoding": "Reuse dynamics-projection-20260929; discovery only",
                "hypothesis": "Fixed alpha improves new-animal decoding and cross-animal consistency",
                "evaluation": "Report every animal and seed, all 12 directed consistency pairs",
                "scope": "Four-rat public subset; not RNN or macaque paper experiments",
                "source_sha256": {
                    str(path.relative_to(ROOT)): sha256(path)
                    for path in (
                        ROOT / "scripts/marble/run_multirat_projection.py",
                        ROOT / "scripts/marble/multirat_reference.py",
                        ROOT / "scripts/marble/train_consistency.py",
                        ROOT / "utils/utils_dynamics_projection.py",
                    )
                },
            },
        )

    def prepare(job):
        protocol, animal, condition, alpha = job
        folder = output / protocol / animal / condition
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / "input"
        if (source / "prepared.json").is_file():
            return folder
        assert shutil.disk_usage(output).free > 2 * 1024**3, "Insufficient disk space"
        execute(
            [
                cpu,
                "-m",
                "scripts.marble.multirat_reference",
                "--marble-repo",
                repository,
                "--data",
                data,
                "--output",
                source,
                "--animal",
                animal,
                "--protocol",
                protocol,
                "--alpha",
                str(alpha),
            ],
            folder / "prepare.log",
        )
        return folder

    work = jobs()
    # One original CPU preparation overlaps at most one modern GPU training job.
    with ThreadPoolExecutor(max_workers=1) as pool:
        prepared = pool.submit(prepare, work[0])
        for index, job in enumerate(work):
            protocol, animal, condition, _ = job
            folder = prepared.result()
            if index + 1 < len(work):
                prepared = pool.submit(prepare, work[index + 1])
            source, destination = folder / "input", folder / "run"
            if condition == "joint-01":
                check_pair(source, folder.parent / "pca/input")
            if (
                protocol == "consistency"
                and not (source / "precision_reference_receipt.json").is_file()
            ):
                execute(
                    [
                        cpu,
                        "-m",
                        "scripts.marble.precision_reference",
                        "--marble-repo",
                        repository,
                        "--input",
                        source,
                    ],
                    folder / "precision.log",
                )
            if not (destination / "summary.json").is_file():
                if protocol == "decoding":
                    options = read_json(ROOT / "options/marble/train_achilles.json")
                    options["task"] = f"marble_{animal}_fixed_alpha_multirat"
                    options["dataset"]["dataroot"] = str(source)
                    save_json(folder / "options.json", options)
                    command = [
                        gpu,
                        "main_train_marble.py",
                        "--opt",
                        folder / "options.json",
                    ]
                else:
                    command = [
                        gpu,
                        "-m",
                        "scripts.marble.train_consistency",
                        "--input",
                        source,
                    ]
                execute([*command, "--output", destination], folder / "train.log")
            if not (destination / "original_code_audit.json").is_file():
                module = (
                    "check_original" if protocol == "decoding" else "audit_consistency"
                )
                execute(
                    [
                        cpu,
                        "-m",
                        f"scripts.marble.{module}",
                        "--marble-repo",
                        repository,
                        "--input",
                        source,
                        "--run",
                        destination,
                    ],
                    folder / "audit.log",
                )
            save_json(folder / "completed.json", {"job": job, "audited": True})
            logger.info("Completed %d/%d: %s", index + 1, len(work), job)
    save_json(output / "completed.json", {"jobs": len(work), "new_training_seeds": 42})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marble-repo", type=Path, default=ROOT.parent / "MARBLE")
    parser.add_argument(
        "--data", type=Path, default=ROOT.parent / "MARBLE-reproduction/paper/data/rat"
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    run(
        arguments.output.resolve(),
        arguments.marble_repo.resolve(),
        arguments.data.resolve(),
        arguments.resume,
    )
