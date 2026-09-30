"""Run a bounded, paired alpha grid, retaining hashes of reused experiments."""

import argparse
import logging
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.marble.run_multirat_projection import RATS, ROOT, check_pair, execute
from utils.utils_marble import read_json, save_json, sha256

ALPHAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
FINE_ALPHAS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05)
SEEDS = (0, 1, 2)
logger = logging.getLogger(__name__)


def condition(alpha):
    assert math.isfinite(alpha) and alpha >= 0
    return "alpha-" + f"{alpha:g}".replace(".", "p")


def grid_alphas(work):
    """Require a complete Cartesian grid with one immutable folder per cell."""
    assert work and all(
        {"alpha", "protocol", "animal", "folder"} <= j.keys() for j in work
    )
    alphas = tuple(sorted({j["alpha"] for j in work}))
    assert alphas[0] == 0 and all(math.isfinite(a) and a >= 0 for a in alphas)
    expected = {
        (a, p, r) for a in alphas for p in ("decoding", "consistency") for r in RATS
    }
    assert {(j["alpha"], j["protocol"], j["animal"]) for j in work} == expected
    assert len(work) == len(expected) == len({j["folder"] for j in work})
    return alphas


def fine_jobs(output, previous_grid):
    """Reuse audited coarse controls by manifest, excluding large alpha candidates."""
    frozen = read_json(previous_grid / "frozen_protocol.json")
    completed = read_json(previous_grid / "completed.json")
    assert "jobs" in frozen and {"jobs", "receipts"} <= completed.keys()
    prior = frozen["jobs"]
    assert grid_alphas(prior) == ALPHAS
    assert completed["jobs"] == len(prior)
    result = []
    for alpha in FINE_ALPHAS:
        for protocol in ("decoding", "consistency"):
            for animal in RATS:
                reused = alpha in ALPHAS
                if reused:
                    job = find_job(prior, protocol, animal, alpha)
                    assert job["folder"] in completed["receipts"]
                    assert bundle_receipt(job) == completed["receipts"][job["folder"]]
                    folder = Path(job["folder"])
                else:
                    folder = output / protocol / animal / condition(alpha)
                result.append(
                    {
                        "alpha": alpha,
                        "protocol": protocol,
                        "animal": animal,
                        "condition": condition(alpha),
                        "folder": str(folder.resolve()),
                        "reused": reused,
                    }
                )
    assert grid_alphas(result) == FINE_ALPHAS
    return result


def jobs(output, previous, discovery):
    result = []
    for alpha in ALPHAS:
        for protocol in ("decoding", "consistency"):
            for animal in RATS:
                reused = alpha in (0.0, 0.1) or (
                    protocol == "decoding" and animal == "achilles" and alpha == 1.0
                )
                if reused:
                    old_condition = {0.0: "pca", 0.1: "joint-01", 1.0: "joint-1"}[alpha]
                    if protocol == "decoding" and animal == "achilles":
                        folder = discovery / old_condition
                    else:
                        folder = previous / protocol / animal / old_condition
                else:
                    folder = output / protocol / animal / condition(alpha)
                result.append(
                    {
                        "alpha": alpha,
                        "protocol": protocol,
                        "animal": animal,
                        "condition": condition(alpha),
                        "folder": str(folder.resolve()),
                        "reused": reused,
                    }
                )
    return result


def find_job(work, protocol, animal, alpha):
    matches = [
        j
        for j in work
        if j["protocol"] == protocol and j["animal"] == animal and j["alpha"] == alpha
    ]
    assert len(matches) == 1
    return matches[0]


def bundle_receipt(job):
    """Require complete audited data, retaining the actual input/output hashes."""
    folder = Path(job["folder"])
    diagnostic = read_json(folder / "input/projection_diagnostics.json")
    assert diagnostic["alpha"] == job["alpha"]
    assert diagnostic["dimensions"] == (20 if job["protocol"] == "decoding" else 10)
    summary = read_json(folder / "run/summary.json")
    assert summary["seeds"] == list(SEEDS)
    assert sorted(r["seed"] for r in summary["runs"]) == list(SEEDS)
    assert all(r["training_epochs"] == 100 for r in summary["runs"])
    paths = [
        "input/training_input.pt",
        "input/projection_diagnostics.json",
        "run/summary.json",
        "run/original_code_audit.json",
        "run/reference_validation.json",
        "run/provenance.json",
    ]
    for seed in SEEDS:
        paths.extend(
            f"run/seed-{seed}/{name}"
            for name in (
                "initialization.json",
                "embeddings.pt",
                "metrics.json",
                "best_model.pth",
            )
        )
        paths.extend(
            f"input/seed-{seed}/{name}"
            for name in ("initialization.json", "sampling_receipt.json")
        )
    # The sampler receipt's digest is checked again when DatasetMARBLE loads it.
    return {name: sha256(folder / name) for name in paths}


def source_manifest():
    names = [
        "scripts/marble/run_alpha_grid.py",
        "scripts/marble/score_alpha_grid.py",
        "scripts/marble/report_alpha_grid.py",
        "tests/test_marble_alpha_fine_grid.py",
        "scripts/marble/run_multirat_projection.py",
        "scripts/marble/multirat_reference.py",
        "scripts/marble/rat_reference.py",
        "scripts/marble/train_consistency.py",
        "scripts/marble/precision_reference.py",
        "scripts/marble/precision_stepwise.py",
        "scripts/marble/check_original.py",
        "scripts/marble/audit_consistency.py",
        "scripts/marble/prepare.py",
        "main_train_marble.py",
        "models/model_marble.py",
        "models/network_marble.py",
        "data/dataset_marble.py",
        "utils/utils_dynamics_projection.py",
        "utils/utils_marble.py",
        "utils/utils_marble_storage.py",
        "options/marble/train_achilles.json",
        "docs/MARBLE_ALPHA_GRID_PROTOCOL.md",
        "docs/MARBLE_ALPHA_GRID_NUMERICS.md",
        "docs/MARBLE_ALPHA_FINE_GRID_PROTOCOL.md",
        "pyproject.toml",
        "uv.lock",
    ]
    return {name: sha256(ROOT / name) for name in names}


def expected_sources(output):
    frozen = read_json(output / "frozen_protocol.json")
    amendment_path = output / "execution_amendment.json"
    if amendment_path.is_file():
        amendment = read_json(amendment_path)
        assert amendment["frozen_protocol_sha256"] == sha256(
            output / "frozen_protocol.json"
        )
        return amendment["source_sha256"]
    return frozen["source_sha256"]


def freeze(output, work, resume, previous_grid=None):
    alphas = grid_alphas(work)
    parent = (
        {
            "root": str(previous_grid),
            "sha256": {
                name: sha256(previous_grid / name)
                for name in ("frozen_protocol.json", "completed.json", "aggregate.json")
            },
        }
        if previous_grid is not None
        else None
    )
    if output.exists():
        assert resume, "Choose a new directory or explicitly resume"
        frozen = read_json(output / "frozen_protocol.json")
        assert frozen["jobs"] == work
        assert expected_sources(output) == source_manifest(), "Execution code changed"
        if previous_grid is not None:
            assert "parent_grid" in frozen and frozen["parent_grid"] == parent
        for job in work:
            if job["reused"]:
                assert frozen["reuse_receipts"][job["folder"]] == bundle_receipt(job)
        return
    receipts = {j["folder"]: bundle_receipt(j) for j in work if j["reused"]}
    for job in work:
        if job["reused"]:
            baseline = find_job(work, job["protocol"], job["animal"], 0.0)
            check_pair(
                Path(job["folder"]) / "input", Path(baseline["folder"]) / "input"
            )
    output.mkdir(parents=True)
    hashes = source_manifest()
    save_json(
        output / "frozen_protocol.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "alphas": alphas,
            "parent_grid": parent,
            "seeds": SEEDS,
            "epochs": 100,
            "jobs": work,
            "source_sha256": hashes,
            "reuse_receipts": receipts,
            "selection": "Exploratory; all four evaluation records previously examined",
            "primary_decoding": "Equal-animal mean of three-seed frozen-BN MAE (cm)",
            "primary_consistency": "Three-seed mean of 12 directed in-sample binned R2 scores",
            "new_training_seeds": 3 * sum(not j["reused"] for j in work),
        },
    )
    with ZipFile(output / "execution_source.zip", "x", ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(ROOT / name, name)


def run(
    output, previous, discovery, repository, data, resume=False, previous_grid=None
):
    cpu = repository / "reproduction/.venv/Scripts/python.exe"
    gpu = ROOT / ".venv/Scripts/python.exe"
    assert cpu.is_file() and gpu.is_file()
    work = (
        fine_jobs(output, previous_grid)
        if previous_grid is not None
        else jobs(output, previous, discovery)
    )
    freeze(output, work, resume, previous_grid)
    fresh = [j for j in work if not j["reused"]]

    def numerics(job):
        source = Path(job["folder"]) / "input"
        target = output / "numerics" / job["animal"] / job["condition"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not (target / "oracle_receipt.json").is_file():
            execute(
                [
                    cpu,
                    "-m",
                    "scripts.marble.precision_stepwise",
                    "--input",
                    source,
                    "--output",
                    target,
                    "--marble-repo",
                    repository,
                ],
                target.with_suffix(".export.log"),
            )
        if not (target / "validation.json").is_file():
            execute(
                [
                    gpu,
                    "-m",
                    "scripts.marble.precision_stepwise",
                    "--input",
                    source,
                    "--output",
                    target,
                ],
                target.with_suffix(".validate.log"),
            )
        return target / "validation.json"

    # Audit all consistency conditions, storing receipts outside historical data.
    for job in work:
        if (
            job["protocol"] == "consistency"
            and (Path(job["folder"]) / "run/summary.json").is_file()
        ):
            numerics(job)

    def prepare(job):
        folder = Path(job["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / "input"
        if not (source / "prepared.json").is_file():
            assert shutil.disk_usage(output).free > 3 * 1024**3
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
                    job["animal"],
                    "--protocol",
                    job["protocol"],
                    "--alpha",
                    job["alpha"],
                ],
                folder / "prepare.log",
            )
        return folder

    with ThreadPoolExecutor(max_workers=1) as pool:
        prepared = pool.submit(prepare, fresh[0])
        for index, job in enumerate(fresh):
            folder = prepared.result()
            if index + 1 < len(fresh):
                prepared = pool.submit(prepare, fresh[index + 1])
            source, destination = folder / "input", folder / "run"
            baseline = find_job(work, job["protocol"], job["animal"], 0.0)
            check_pair(source, Path(baseline["folder"]) / "input")
            numerical_audit = (
                numerics(job) if job["protocol"] == "consistency" else None
            )
            if not (destination / "summary.json").is_file():
                if job["protocol"] == "decoding":
                    options = read_json(ROOT / "options/marble/train_achilles.json")
                    options["task"] = f"marble_{job['animal']}_alpha_grid"
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
                        "--numerics-audit",
                        numerical_audit,
                    ]
                execute([*command, "--output", destination], folder / "train.log")
            if not (destination / "original_code_audit.json").is_file():
                module = (
                    "check_original"
                    if job["protocol"] == "decoding"
                    else "audit_consistency"
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
            receipt = bundle_receipt(job)
            save_json(folder / "completed.json", {"job": job, "sha256": receipt})
            logger.info("Completed new condition %d/%d: %s", index + 1, len(fresh), job)
    assert source_manifest() == expected_sources(output)
    save_json(
        output / "completed.json",
        {
            "jobs": len(work),
            "new_training_seeds": 3 * len(fresh),
            "reuse_training_seeds": 3 * (len(work) - len(fresh)),
            "receipts": {j["folder"]: bundle_receipt(j) for j in work},
            "completed_utc": datetime.now(UTC).isoformat(),
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prior = parser.add_mutually_exclusive_group(required=True)
    prior.add_argument("--previous", type=Path)
    prior.add_argument("--previous-grid", type=Path)
    parser.add_argument(
        "--discovery", type=Path, default=ROOT / "results/dynamics-projection-20260929"
    )
    parser.add_argument("--marble-repo", type=Path, default=ROOT.parent / "MARBLE")
    parser.add_argument(
        "--data", type=Path, default=ROOT.parent / "MARBLE-reproduction/paper/data/rat"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(
        args.output.resolve(),
        args.previous.resolve() if args.previous is not None else None,
        args.discovery.resolve(),
        args.marble_repo.resolve(),
        args.data.resolve(),
        args.resume,
        args.previous_grid.resolve() if args.previous_grid is not None else None,
    )
