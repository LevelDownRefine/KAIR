"""Export every fit and iteration, numerical checks, and behavioral comparisons."""

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = ("baseline", "initial", "palm_no_prior", "joint_quadratic", "palm_quadratic")
LABELS = (
    "Frozen MARBLE",
    "Untrained residual layer",
    "PALM / no graph prior",
    "Joint proximal gradient",
    "Alternating PALM",
)
COLORS = ("#223846", "#939CA3", "#A96B3E", "#268CA1", "#167254")
logger = logging.getLogger(__name__)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def report(source, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("protocol.json", "fit_complete.json", "summary.json"):
        shutil.copyfile(source / name, output / name)
    summary = read(source / "summary.json")
    diagnostics, histories, checks = {}, [], {}
    for path in sorted(source.glob("seed-*/*/*/diagnostics.json")):
        seed, task, method, _ = path.relative_to(source).parts
        value = read(path)
        records = value.pop("history")
        for record in records:
            histories.append({"seed": seed, "task": task, "method": method, **record})
        key = f"{seed}/{task}/{method}"
        diagnostics[key] = value
        if records:
            checks[key] = {
                "iterations": len(records),
                "initial_objective": records[0]["objective_before"],
                "final_objective": records[-1]["objective_after"],
                "minimum_majorization_margin": min(
                    r["majorization_margin"] for r in records
                ),
                "minimum_descent_margin": min(r["descent_margin"] for r in records),
                "negative_descent_margin_count": sum(
                    r["descent_margin"] < 0 for r in records
                ),
                "objective_increase_count": sum(
                    r["objective_after"] > r["objective_before"] for r in records
                ),
                "maximum_constraint_excess": max(
                    r["constraint_excess"] for r in records
                ),
                "parameter_step_min": min(r["parameter_step"] for r in records),
                "parameter_step_max": max(r["parameter_step"] for r in records),
                "backtracks_max": max(r["backtracks"] for r in records),
                "backtracks_total": sum(r["backtracks"] for r in records),
                "fit_seconds": value["fit_seconds"],
                "initial_stationarity": value["initial_stationarity"],
                "final_stationarity": value["final_stationarity"],
                "effective_rank": value["representation"]["effective_rank"],
            }
    assert len(diagnostics) == 60 and len(checks) == 45 and len(histories) == 13500
    save(output / "fit_diagnostics.json", diagnostics)
    save(output / "numerical_checks.json", checks)
    with (output / "iteration_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(histories[0]))
        writer.writeheader()
        writer.writerows(histories)
    aggregate = {}
    for method in METHODS[2:]:
        values = [v for key, v in checks.items() if key.endswith("/" + method)]
        aggregate[method] = {
            "fit_count": len(values),
            "fit_seconds_total": sum(v["fit_seconds"] for v in values),
            "objective_increases": sum(v["objective_increase_count"] for v in values),
            "minimum_majorization_margin": min(
                v["minimum_majorization_margin"] for v in values
            ),
            "minimum_descent_margin": min(v["minimum_descent_margin"] for v in values),
            "maximum_constraint_excess": max(
                v["maximum_constraint_excess"] for v in values
            ),
            "parameter_step_min": min(v["parameter_step_min"] for v in values),
            "parameter_step_max": max(v["parameter_step_max"] for v in values),
            "backtracks_max": max(v["backtracks_max"] for v in values),
            "backtracks_total": sum(v["backtracks_total"] for v in values),
            "final_parameter_mapping_min": min(
                v["final_stationarity"]["parameter_proximal_mapping_norm"]
                for v in values
            ),
            "final_parameter_mapping_max": max(
                v["final_stationarity"]["parameter_proximal_mapping_norm"]
                for v in values
            ),
            "final_auxiliary_force_rms_max": max(
                v["final_stationarity"]["auxiliary_force_rms"] for v in values
            ),
        }
    save(output / "checks_summary.json", aggregate)
    paired = {}
    for method, control in (
        ("palm_quadratic", "baseline"),
        ("joint_quadratic", "baseline"),
        ("palm_quadratic", "palm_no_prior"),
        ("palm_quadratic", "joint_quadratic"),
    ):
        records = []
        for candidate, reference in zip(
            summary[method]["runs"], summary[control]["runs"], strict=True
        ):
            records.append(
                {
                    "seed": candidate["seed"],
                    "mae_change_cm": 100
                    * (
                        candidate["decoding"]["mean_absolute_error_m"]
                        - reference["decoding"]["mean_absolute_error_m"]
                    ),
                    "consistency_change": candidate["consistency"]["mean_r2"]
                    - reference["consistency"]["mean_r2"],
                }
            )
        paired[f"{method}_minus_{control}"] = records
    save(output / "paired_changes.json", paired)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, layout="constrained")
    for row, (method, color) in enumerate(zip(METHODS, COLORS, strict=True)):
        record = summary[method]
        for axis, key, sd, samples in (
            (
                axes[0],
                "mae_cm_mean",
                "mae_cm_sd",
                [r["decoding"]["mean_absolute_error_m"] * 100 for r in record["runs"]],
            ),
            (
                axes[1],
                "consistency_mean",
                "consistency_sd",
                [r["consistency"]["mean_r2"] for r in record["runs"]],
            ),
        ):
            axis.errorbar(
                record[key], row, xerr=record[sd], fmt="D", capsize=4, color=color
            )
            axis.scatter(
                samples,
                row + np.linspace(-0.11, 0.11, 3),
                color=color,
                alpha=0.45,
                s=18,
            )
    axes[0].set_yticks(range(len(METHODS)), LABELS)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Position MAE (cm): lower is better")
    axes[1].set_xlabel("Cross-animal consistency R²: higher is better")
    for axis, key in zip(axes, ("mae_cm_mean", "consistency_mean"), strict=True):
        axis.axvline(
            summary["baseline"][key], color="#A9B3BA", linestyle="--", linewidth=1
        )
        axis.grid(axis="x", alpha=0.15)
    fig.suptitle(
        "MARBLE: smooth residual model with provable proximal splits", fontsize=14
    )
    fig.supxlabel(
        "Diamonds: means; bars: seed SD, not confidence intervals; dots: all seeds.\n"
        "32D decoding / 3D consistency use separate fits; all encoders are frozen.",
        fontsize=9,
    )
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for seed, axis in enumerate(axes):
        for method, color in zip(METHODS[3:], COLORS[3:], strict=True):
            rows = [
                r
                for r in histories
                if r["seed"] == f"seed-{seed}"
                and r["task"] == "decoding"
                and r["method"] == method
            ]
            axis.plot(
                [r["iteration"] for r in rows],
                [r["objective_after"] for r in rows],
                label=method,
                color=color,
            )
        axis.set_title(f"Decoding model / seed {seed}")
        axis.set_xlabel("Full-batch sweeps")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Same penalized training objective")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Descent under a fixed budget does not certify a stationary final iterate"
    )
    fig.savefig(output / "objectives.png", dpi=180)
    plt.close(fig)
    logger.info("Saved %d fits and %d iterations", len(checks), len(histories))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(arguments.input, arguments.output)
