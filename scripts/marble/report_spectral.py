"""Collect all spectral-network results and fixed-point certificates for review."""

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = ("baseline", "initial", "unconstrained", "soft_spectral", "hard_spectral")
LABELS = (
    "Frozen MARBLE",
    "Untrained implicit layer",
    "No spectral constraint",
    "Soft spectral penalty",
    "Hard spectral cap",
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
    diagnostics, history = {}, []
    for path in sorted(source.glob("seed-*/*/*/diagnostics.json")):
        seed, task, method, _ = path.relative_to(source).parts
        value = read(path)
        records = value.pop("history")
        for record in records:
            bound = record.pop("post_update_bounds")
            history.append(
                {
                    "seed": seed,
                    "task": task,
                    "method": method,
                    **record,
                    "post_update_LipN_bound": bound["operator_lipschitz_upper_bound"],
                    "post_update_contraction_bound": bound[
                        "iteration_contraction_upper_bound"
                    ],
                }
            )
        diagnostics[f"{seed}/{task}/{method}"] = value
    assert len(diagnostics) == 60 and len(history) == 1350
    save(output / "fit_diagnostics.json", diagnostics)
    with (output / "iteration_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    certificates = {}
    for method in METHODS[1:]:
        values = [v for key, v in diagnostics.items() if key.endswith("/" + method)]
        solvers = [value["train_solver"] for value in values]
        bounds = [s["operator_lipschitz_upper_bound"] for s in solvers]
        certificates[method] = {
            "fit_count": len(values),
            "LipN_bound_min": min(bounds),
            "LipN_bound_max": max(bounds),
            "D_half_strict_certified_count": sum(
                s["half_strict_pseudocontractive_certified"] for s in solvers
            ),
            "inference_contraction_certified_count": sum(
                s["inference_contraction_certified"] for s in solvers
            ),
            "inference_tolerance_met_count": sum(s["tolerance_met"] for s in solvers),
            "inference_steps_max": max(s["steps"] for s in solvers),
            "inference_residual_max": max(
                s["fixed_point_residual_max"] for s in solvers
            ),
            "unroll_vs_solve_max": max(v["unroll_vs_solve_max"] for v in values),
            "sampled_D_jacobian_norm_max": max(
                v["jacobians"]["sampled_D_jacobian_norm_max"] for v in values
            ),
            "sampled_D_symmetric_eigenvalue_max": max(
                v["jacobians"]["sampled_D_symmetric_eigenvalue_max"] for v in values
            ),
            "all_representations_full_rank": all(
                v["representation"]["effective_rank"]
                == len(v["representation"]["singular_values"])
                for v in values
            ),
        }
    save(output / "certificates.json", certificates)
    paired = {}
    for method in METHODS[3:]:
        records = []
        for candidate, control, baseline in zip(
            summary[method]["runs"],
            summary["unconstrained"]["runs"],
            summary["baseline"]["runs"],
            strict=True,
        ):
            records.append(
                {
                    "seed": candidate["seed"],
                    "mae_change_from_control_cm": 100
                    * (
                        candidate["decoding"]["mean_absolute_error_m"]
                        - control["decoding"]["mean_absolute_error_m"]
                    ),
                    "mae_change_from_baseline_cm": 100
                    * (
                        candidate["decoding"]["mean_absolute_error_m"]
                        - baseline["decoding"]["mean_absolute_error_m"]
                    ),
                    "consistency_change_from_control": candidate["consistency"][
                        "mean_r2"
                    ]
                    - control["consistency"]["mean_r2"],
                    "consistency_change_from_baseline": candidate["consistency"][
                        "mean_r2"
                    ]
                    - baseline["consistency"]["mean_r2"],
                }
            )
        paired[method] = records
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
                samples, row + np.linspace(-0.11, 0.11, 3), color=color, alpha=0.4, s=18
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
        "MARBLE: anchored implicit layer with spectral constraints", fontsize=14
    )
    fig.supxlabel(
        "Diamonds: means; bars: seed SD, not confidence intervals; dots: all seeds.\n"
        "32D decoding / 3D consistency are separate fits; frozen encoder in every arm.",
        fontsize=9,
    )
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    logger.info(
        "Saved %d diagnostics and %d iterations", len(diagnostics), len(history)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(arguments.input, arguments.output)
