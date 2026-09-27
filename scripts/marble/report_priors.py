"""Collect small audit artifacts and plot the completed geometric-prior experiment."""

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
METHODS = (
    "baseline",
    "control",
    "direct_quadratic",
    "admm_quadratic",
    "admm_tv",
    "pnp",
)
LABELS = (
    "Frozen MARBLE",
    "Continuation control",
    "Quadratic / direct",
    "Quadratic / ADMM",
    "Vector TV / ADMM",
    "Adaptive PnP",
)
COLORS = ("#223846", "#8A9198", "#268CA1", "#167254", "#A75B39", "#83549D")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def report(source, output):
    output.mkdir(parents=True, exist_ok=False)
    summary = read(source / "summary.json")
    for name in ("protocol.json", "fit_complete.json", "summary.json"):
        shutil.copyfile(source / name, output / name)
    diagnostics, rows = {}, []
    for path in sorted(source.glob("seed-*/*/*/diagnostics.json")):
        seed, task, method, _ = path.relative_to(source).parts
        value = read(path)
        history = value.pop("history")
        value["first_iteration"] = history[0]
        value["last_iteration"] = history[-1]
        diagnostics[f"{seed}/{task}/{method}"] = value
        for step in history:
            rows.append({"seed": seed, "task": task, "method": method, **step})
    assert len(diagnostics) == 75 and len(rows) == 2250
    save(output / "fit_diagnostics.json", diagnostics)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "iteration_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    paired = {}
    for method in METHODS[2:]:
        records = []
        for candidate, control, baseline in zip(
            summary[method]["runs"],
            summary["control"]["runs"],
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained", sharey=True)
    for row, (method, color) in enumerate(zip(METHODS, COLORS, strict=True)):
        result = summary[method]
        axes[0].errorbar(
            result["mae_cm_mean"],
            row,
            xerr=result["mae_cm_sd"],
            fmt="D",
            color=color,
            capsize=4,
        )
        axes[1].errorbar(
            result["consistency_mean"],
            row,
            xerr=result["consistency_sd"],
            fmt="D",
            color=color,
            capsize=4,
        )
        axes[0].scatter(
            [r["decoding"]["mean_absolute_error_m"] * 100 for r in result["runs"]],
            row + np.array([-0.08, 0, 0.08]),
            color=color,
            alpha=0.5,
            s=22,
        )
        if method != "baseline":
            axes[1].scatter(
                [r["consistency"]["mean_r2"] for r in result["runs"]],
                row + np.array([-0.08, 0, 0.08]),
                color=color,
                alpha=0.5,
                s=22,
            )
    axes[0].set_yticks(range(len(METHODS)), LABELS)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Position MAE (cm): lower is better")
    axes[1].set_xlabel("Cross-animal consistency R²: higher is better")
    axes[0].axvline(
        summary["baseline"]["mae_cm_mean"], color="#A9B3BA", linestyle="--", linewidth=1
    )
    axes[1].axvline(
        summary["baseline"]["consistency_mean"],
        color="#A9B3BA",
        linestyle="--",
        linewidth=1,
    )
    for axis in axes:
        axis.grid(axis="x", alpha=0.15)
    fig.suptitle(
        "MARBLE geometric priors: first fixed-budget experiment",
        fontsize=15,
        fontweight="bold",
    )
    fig.supxlabel(
        "Diamonds: means; bars: seed SD, not confidence intervals; "
        "small points: all seeds.\n"
        "32D decoding / 3D consistency are separate fits. "
        "Four animals share one author-checkpoint baseline.",
        fontsize=9,
    )
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    logger.info(
        "Saved %d fits and %d iteration records to %s",
        len(diagnostics),
        len(rows),
        output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(arguments.input, arguments.output)
