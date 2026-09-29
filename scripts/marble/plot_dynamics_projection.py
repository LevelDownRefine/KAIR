"""Plot every prespecified projection condition, without selecting on test MAE."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from utils.utils_marble import read_json


def plot(source, output):
    summary = read_json(source / "summary.json")
    records = read_json(source / "completed_results.json")
    assert len(summary) == 4 and len(records) == 12
    labels = ["PCA", r"$\alpha=0.1$", r"$\alpha=1$", r"$\alpha=10$"]
    positions = np.arange(4)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout="constrained")
    means = [row["mae_mean_cm"] for row in summary]
    deviations = [row["mae_sd_cm"] for row in summary]
    axes[0].errorbar(
        positions,
        means,
        yerr=deviations,
        fmt="o",
        capsize=5,
        color="#007f91",
        label="Mean +/- seed SD",
        zorder=3,
    )
    for index, row in enumerate(summary):
        values = [r["mae_cm"] for r in records if r["condition"] == row["condition"]]
        axes[0].scatter(
            index + np.linspace(-0.13, 0.13, 3),
            values,
            s=23,
            color="#65758b",
            alpha=0.8,
            label="Individual seeds" if index == 0 else None,
        )
    axes[0].set(title="Held-out position decoding", ylabel="MAE (cm); lower is better")
    axes[0].legend(fontsize=8)
    for axis, split, title in zip(
        axes[1:],
        ("train", "test"),
        ("Training projection", "Held-out projection"),
        strict=True,
    ):
        for kind, color in (("state", "#007f91"), ("increment", "#cb6e27")):
            values = [
                100 * row["projection"][split][f"{kind}_residual_fraction"]
                for row in summary
            ]
            axis.plot(positions, values, "o-", color=color, label=kind.capitalize())
        axis.set(title=title, ylabel="Residual squared energy (%)")
        axis.legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Achilles | 120 to 20 dimensions | unchanged MARBLE | 3 seeds x 100 epochs"
    )
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = output / f"projection_comparison.{extension}"
        fig.savefig(path, dpi=180)
        if extension == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8"
            )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)
