"""Render the fixed three-method comparison from recorded, sealed result files."""

import argparse
import json
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
METHODS = ("diffusion", "koopman", "transport")
NAMES = ("Diffusion", "Koopman / VAMP", "MARBLE + GW")
COLORS = ("#BA6338", "#197D92", "#7954A1")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def report(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for method, source in zip(METHODS, inputs, strict=True):
        summary = read(source / "summary.json")
        assert summary["method"] == method
        summaries.append(summary)
        folder = output / method
        folder.mkdir()
        for name in ("summary.json", "protocol.json", "fit_complete.json"):
            shutil.copyfile(source / name, folder / name)

    base_mae = summaries[0]["baseline_mae_cm_mean"]
    base_r2 = summaries[0]["baseline_consistency"]["mean_r2"]
    fig, ax = plt.subplots(figsize=(9.2, 6.2), layout="constrained")
    ax.axvline(base_mae, color="#98A2A9", linestyle="--", linewidth=1)
    ax.axhline(base_r2, color="#98A2A9", linestyle="--", linewidth=1)
    ax.scatter(
        [base_mae],
        [base_r2],
        color="#142C3B",
        marker="*",
        s=160,
        label="MARBLE baseline",
        zorder=5,
    )
    for summary, name, color in zip(summaries, NAMES, COLORS, strict=True):
        x = summary["candidate_mae_cm_mean"]
        y = summary["candidate_consistency_mean"]
        ax.errorbar(
            x,
            y,
            xerr=summary["candidate_mae_cm_seed_sd"],
            yerr=summary["candidate_consistency_seed_sd"],
            color=color,
            fmt="o",
            markersize=8,
            capsize=4,
            linewidth=1.7,
            label=name,
            zorder=4,
        )
        ax.scatter(
            [r["candidate"]["mean_absolute_error_m"] * 100 for r in summary["runs"]],
            [r["consistency"]["mean_r2"] for r in summary["runs"]],
            color=color,
            s=20,
            alpha=0.45,
        )
    ax.text(6.85, 0.865, "Better on both tasks", color="#287152", fontsize=10)
    ax.set(
        xlabel="Position MAE (cm) — lower is better",
        ylabel="Cross-animal consistency R² — higher is better",
        xlim=(6.7, 10.5),
        ylim=(0.46, 0.90),
    )
    ax.set_title(
        "Structural alternatives to MARBLE: first exploratory run",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(alpha=0.13)
    ax.legend(loc="lower left", frameon=False)
    fig.supxlabel(
        "Bars: SD across 3 landmark seeds, not confidence intervals.\n"
        "32D decoding and 3D consistency are separate fits; "
        "GW uses pretrained MARBLE and an unlabeled reference.",
        fontsize=9,
        color="#4A5660",
    )
    fig.savefig(output / "comparison.png", dpi=190)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    table = [
        {
            "method": s["method"],
            "mae_cm": s["candidate_mae_cm_mean"],
            "mae_seed_sd": s["candidate_mae_cm_seed_sd"],
            "relative_mae_reduction": 1 - s["candidate_mae_cm_mean"] / base_mae,
            "consistency_r2": s["candidate_consistency_mean"],
            "consistency_seed_sd": s["candidate_consistency_seed_sd"],
            "consistency_change": s["candidate_consistency_mean"] - base_r2,
        }
        for s in summaries
    ]
    (output / "comparison.json").write_text(
        json.dumps(
            {
                "baseline_mae_cm": base_mae,
                "baseline_consistency": base_r2,
                "methods": table,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert np.isfinite([r["mae_cm"] for r in table]).all()
    logger.info("Comparison report saved to %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for method in METHODS:
        parser.add_argument("--" + method, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report([getattr(args, method) for method in METHODS], args.output)
