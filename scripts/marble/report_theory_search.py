"""Render all candidates, including ineligible candidates, without reranking."""

import argparse
import json
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

logger = logging.getLogger(__name__)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def report(source, output):
    protocol, result = read(source / "protocol.json"), read(source / "search.json")
    assert "lags" in protocol and "condition_budgets" in protocol
    assert "candidates" in result and "selected" in result and "seconds" in result
    lags, budgets = protocol["lags"], protocol["condition_budgets"]
    passed, ratios = (
        np.zeros((len(lags), len(budgets))),
        np.zeros((len(lags), len(budgets))),
    )
    lines = [
        "| lag | 条件数预算 | 证书通过数 / 30 | 验证误差 / persistence | 可选 |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["candidates"]:
        assert all(
            k in row
            for k in (
                "lag",
                "condition_budget",
                "certificates_passed",
                "validation_mean_error_ratio",
                "eligible",
            )
        )
        i, j = lags.index(row["lag"]), budgets.index(row["condition_budget"])
        passed[i, j] = row["certificates_passed"]
        ratios[i, j] = row["validation_mean_error_ratio"]
        eligible = "是" if row["eligible"] else "否"
        lines.append(
            f"| {lags[i]} | {budgets[j]} | {int(passed[i, j])} "
            f"| {ratios[i, j]:.5f} | {eligible} |"
        )
    output.mkdir(parents=True, exist_ok=False)
    for name in ("protocol.json", "search.json"):
        shutil.copyfile(source / name, output / name)
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.3), layout="constrained")
    first = axes[0].imshow(passed, vmin=0, vmax=30, cmap="YlGn")
    assert ratios.min() > 0
    second = axes[1].imshow(
        ratios, norm=LogNorm(vmin=ratios.min(), vmax=ratios.max()), cmap="YlOrRd"
    )
    axes[0].set_title("Stability certificates passed / 30")
    axes[1].set_title("Neural forecast MSE / persistence MSE")
    for index, ax in enumerate(axes):
        ax.set_xticks(range(len(budgets)), [str(v) for v in budgets])
        ax.set_yticks(range(len(lags)), [str(v) for v in lags])
        ax.set_xlabel("Regularized covariance condition budget")
        ax.set_ylabel("Prediction lag (samples)")
        for i in range(len(lags)):
            for j in range(len(budgets)):
                value = f"{passed[i, j]:.0f}" if index == 0 else f"{ratios[i, j]:.3g}"
                ax.text(
                    j,
                    i,
                    value,
                    ha="center",
                    va="center",
                    color="#122631",
                    fontsize=10,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.65,
                        "edgecolor": "none",
                        "pad": 2,
                    },
                )
    fig.colorbar(first, ax=axes[0], shrink=0.7)
    fig.colorbar(second, ax=axes[1], shrink=0.7)
    fig.suptitle(
        "Theory-constrained RRR search: all 25 candidates",
        fontsize=15,
        fontweight="bold",
    )
    fig.supxlabel(
        "Selection requires 30/30 certificates. Forecast ratios average 15 tasks; "
        "lower is better.\nDifferent lags predict different targets. "
        "No position labels or new behavioral benchmark.",
        fontsize=9,
    )
    fig.savefig(output / "search.png", dpi=180)
    fig.savefig(output / "search.pdf")
    plt.close(fig)
    logger.info("Saved complete search report to %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(args.input, args.output)
