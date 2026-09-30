"""Write a complete alpha-grid report and scientific figures after metric audit."""

import argparse
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
from scripts.marble.run_alpha_grid import RATS, ROOT, SEEDS
from utils.utils_marble import read_json, save_json, sha256


def plot(result, destination):
    alphas = [c["alpha"] for c in result["decoding"]["common_alpha"]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    positions = np.arange(len(alphas))
    for ax, animal in zip(axes.flat, RATS, strict=False):
        cells = [
            c
            for c in result["decoding"]["cells"]
            if c["animal"] == animal and c["mode"] == "eval"
        ]
        assert [c["alpha"] for c in cells] == list(alphas)
        values = np.array([c["seed_mae_cm"] for c in cells])
        ax.errorbar(
            positions,
            values.mean(axis=1),
            yerr=values.std(axis=1, ddof=1),
            marker="o",
            color="#087f8c",
            capsize=3,
            label="Mean +/- seed SD",
        )
        for seed in SEEDS:
            ax.scatter(positions, values[:, seed], s=17, alpha=0.5, color="#087f8c")
        ax.axhline(values[0].mean(), color="#899499", linestyle="--", label="Exact PCA")
        ax.set_title(animal.title())
        ax.set_ylabel("Position MAE (cm); lower is better")
    common = result["decoding"]["common_alpha"]
    ax = axes[1, 1]
    ax.errorbar(
        positions,
        [c["macro_mean_cm"] for c in common],
        yerr=[c["seed_macro_sd_cm"] for c in common],
        marker="o",
        color="#087f8c",
        capsize=3,
    )
    ax.axhline(common[0]["macro_mean_cm"], color="#899499", linestyle="--")
    ax.set_title("Four-animal decoding mean")
    ax.set_ylabel("Equal-animal MAE (cm); lower is better")
    ax = axes[1, 2]
    values = np.array([c["seed_mean_r2"] for c in result["consistency_summary"]])
    ax.errorbar(
        positions,
        values.mean(axis=1),
        yerr=values.std(axis=1, ddof=1),
        marker="o",
        color="#d47628",
        capsize=3,
    )
    for seed in SEEDS:
        ax.scatter(positions, values[:, seed], s=17, alpha=0.5, color="#d47628")
    ax.axhline(values[0].mean(), color="#899499", linestyle="--")
    ax.set_title("Cross-animal consistency (12 directions)")
    ax.set_ylabel("Binned affine R2; higher is better")
    for ax in axes.flat:
        ax.set_xticks(positions, [f"{a:g}" for a in alphas])
        ax.set_xlabel("Projection alpha (categorical spacing)")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "MARBLE alpha grid: all points, four rats, three seeds\n"
        "Exploratory tuning on previously examined data; bars are seed SD",
        fontsize=14,
    )
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.savefig(destination / "alpha_grid.png", dpi=170)
    fig.savefig(destination / "alpha_grid.pdf")
    plt.close(fig)


def report(root, destination):
    assert not destination.exists(), "Never overwrite a report"
    result = read_json(root / "aggregate.json")
    frozen = read_json(root / "frozen_protocol.json")
    alphas = frozen["alphas"]
    assert [c["alpha"] for c in result["decoding"]["common_alpha"]] == alphas
    assert [c["alpha"] for c in result["consistency_summary"]] == alphas
    completed = read_json(root / "completed.json")
    audit = read_json(root / "cebra_metric_audit.json")
    assert audit["aggregate_sha256"] == sha256(root / "aggregate.json")
    assert len(audit["results"]) == len(alphas) * len(SEEDS)
    destination.mkdir(parents=True)
    for name in (
        "aggregate.json",
        "cebra_metric_audit.json",
        "completed.json",
        "frozen_protocol.json",
        "execution_source.zip",
    ):
        shutil.copyfile(root / name, destination / name)
    for name in (
        "execution_amendment.json",
        "execution_source_amended.zip",
        "precision-diagnostic.json",
    ):
        if (root / name).is_file():
            shutil.copyfile(root / name, destination / name)
    assert len(result["numerics"]) == len(alphas) * len(RATS)
    shutil.copyfile(
        ROOT / "docs/MARBLE_ALPHA_GRID_NUMERICS.md", destination / "NUMERICS.md"
    )
    audits = []
    for job in frozen["jobs"]:
        folder = Path(job["folder"])
        audits.append(
            {
                **job,
                "original_code": read_json(folder / "run/original_code_audit.json"),
                "reference_validation": read_json(
                    folder / "run/reference_validation.json"
                ),
                "provenance": read_json(folder / "run/provenance.json"),
            }
        )
    save_json(destination / "audits.json", audits)
    with ZipFile(destination / "analysis_source.zip", "x", ZIP_DEFLATED) as archive:
        for name in (
            "scripts/marble/score_alpha_grid.py",
            "scripts/marble/report_alpha_grid.py",
            "utils/utils_marble_consistency.py",
            "tests/test_marble_alpha_grid.py",
            "docs/MARBLE_ALPHA_GRID_PROTOCOL.md",
            "docs/MARBLE_ALPHA_FINE_GRID_PROTOCOL.md",
            "tests/test_marble_alpha_fine_grid.py",
        ):
            archive.write(ROOT / name, name)
    plot(result, destination)
    decoding = result["decoding"]
    best = decoding["best_common"]
    baseline = decoding["common_alpha"][0]["macro_mean_cm"]
    best_c = result["best_consistency"]
    baseline_c = result["consistency_summary"][0]["mean_r2"]
    change = 100 * (best["macro_mean_cm"] / baseline - 1)
    lines = [
        "# MARBLE α 网格搜索结果",
        "",
        "这是一轮探索性调参：四只大鼠的评估数据此前已查看，最优分数不能视为独立泛化证据。",
        "",
        "α 是本项目“状态＋增量联合投影”的权重：α=0 对应精确 PCA，α 越大越强调保留神经状态增量。"
        "其余设置固定。解码使用 20 维投影、32 维输出；一致性使用 10 维投影、3 维输出。",
        "",
        f"位置解码的共同最优 α={best['alpha']:g}，四动物平均 MAE 为 **{best['macro_mean_cm']:.4f} cm**，"
        f"PCA 为 {baseline:.4f} cm（变化 {change:+.2f}%；{best['animals_improved']}/4 动物均值改善，"
        f"{best['paired_seeds_improved']}/12 个配对种子改善）。",
        f"跨个体一致性的最优 α={best_c['alpha']:g}，平均 R² 为 **{best_c['mean_r2']:.4f}**，"
        f"PCA 为 {baseline_c:.4f}（差值 {best_c['mean_r2'] - baseline_c:+.4f}）。",
        "",
        f"{completed['jobs']} 组条件、{completed['jobs'] * len(SEEDS)} 个训练种子："
        f"复用 {completed['reuse_training_seeds']} 个种子，新增 {completed['new_training_seeds']} 个种子，"
        "每个种子训练 100 epochs。固定维数、网络及其他超参数，只改变联合投影 α。",
        "",
        "![All alpha curves](alpha_grid.png)",
        "",
        "## 所有网格点",
        "",
        "MAE 单位 cm，↓更好；一致性 R²，↑更好。动物单元格为三种子均值±样本标准差。",
        "",
        "| α | Achilles | Buddy | Cicero | Gatsby | 四动物 MAE 均值 | 一致性 R² |",
        "|---|---|---|---|---|---|---|",
    ]
    for alpha in alphas:
        cells = [
            c for c in decoding["cells"] if c["alpha"] == alpha and c["mode"] == "eval"
        ]
        common = next(c for c in decoding["common_alpha"] if c["alpha"] == alpha)
        consistency = next(
            c for c in result["consistency_summary"] if c["alpha"] == alpha
        )
        text = " | ".join(f"{c['mean_cm']:.3f} ± {c['sd_cm']:.3f}" for c in cells)
        lines.append(
            f"| {alpha:g} | {text} | {common['macro_mean_cm']:.3f} | "
            f"{consistency['mean_r2']:.4f} ± {consistency['sd_r2']:.4f} |"
        )
    lines += [
        "",
        "两个最优参数分别按各自的预先定义指标选择；不在事后把两个指标合成新的分数。"
        "逐只动物选择 α 的最小值，仅反映在这些数据上的调参空间。",
        "",
        "## 每只动物的最优值与留一动物选择诊断",
        "",
        "| 动物 | 该动物自身最优 α | 自身最优 MAE | 其余三只选择的 α | 留出动物 MAE | PCA MAE |",
        "|---|---|---|---|---|---|",
    ]
    for cell, fold in zip(
        decoding["best_per_animal"], decoding["loao"]["folds"], strict=True
    ):
        lines.append(
            f"| {cell['animal']} | {cell['alpha']:g} | {cell['mean_cm']:.4f} | "
            f"{fold['selected_alpha']:g} | {fold['excluded_mean_cm']:.4f} | {fold['pca_mean_cm']:.4f} |"
        )
    loao = decoding["loao"]
    lines += [
        "",
        f"留一动物选择的平均 MAE：{loao['macro_mean_cm']:.4f} cm；"
        f"PCA：{loao['pca_macro_mean_cm']:.4f} cm。选择 α 时不使用被留出动物的网格分数，"
        "但这是已研究数据上的回顾性诊断，不是新增独立动物验证。",
        "",
        "## 每个种子的解码结果",
        "",
        "主指标使用 eval（冻结 BN）；notebook 模式为次要敏感性分析，不按结果切换主指标。",
        "",
        "| 动物 | α | 模式 | 种子 | MAE cm | 位置 R² | 最优 epoch |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result["decoding_rows"]:
        lines.append(
            f"| {r['animal']} | {r['alpha']:g} | {r['mode']} | {r['seed']} | "
            f"{r['mae_cm']:.5f} | {r['position_r2']:.5f} | {r['best_epoch']} |"
        )
    lines += [
        "",
        "## 每个种子的跨个体一致性",
        "",
        "| α | 种子 | 12 个方向平均 R² |",
        "|---|---|---|",
    ]
    for r in result["consistency"]:
        lines.append(f"| {r['alpha']:g} | {r['seed']} | {r['mean_r2']:.6f} |")
    lines += [
        "",
        "## 各方向的三种子均值",
        "",
        "| 来源 → 目标 | " + " | ".join(f"α={a:g}" for a in alphas) + " |",
        "|---|" + "---|" * len(alphas),
    ]
    for i, pair in enumerate(result["consistency_summary"][0]["pairs"]):
        values = [c["pair_seed_mean_r2"][i] for c in result["consistency_summary"]]
        lines.append(
            f"| {pair[0]} → {pair[1]} | "
            + " | ".join(f"{v:.5f}" for v in values)
            + " |"
        )
    maximum = max(r["max_r2_error"] for r in audit["results"])
    lines += [
        "",
        "## 核查与边界",
        "",
        "- 每组均完成原始 CPU 实现的数值核对和训练后模型审计；保存各组原始采样计划，"
        "初始权重和节点划分配对一致，邻接图及邻域随 α 重新计算。",
        "- 沿用已修订的一致性校验：同起点 float32 单步核对及完整 float64 轨迹核对；"
        f"全部 {len(result['numerics'])} 组通过。正式训练仍为 float32。"
        "不宣称完整 float32 轨迹一致，详见 [数值验证背景](NUMERICS.md)。",
        f"- {len(audit['results'])} 组一致性计算通过 CEBRA 0.4.0 / sklearn 1.3.2 独立核查，最大 R² 差异 {maximum:.3g}。",
        "- 一致性采用行为位置分箱后的同样本仿射回归 R²，不是零样本迁移或留出行为解码。",
        "- 表征训练不输入行为标签；本轮用行为解码及位置对齐指标选 α，因此参数选择使用了行为标签。",
        "- 种子和 12 个动物对不是独立生物学重复；这里报告描述统计，不声明显著性。",
        f"- α={max(alphas):g} 是本轮上边界；离散局部网格不证明全局最优。",
        "- 只覆盖公开四大鼠子集及 checkpoint/notebook 协议，不代表论文全部实验。",
        "- 原始代码审计中的浮点距离并列近邻差异若存在，完整保存在 audits.json；不能称逐位完全一致。",
        "",
        "Material Passport: academic-research-suite / experiment-agent; mode validate; "
        "origin 2026-09-30; status ANALYZED (all grid runs executed and audited, descriptive inference only).",
        "",
        f"Large artifacts: `{root}`. Execution and analysis source snapshots accompany this report.",
        "",
    ]
    (destination / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.root.resolve(), args.output.resolve())
