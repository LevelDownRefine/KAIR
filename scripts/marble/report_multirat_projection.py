"""Make a traceable report from all audited fixed-alpha multi-rat results."""

import argparse
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from utils.utils_marble import read_json, save_json, sha256

ROOT = Path(__file__).resolve().parents[2]
RATS = ("achilles", "buddy", "cicero", "gatsby")
CONDITIONS = ("pca", "joint-01")


def plot(results, destination):
    rows = [r for r in results["decoding_summary"] if r["mode"] == "eval"]
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), layout="constrained")
    colors = ("#677787", "#008D9D")
    positions = np.arange(len(rows))
    for index, prefix in enumerate(("pca", "joint")):
        axes[0].bar(
            positions + (index - 0.5) * 0.32,
            [r[f"{prefix}_mean_cm"] for r in rows],
            yerr=[r[f"{prefix}_sd_cm"] for r in rows],
            width=0.32,
            color=colors[index],
            capsize=3,
            label="Exact PCA" if index == 0 else r"Joint projection, $\alpha=0.1$",
        )
    axes[0].set_xticks(positions, ["Achilles*", "Buddy", "Cicero", "Gatsby"])
    axes[0].set_ylabel("Test position MAE (cm), lower is better")
    axes[0].set_title(
        "State projection 20D / embedding 32D\n*Achilles: reused discovery result",
        fontsize=11,
    )
    axes[0].legend(fontsize=8)
    for index, condition in enumerate(CONDITIONS):
        values = results["consistency_summary"][condition]["seed_mean_r2"]
        axes[1].bar(
            index,
            np.mean(values),
            yerr=np.std(values, ddof=1),
            color=colors[index],
            width=0.45,
            capsize=4,
        )
        axes[1].scatter(np.full(3, index), values, color="#17313E", s=18, zorder=3)
    for seed in range(3):
        axes[1].plot(
            [0, 1],
            [
                results["consistency_summary"][c]["seed_mean_r2"][seed]
                for c in CONDITIONS
            ],
            color="#17313E",
            alpha=0.35,
            linewidth=0.8,
        )
    axes[1].set_xticks([0, 1], ["Exact PCA", r"Joint $\alpha=0.1$"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean directed-pair consistency R²")
    axes[1].set_title(
        "State projection 10D / embedding 3D\n4 fresh-trained rats; in-sample alignment",
        fontsize=11,
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    figure.savefig(destination / "overview.png", dpi=180)
    figure.savefig(destination / "overview.pdf")
    plt.close(figure)


def report(root, destination):
    results = read_json(root / "aggregate.json")
    audit = read_json(root / "cebra_metric_audit.json")
    assert audit["aggregate_sha256"] == sha256(root / "aggregate.json")
    destination.mkdir(parents=True, exist_ok=False)
    display_root = (
        root.relative_to(ROOT).as_posix() if root.is_relative_to(ROOT) else str(root)
    )
    for name in (
        "aggregate.json",
        "cebra_metric_audit.json",
        "frozen_protocol.json",
        "completed.json",
    ):
        shutil.copy2(root / name, destination / name)
    if (root / "storage_transition.json").is_file():
        shutil.copy2(
            root / "storage_transition.json", destination / "storage_transition.json"
        )
    if (root / "precision_amendment.json").is_file():
        shutil.copy2(
            root / "precision_amendment.json", destination / "precision_amendment.json"
        )
        for name in ("precision-diagnostic.json", "precision-cuda-ten-checks.json"):
            shutil.copy2(root / name, destination / name)
    audited = {}
    for protocol in ("decoding", "consistency"):
        for animal in RATS[1:] if protocol == "decoding" else RATS:
            for condition in CONDITIONS:
                folder = root / protocol / animal / condition
                key = f"{protocol}/{animal}/{condition}"
                audited[key] = {
                    "original_code": read_json(folder / "run/original_code_audit.json"),
                    "reference_steps": read_json(
                        folder / "run/reference_validation.json"
                    ),
                    "projection": read_json(
                        folder / "input/projection_diagnostics.json"
                    ),
                    "protocol": read_json(folder / "input/protocol.json"),
                    "run_provenance": read_json(folder / "run/provenance.json"),
                    "input_provenance": read_json(folder / "input/provenance.json"),
                    "initialization": {
                        str(seed): read_json(
                            folder / f"input/seed-{seed}/initialization.json"
                        )
                        for seed in (0, 1, 2)
                    },
                    "metrics": {
                        str(seed): read_json(folder / f"run/seed-{seed}/metrics.json")
                        for seed in (0, 1, 2)
                    },
                }
                if protocol == "consistency":
                    audited[key]["precision_reference"] = read_json(
                        folder / "input/precision_reference_receipt.json"
                    )
    save_json(destination / "audits.json", audited)
    plot(results, destination)
    holdout = results["new_animals_eval"]
    lines = [
        "# MARBLE 联合投影：新增动物与跨个体验证",
        "",
        "材料性质：固定参数的探索性后续实验；数据为公开大鼠记录；所有结果保留。",
        "这份报告验证新增动物和跨个体协议，尚未覆盖论文的 RNN、猕猴等实验。",
        "采用发布 checkpoint／notebook 衍生协议；论文 PCA=5 等参数设置另列实验，不能用本结果替代。",
        "",
        "## 预先固定的比较",
        "",
        "- Exact PCA（α=0）与状态–增量联合投影（α=0.1）；α 来自 Achilles 探索，新增动物上不调参。",
        "- 每条件 3 个种子 × 100 epochs；同种子初始权重、节点划分一致；按无监督验证损失选 checkpoint。",
        "- 解码：20 维输入投影、32 维输出，时间上前 80% 训练／后 20% 测试。测试图使用完整测试段，属于离线解码。",
        "- 一致性：10 维输入投影、3 维输出、dropout=0.5，四只动物分别使用全部公开记录从头训练。",
        "- Buddy：6,577×48；Achilles：10,000×120；Cicero：10,000×55；Gatsby：10,000×66。行是时间点，列是神经元。",
        "- 原版 CPU 构图及采样；PyTorch CUDA 13 训练；保持作者平滑、非零 bin 事件转换等预处理约定。",
        "",
        "## 1. 单动物位置解码",
        "",
        "主指标：冻结 BN 的测试位置 MAE（cm，越低越好）。均值±样本 SD 来自三个训练种子。",
        "",
        "| 动物 | PCA | 联合投影 | MAE 相对变化 | 改善种子 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results["decoding_summary"]:
        if row["mode"] != "eval":
            continue
        name = row["animal"].capitalize() + (
            "（探索，复用）" if row["discovery"] else "（新增）"
        )
        lines.append(
            f"| {name} | {row['pca_mean_cm']:.3f}±{row['pca_sd_cm']:.3f} | "
            f"{row['joint_mean_cm']:.3f}±{row['joint_sd_cm']:.3f} | "
            f"{row['relative_change_percent']:+.2f}% | {row['seeds_improved']}/3 |"
        )
    lines += [
        "",
        f"仅看新增三只动物：{holdout['animals_improved']}/3 只平均 MAE 改善，"
        f"{holdout['paired_seeds_improved']}/9 个配对种子改善。等权宏平均 MAE："
        f"{holdout['pca_macro_mean_cm']:.3f} → {holdout['joint_macro_mean_cm']:.3f} cm。",
        "",
        "Achilles 用于提出 α 候选，不能再作为独立验证。不得只选最好种子或最好动物。",
        "",
        "### 配对种子明细（联合 − PCA，cm；负值为改善）",
        "",
        "| 动物 | seed 0 | seed 1 | seed 2 | PCA R²均值 | 联合 R²均值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results["decoding_summary"]:
        if row["mode"] == "eval":
            differences = " | ".join(f"{d:+.4f}" for d in row["paired_seed_deltas_cm"])
            lines.append(
                f"| {row['animal']} | {differences} | {row['pca_mean_r2']:.5f} | {row['joint_mean_r2']:.5f} |"
            )
    lines += [
        "",
        "### Notebook BN 敏感性（次要指标）",
        "",
        "| 动物 | PCA MAE/cm | 联合 MAE/cm | 相对变化 |",
        "|---|---:|---:|---:|",
    ]
    for row in results["decoding_summary"]:
        if row["mode"] == "notebook":
            lines.append(
                f"| {row['animal']} | {row['pca_mean_cm']:.3f}±{row['pca_sd_cm']:.3f} | "
                f"{row['joint_mean_cm']:.3f}±{row['joint_sd_cm']:.3f} | {row['relative_change_percent']:+.2f}% |"
            )
    lines += [
        "",
        "## 2. 跨个体表示一致性",
        "",
        "每个种子包含 12 个有方向的动物对。先按位置分箱、归一化平均表示，再做带截距线性回归；"
        "在相同分箱点拟合并评分。位置标签仅用于评估对齐。该 R² 不是跨个体零样本解码，也不是行为留出集性能。",
        "",
        "| 条件 | seed 0 | seed 1 | seed 2 | 均值±SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        result = results["consistency_summary"][condition]
        values = " | ".join(f"{v:.6f}" for v in result["seed_mean_r2"])
        lines.append(
            f"| {condition} | {values} | {result['mean_r2']:.6f}±{result['sd_r2']:.6f} |"
        )
    lines += [
        "",
        "补充诊断：仅看 Buddy、Cicero、Gatsby 之间的 6 个方向（保留四动物统一分箱）。"
        "该诊断在第一套一致性模型训练前加入；全部 12 个方向仍是预定主指标。",
        "",
    ]
    for condition in CONDITIONS:
        result = results["new_animal_pair_consistency_summary"][condition]
        lines.append(
            f"- {condition}：R²={result['mean_r2']:.6f}±{result['sd_r2']:.6f}。"
        )
    lines += [
        "",
        "以下每格是三个种子的均值；完整逐对、逐种子数值在 `aggregate.json`。",
        "",
        "| 方向 | PCA R² | 联合 R² | 差值 |",
        "|---|---:|---:|---:|",
    ]
    entries = results["consistency"]
    for index, pair in enumerate(entries[0]["pairs"]):
        means = [
            np.mean([e["scores"][index] for e in entries if e["condition"] == c])
            for c in CONDITIONS
        ]
        lines.append(
            f"| {' → '.join(pair)} | {means[0]:.6f} | {means[1]:.6f} | {means[1] - means[0]:+.6f} |"
        )
    lines += [
        "",
        "## 3. 投影目标检查",
        "",
        "以下为拟合段残差能量占比；两种残差分别除以状态能量和增量能量。"
        "每行只拟合一次投影，不因三个网络种子而重复计算样本量。",
        "",
        "| 协议／动物 | PCA 状态残差 | 联合状态残差 | PCA 增量残差 | 联合增量残差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for protocol in ("decoding", "consistency"):
        for animal in RATS[1:] if protocol == "decoding" else RATS:
            projection = audited[f"{protocol}/{animal}/joint-01"]["projection"]
            reference, joint = projection["pca_train"], projection["train"]
            lines.append(
                f"| {protocol}/{animal} | {100 * reference['state_residual_fraction']:.4f}% | "
                f"{100 * joint['state_residual_fraction']:.4f}% | "
                f"{100 * reference['increment_residual_fraction']:.4f}% | "
                f"{100 * joint['increment_residual_fraction']:.4f}% |"
            )
    lines += [
        "",
        "## 4. 实现核对与结论边界",
        "",
        "- 新增 42 次训练均完成 100 epochs；14 个条件均保存原版三步 SGD 对照及原版 CPU checkpoint 复核。",
        "- 3D 训练前校验采用原版双精度基准：原版单精度与 CUDA 单精度分别按原容差核对；正式训练仍为单精度。",
        "- 解码器对照原版 CEBRA KNNDecoder；如遇边界等距邻居，必须通过独立等距证明，不能放宽整体误差容限。",
        "- 全部跨个体 R² 使用固定版本 CEBRA 0.4.0、scikit-learn 1.3.2 独立复核。",
        "- 三个种子衡量优化波动；12 个有向动物对共享四只动物，不能当作 12 个独立生物样本。",
        "- 增量投影残差降低是该目标的数学性质，不保证行为 MAE 或跨个体一致性改善。",
        "- 跨个体实验均为从头训练；此前作者 checkpoint 的成绩不混入此处。",
        "",
        "![全部结果](overview.png)",
        "",
        "## 可复核材料",
        "",
        "`aggregate.json`：全部指标；`audits.json`：14 条件的实现核对和来源哈希；"
        "`cebra_metric_audit.json`：独立一致性度量审计；`frozen_protocol.json`：训练前固定方案；"
        "`source.zip`：执行代码快照。原始权重、图、采样和逐 epoch 损失保留在本地结果目录。",
        "",
        f"结果目录：`{display_root}`。",
        "",
    ]
    (destination / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    sources = [
        *sorted((ROOT / "scripts/marble").glob("*.py")),
        ROOT / "utils/utils_dynamics_projection.py",
        ROOT / "utils/utils_marble.py",
        ROOT / "utils/utils_marble_consistency.py",
        ROOT / "utils/utils_marble_storage.py",
        ROOT / "models/network_marble.py",
        ROOT / "models/model_marble.py",
        ROOT / "data/dataset_marble.py",
        ROOT / "main_train_marble.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "docs/MARBLE_MULTIRAT_PROTOCOL.md",
        ROOT / "tests/test_marble_multirat.py",
        ROOT / "tests/test_marble_storage.py",
        ROOT / "tests/test_dynamics_projection.py",
        ROOT / "tests/test_marble_precision.py",
        ROOT / "tests/fixtures/marble_float64_reference.pt.gz",
        ROOT / "tests/fixtures/marble_float64_reference.pt.json",
        ROOT / "licenses/MARBLE.txt",
        ROOT / "licenses/CEBRA.txt",
    ]
    with zipfile.ZipFile(
        destination / "source.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for source in sources:
            archive.write(source, source.relative_to(ROOT).as_posix())
        for source in sorted((root / "precision-gate-before").glob("*.py")):
            archive.write(source, f"prior_precision_gate/{source.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.root.resolve(), args.output.resolve())
