# MARBLE 联合投影：四动物扩展结果

**固定 α=0.1 没有在新增三只动物上复现 Achilles 的平均 MAE 收益；
四动物表示一致性的平均 R² 也下降。本轮不支持“这种投影普遍优于 PCA”。**

2026-09-29 至 09-30 完成新增 42 次 CUDA 13 训练，每次 100 epochs。
其中 18 次用于新增三只动物的位置解码，24 次用于四只动物的 3D 模型从头训练。
Achilles 的 32D 解码复用上一轮六次训练，单列为发现 α 的探索结果。
未根据新增动物重新选择 α、维数、训练轮数或最好的种子。

## 位置解码：收益未推广

固定 20 维状态投影、32 维 MARBLE 表示；前 80%／后 20% 时间划分；
投影只在训练段拟合。主指标冻结 BN，MAE 越低越好，± 为三种子样本 SD。

| 动物 | 精确 PCA，MAE/cm | 联合投影，MAE/cm | 相对变化 |
|---|---:|---:|---:|
| Achilles（原探索结果） | 9.559 ± 0.514 | 9.142 ± 0.609 | −4.36% |
| Buddy（新增） | 9.685 ± 0.342 | 9.797 ± 0.307 | +1.16% |
| Cicero（新增） | 12.132 ± 1.602 | 13.218 ± 2.451 | +8.96% |
| Gatsby（新增） | 7.818 ± 0.098 | 7.920 ± 0.263 | +1.30% |

新增动物的等权宏平均 MAE 从 **9.878 升至 10.312 cm**。
0/3 只动物的三种子均值改善，但 4/9 个配对种子改善；不能只选最好的一次。
Gatsby 的位置 R² 有改善、MAE 却上升，Notebook BN 的结果也有不同；
这些次要指标均保留在完整报告，主结论仍依据预定 MAE。

## 跨个体一致性：平均分数下降

这部分使用独立的 10 维状态投影、3 维输出、dropout=0.5 配置。
全部四只动物都从头训练，未使用作者训练权重。位置标签只参与评估对齐。

| 评估范围 | 精确 PCA，R² | 联合投影，R² |
|---|---:|---:|
| 四动物全部 12 个有向对（主指标） | 0.721645 ± 0.073366 | 0.651711 ± 0.072389 |
| 不含 Achilles 的 6 个有向对（补充诊断） | 0.721102 ± 0.005497 | 0.621633 ± 0.118393 |

主指标的配对种子变化为 −0.1798、+0.0937、−0.1237，存在明显训练波动；
12 个方向中，11 个方向的三种子平均 R² 下降。上述方向共享动物，不能当成
独立生物样本。这是位置分箱后在相同分箱点拟合与评分的一致性，不能解释为
跨动物零样本解码或留出集泛化。补充诊断在第一套一致性模型训练前登记。

## 数学性质与下游性能要分开

七个新拟合的正权重投影都满足预期：训练状态残差略增，训练增量残差下降。
因此本轮验证了投影目标的实现，也给出了“保留更多增量能量却不改善下游主指标”
的实际例子。全局最优子空间是相对于该投影目标而言，不等于最优行为表示。
这不排除其他动力学投影的价值，也不证明该策略在所有数据上都更差。

![全部实验概览](reports/marble-dynamics-multirat-20260929/overview.png)

## 复核、材料与范围

- 14 个条件、42 个新模型均完成训练及原版 CPU checkpoint 核对，最大表征差
  4.77×10⁻⁷；全部 CEBRA 一致性分数独立复核通过，最大 R² 差为 9.30×10⁻⁸。
- kNN 解码有 3 个样本存在已证明的边界等距邻居选择差异（2 处 Notebook、
  1 处主评估）。单个种子 MAE 的影响上界均小于 0.0004 cm，保留完整差异记录。
- 新增压缩存储、配对检查、完整种子汇总及双精度校验回归测试；全套 **321 项通过**，
  15 个变更 Python 文件的 Ruff 检查与格式检查通过。
- 3D 首次训练前遇到单精度校验边界误差。随后以原版双精度为基准，原版单精度与
  CUDA 单精度分别按原容差核验；正式训练仍为单精度，未改变超参数。失败日志、
  诊断、旧源码及修订记录都保留。未把重复尝试计入 42 次训练。
- Buddy 实际有 6,577 个时间点，其余各 10,000 个。首轮样本数预检在训练前纠正。
  C 盘空间不足后迁到 D 盘，逐文件验证哈希并保留工作区链接，已完成训练未重跑。

[完整逐种子、逐方向结果](reports/marble-dynamics-multirat-20260929/RESULTS.md)、
[机器可读指标](reports/marble-dynamics-multirat-20260929/aggregate.json)、
[执行前方案与修订说明](MARBLE_MULTIRAT_PROTOCOL.md)、
[源码快照](reports/marble-dynamics-multirat-20260929/source.zip)。

本轮覆盖公开四大鼠数据的两套协议，采用发布 checkpoint／notebook 衍生设置。
论文 PCA=5 等参数、任务 RNN、猕猴及其他实验仍未在这套新投影上完成验证，
不能称为整篇论文的完整复现。

## 复跑入口

在 KAIR 根目录执行，输出目录必须不存在；图和采样使用同级 MARBLE 的固定 CPU 环境。
汇总默认复用 `results/dynamics-projection-20260929` 中的 Achilles 探索结果，
可通过 `--discovery` 指定其位置。

```powershell
$env:MPLBACKEND = "Agg"
uv run --locked python -m scripts.marble.run_multirat_projection --output results/multirat-new
uv run --locked python -m scripts.marble.score_multirat_projection --root results/multirat-new
../MARBLE/reproduction/.venv/Scripts/python.exe -m scripts.marble.score_multirat_projection --root results/multirat-new --audit-cebra
```

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-29 至 2026-09-30
- Verification Status: ANALYZED（42 次训练完成、原版算子与指标核对通过；未独立重复整套实验）
- Version Label: dynamics_multirat_result_v1
- 解释范围：种子间 SD 为描述统计；新增动物未选 α；全部结果保留。
  不将种子或共享动物的方向对当作独立生物样本，不作显著性或生理因果推断。
