# MARBLE α=0.01–0.05 局部网格结果

补完 **0.02、0.04、0.05**：四只大鼠、两套协议、三个种子，每次 100 epochs，
共新增 **72 次 CUDA 13 float32 训练**。复用 0、0.01、0.03 的 72 次已审计训练，
本轮共比较 48 组条件 / 144 个模型。降维维数、网络和其他训练参数保持固定。

| α | 四动物平均位置 MAE / cm ↓ | 跨个体一致性 R² ↑ |
|---|---:|---:|
| 0（精确 PCA） | 9.798 | 0.7216 |
| 0.01 | 9.616 | 0.6250 |
| 0.02（新增） | 9.717 | 0.6894 |
| 0.03 | **9.505** | 0.7008 |
| 0.04（新增） | 9.567 | **0.7378** |
| 0.05（新增） | 9.604 | 0.6941 |

**位置解码优先：0.03 仍最好。** MAE 相对 PCA 降低 3.00%，4/4 动物均值、
8/12 配对种子改善。四个回顾性留一动物选参折仍全部选到 0.03。

**两项均值都看：0.04 值得保留。** MAE 相对 PCA 降低 2.36%，一致性 R² 增加
0.01616。解码在 3/4 动物均值、9/12 配对种子上改善；一致性却仅 **1/3 配对种子**改善，
三个种子的 R² 差值分别为 **−0.03720、+0.16105、−0.07538**。
因此这是总体均值上的双提升，尚未证明稳定的双提升。

0.02、0.05 都未超过上述两个候选。不同动物的最优 α 并不相同：Achilles/Cicero 为
0.03，Buddy 为 0.01，Gatsby 为 0.04。共同参数按四动物等权均值选择，不拼接各动物最优值。

解码采用 20 维投影 / 32 维输出；一致性采用 10 维投影 / 3 维输出，是两套独立训练协议。
一致性是按位置分箱后的同样本仿射回归 R²，不是留出个体迁移准确率。
这是已研究数据上的适应性选参，没有新增独立动物；报告描述统计，不声明显著性。

- [完整结果、各种子与各方向](reports/marble-alpha-fine-grid-20260930/RESULTS.md)
- [曲线 PNG](reports/marble-alpha-fine-grid-20260930/alpha_grid.png) · [矢量 PDF](reports/marble-alpha-fine-grid-20260930/alpha_grid.pdf)
- [配对种子与统计解释](reports/marble-alpha-fine-grid-20260930/PAIRED_AND_REVIEW.md)
- [冻结方案](MARBLE_ALPHA_FINE_GRID_PROTOCOL.md) · [机器可读指标](reports/marble-alpha-fine-grid-20260930/aggregate.json)
- [验证记录](reports/marble-alpha-fine-grid-20260930/validation_summary.json) · [旧网格全部结果](MARBLE_ALPHA_GRID_RESULTS.md)

## 核查与存储

48 组新旧模型审计记录完整；24 组一致性条件通过固定阈值的 float32 单步 / float64
连续轨迹校验，正式模型均为 float32。18 组指标通过 CEBRA 0.4.0 / sklearn 1.3.2
独立核查，最大 R² 差异为 **1.66×10⁻⁷**。342 项测试通过（53 条既有依赖警告），
四个变更 Python 文件通过 Ruff 与格式检查。图表已目视检查，无遮挡或裁切。

只调整过输入预备的 CPU 调度：提前并行准备 0.05 和 0.04，正式 GPU 训练仍按原顺序
单进程运行。冻结的执行源码、训练设置、种子与数值阈值均未修改；调度记录及源码另行归档。

按用户要求，旧 α=0.1、0.3、1 及早期 α=10 的 378 个大型派生文件已清理，释放
**11.603 GiB**；保留 967 个指标、日志和审计文件，原始数据及本轮全部依赖保留。
历史分数仍参与可追溯比较；被删模型若需重新核验须重建。
[完整清理说明和文件哈希](MARBLE_ARTIFACT_RETENTION.md)。

大型本轮产物：`D:/MARBLE-experiments/KAIR/alpha-fine-grid-20260930`。

## 重跑入口

在已配置的 KAIR uv 环境中，选择一个不存在的新输出目录。旧网格中保留的
0/0.01/0.03 可继续复用；细网格入口不会读取已清理的大 α 张量文件。

```powershell
.venv/Scripts/python.exe -m scripts.marble.run_alpha_grid --output D:/MARBLE-experiments/KAIR/alpha-fine-grid-rerun --previous-grid D:/MARBLE-experiments/KAIR/alpha-grid-20260930
.venv/Scripts/python.exe -m scripts.marble.score_alpha_grid --root D:/MARBLE-experiments/KAIR/alpha-fine-grid-rerun
../MARBLE/reproduction/.venv/Scripts/python.exe -m scripts.marble.score_alpha_grid --root D:/MARBLE-experiments/KAIR/alpha-fine-grid-rerun --audit-cebra
.venv/Scripts/python.exe -m scripts.marble.report_alpha_grid --root D:/MARBLE-experiments/KAIR/alpha-fine-grid-rerun --output docs/reports/marble-alpha-fine-grid-rerun
```

Material Passport: academic-research-suite / experiment-agent; mode run/validate;
origin 2026-09-30; status ANALYZED (execution/audits complete; exploratory inference).
