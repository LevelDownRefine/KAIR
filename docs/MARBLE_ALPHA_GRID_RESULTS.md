# MARBLE 联合投影 α 网格结果

存储更新（2026-09-30）：按用户要求，α=0.1、0.3、1 的大型派生数据与模型文件已清理，
指标、日志及本文完整结果保留。下方审计描述的是清理前的核查；这些条件须重新生成后才能
再次审计或复用。[清理范围与逐文件记录](MARBLE_ARTIFACT_RETENTION.md)。

后续已完成 [0.01–0.05 局部细搜](MARBLE_ALPHA_FINE_GRID_RESULTS.md)：共同解码最优仍为 0.03，
一致性均值最优改为 0.04；0.04 两项均值均优于 PCA，但一致性仅 1/3 配对种子改善。

本轮完成 6 个 α × 4 只大鼠 × 2 套协议 × 3 个种子，共 144 次训练结果；
新增 93 次 CUDA 13 训练，复用 51 次已审计训练，每次 100 epochs。
α 为本项目状态与增量联合投影的权重，0 对应精确 PCA；其余参数固定。

| α | 四动物平均位置 MAE / cm ↓ | 跨个体一致性 R² ↑ |
|---|---:|---:|
| 0（PCA） | 9.798 | **0.7216** |
| 0.01 | 9.616 | 0.6250 |
| 0.03 | **9.505** | 0.7008 |
| 0.1 | 10.019 | 0.6517 |
| 0.3 | 9.742 | 0.6833 |
| 1 | 9.955 | 0.6512 |

解码的共同最优 α=0.03，MAE 相对 PCA 下降 **3.00%**：4/4 动物均值改善，
8/12 个配对种子改善。四个留一动物选参折也都选到 0.03。逐只动物单独选参时，
Achilles/Cicero 最优为 0.03，Buddy/Gatsby 为 0.01。
跨个体一致性的最佳均值仍是 α=0，本轮未找到同时提升两项指标的 α。

这是已查看数据上的探索性调参及回顾性留一动物诊断，不能视为新动物上的独立
泛化验证。参数选择使用了行为指标；表征训练本身不使用行为标签。解码采用
20 维投影和 32 维输出，一致性采用 10 维投影和 3 维输出，属于两套独立协议。
一致性为位置分箱后的同样本仿射回归 R²，不是零样本迁移。

- [完整结果、每个种子、每个方向及曲线](reports/marble-alpha-grid-20260930/RESULTS.md)
- [高清曲线](reports/marble-alpha-grid-20260930/alpha_grid.png) · [矢量 PDF](reports/marble-alpha-grid-20260930/alpha_grid.pdf)
- [冻结的搜索方案](MARBLE_ALPHA_GRID_PROTOCOL.md) · [数值验证修订](MARBLE_ALPHA_GRID_NUMERICS.md)
- [机器可读汇总](reports/marble-alpha-grid-20260930/aggregate.json) · [验证记录](reports/marble-alpha-grid-20260930/validation_summary.json)

全部 48 组通过原始代码 checkpoint 审计；全部 24 组一致性条件通过统一的
float32 同起点单步核对与 float64 完整轨迹核对。原连续 float32 校验的失败日志
保留并披露，正式训练仍为 float32。18 组一致性指标通过 CEBRA 0.4.0 独立核查，
最大 R² 差异 1.66e-7。337 项测试通过；8 个变更 Python 文件通过 Ruff 检查和格式检查。

大型产物保存在 `D:/MARBLE-experiments/KAIR/alpha-grid-20260930`。
报告目录保存完整指标、审计记录、原始与修订后的源码快照；历史训练产物未修改。

## 重跑入口

在 KAIR 根目录，使用已配置的 uv 环境；选择一个不存在的新输出目录：

```powershell
.venv/Scripts/python.exe -m scripts.marble.run_alpha_grid --output D:/MARBLE-experiments/KAIR/alpha-grid-rerun --previous D:/MARBLE-experiments/KAIR/dynamics-multirat-20260929-v2
.venv/Scripts/python.exe -m scripts.marble.score_alpha_grid --root D:/MARBLE-experiments/KAIR/alpha-grid-rerun
../MARBLE/reproduction/.venv/Scripts/python.exe -m scripts.marble.score_alpha_grid --root D:/MARBLE-experiments/KAIR/alpha-grid-rerun --audit-cebra
.venv/Scripts/python.exe -m scripts.marble.report_alpha_grid --root D:/MARBLE-experiments/KAIR/alpha-grid-rerun --output docs/reports/marble-alpha-grid-rerun
```

重跑仍复用既有审计基线；失败或中断时保留日志，查明原因后使用 `--resume`。
上述旧网格入口依赖的 α=0.1 和 Achilles α=1 文件已清理，不能直接运行，需先重建对应条件。
CUDA 稀疏计算不保证逐位确定性，重跑数值可能有细微差别。
