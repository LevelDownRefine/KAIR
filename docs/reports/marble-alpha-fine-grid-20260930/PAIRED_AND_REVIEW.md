# 配对种子与统计解释

Material Passport: academic-research-suite / experiment-agent; mode validate; origin 2026-09-30; status ANALYZED; interpretation CAUTION (exploratory adaptive tuning).

一致性采用三种子均值选参，但均值提高不代表每个种子都提高。下表为相对同种子 PCA 的 R² 差值。

| α | seed 0 ΔR² | seed 1 ΔR² | seed 2 ΔR² | 均值 ΔR² | 改善种子数 |
|---|---:|---:|---:|---:|---:|
| 0 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0/3 |
| 0.01 | -0.13352 | +0.04691 | -0.20340 | -0.09667 | 1/3 |
| 0.02 | -0.09205 | +0.15163 | -0.15640 | -0.03228 | 1/3 |
| 0.03 | -0.02386 | +0.15132 | -0.18987 | -0.02080 | 1/3 |
| 0.04 | -0.03720 | +0.16105 | -0.07538 | +0.01616 | 1/3 |
| 0.05 | -0.04457 | +0.12594 | -0.16407 | -0.02757 | 1/3 |

α=0.04 是本轮唯一同时改善两项总体均值的非零参数：MAE 降低约 2.36%，一致性 R² 增加 0.01616。其解码在 3/4 动物均值、9/12 配对种子上改善；一致性仅 1/3 配对种子改善，均值增益主要由 seed 1 带动（+0.16105），seed 0/2 分别下降 0.03720/0.07538。这是候选效果，不是稳定提升的证据。

α=0.03 仍是共同解码最优值：4/4 动物均值改善，8/12 配对种子改善；四个回顾性留一动物选参折仍全部选择 0.03。Gatsby 的单动物最优改为 0.04，Buddy 为 0.01，Achilles/Cicero 为 0.03。不能用单动物最优拼成一个未经声明的共同参数。

图中误差条是种子样本标准差，不是置信区间；没有将时间点、种子或 12 个方向当作独立动物样本。本轮是在旧网格结果基础上的适应性细搜，行为标签用于指标及选参，不能作为独立测试证据。

## 方法学检查：11/11

- **Simpson's paradox**：Animal means and paired differences are retained; macro MAE gives each animal equal weight, not weight by record length.
- **Ecological fallacy**：Inference is restricted to four public rat recordings; seeds, time bins and directed pairs are not biological replicates.
- **Berkson/selection bias**：The released subset is not a random population sample. No population prevalence or universal neural property is inferred.
- **Collider bias**：No post-hoc animal/seed filtering. Position conditioning defines the in-sample consistency metric and is disclosed, not used as causal adjustment.
- **Base-rate neglect**：Not applicable to these continuous regression/reconstruction endpoints; no diagnostic sensitivity or predictive-value claim.
- **Regression to the mean**：This search follows a selected coarse optimum on the same records; winner's optimism remains. Reused controls and all seeds are retained, and fresh-data confirmation is absent.
- **Survivorship bias**：All 24 planned new conditions and 72 seeds completed. Poor historical tensor files were deleted at the user's request, while every score and audit record remains available.
- **Look-elsewhere effect**：All six local points and the prior coarse report are shown; selection is adaptive and exploratory. No p-values or significance claims are made.
- **Garden of forking paths**：Local points/endpoints were frozen before new outcomes. Only CPU input-preparation scheduling changed during execution; algorithm sources and training settings stayed fixed.
- **Correlation/causation**：Claims concern measured performance under an algorithmic intervention on fixed recordings, not causal neural or behavioral mechanisms.
- **Reverse causality**：Offline position decoding and position-binned affine similarity do not establish a causal direction or future-behavior prediction.

输入预备曾用两个 CPU 工作进程提前生成 0.05、随后 0.04 条件；正式 GPU 训练保持单进程、既定顺序及 float32。两个调度记录和脚本快照保存在本目录，正式执行源码哈希未改变。

[清理范围及影响](../../MARBLE_ARTIFACT_RETENTION.md)：已按用户要求删除旧劣势参数的大型张量和权重，历史分数仍保留。本次局部网格及所用 0/0.01/0.03 对照全部保留。
