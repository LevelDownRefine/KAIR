# 高阶脑连接工具

`utils.utils_hoi` 实现了 [Santoro 等人的研究《Charting higher-order models of brain function beyond pairwise interactions》（2026）](https://doi.org/10.1038/s41467-026-75959-w) 中讨论的十一种脑连接指标。
实现基于数学定义，仅使用项目已有的 NumPy/SciPy 依赖，在 CPU 上运行，无需额外安装 JAX、Julia、SxPID 或拓扑库，也未复制外部参考源码。
这是可复用的指标实现，**不代表对论文完整 HCP 分析逐位一致的复现**。

函数文档统一使用中文 Google 风格：`Args` 说明参数，`Returns` 说明返回值，`Raises` 说明异常；生成器使用 `Yields`，结果类使用 `Attributes`。
标识符和这些标准段名保留英文，说明正文使用中文。

## 输入与输出

输入为**单次扫描**的有限实数数组，形状为 `(时间点数, 脑区数)`。例如 116 个 ROI、1,200 个时间点对应 `(1200, 116)`。
影像预处理、混杂回归、滤波、删帧、脑区分割与 ROI 信号提取需在调用前完成。跨扫描或跨指标比较时，必须保持相同的脑区顺序。
常量脑区会触发异常，但显式选择的离散／二值信息量估计允许常量。函数不修改输入，也不会自动删除缺失样本、脑区或帧。

116 个脑区共有 **253,460 个三元组**和 **6,670 个脑区对**，每条边关联 114 个三元组。
投影结果为 `(116, 116)` 对称矩阵，除 FC 的对角线为 1 外，其余指标对角线为零。
这些指标描述统计关系，不能据此直接确立有向因果效应或意识机制。

```python
import numpy as np
from utils import utils_hoi as hoi

# 合成扫描示例；实际使用时替换为预处理后的“时间点 × ROI”数组。
x = np.random.default_rng(42).normal(size=(300, 8))

# 显式选择指标；metrics=hoi.ALL_METRICS 可计算全部十一种指标。
maps = hoi.compute_hoi(
    x,
    metrics=("FC", "O-Red", "O-Syn", "PED-Red", "PED-Syn",
             "PhiID-Red", "PhiID-Syn"),
    reduction="mean",
    lag=1,
)
names, rho = hoi.compare_connectomes(maps)
np.savez("hoi_maps.npz", **maps)
np.savez("hoi_similarity.npz", names=np.array(names), rho=rho)
```

保留扫描边界，尤其不要为计算 PhiID 而拼接不同扫描或受试者：拼接会生成跨边界的人为滞后配对。
删帧后，`lag=1` 表示相邻的**保留样本**，未必对应相邻采集时刻；需要固定实际时间间隔时，应使用连续有效片段。
工具不会推断时间缺口，也不会代为汇总跨扫描或受试者结果。保存输出时，应同时记录 ROI 标签、扫描标识、预处理流程以及估计器参数。

## 指标与计算约定

| 指标 | 计算方式 | 单位／符号 |
| --- | --- | --- |
| FC | 脑区序列之间的 Pearson 相关 | 有正负的相关系数 |
| O-Red | 三元组 O-information 的正值部分，再投影 | 比特；非负 |
| O-Syn | 三元组 O-information 的负值部分，再投影 | 比特；**保留负号** |
| PED-Red | 二值共享排除熵分解的底部原子 `{1}{2}{3}` | 比特 |
| PED-Syn | 七个指定的联合源熵原子之和 | 比特 |
| PhiID-Red | 过去／未来脑区对的 MMI 冗余到冗余原子 `rtr` | 比特 |
| PhiID-Syn | MMI 协同到协同原子 `sts` | 比特；不截断原子值 |
| Triangles | 违反面顺序的三角形正权重的时间均值，再投影 | 标准化乘积权重 |
| TScaffold | 逐帧有效复形上 H1 代表环频次的时间均值 | 环计数 |
| Fscaff | 静态 `1 - FC` 滤过的 H1 代表环频次 | 环计数 |
| Pscaff | 将 H1 代表环的持久长度累加到其各条边 | `1 - FC` 滤过参数单位 |

### O-information 与投影

对于 `d` 个变量，`Omega = (d-2) H(X) + sum_i [H(X_i) - H(X_without_i)]`。
在三元组上，它等于共信息。`o_information(x)` 将**全部输入列**视为一组变量；`oinfo_triplets(x)` 则分别计算各个三元组。

默认 `method="gc"` 使用边缘平均秩的 `ndtri(rank / (T+1))` 变换，然后通过高斯协方差对数行列式估计熵，并进行有限样本 digamma 偏差校正。
`method="gaussian"` 使用中心化样本协方差，默认不开启该校正。`biascorrect` 可覆盖此选择。
函数不自动正则化，相关协方差块奇异时会报错；只计算三元组时，不要求完整 N 维协方差可逆。
高斯／copula 估计无法捕获任意非线性依赖，例如二值 XOR。类别观测可使用 `method="discrete"`，此时不同数值即不同类别，不进行自动分箱。

```python
raw = hoi.oinfo_triplets(x)  # triplets 形状为 (K, 3)，values 形状为 (K,)。
edge_maps = hoi.split_oinfo(raw, reduction="mean")
node_maps = hoi.split_oinfo(raw, target="nodes", reduction="mean")
```

冗余与协同保留为**两张独立的图**。比较各项指标前将两者相加，会丢失分离正负贡献所保留的信息。
`project_triplets(values, triplets, n_regions, ...)` 支持：

- `sum`：累加所有提供的关联三元组分数。
- `mean`（默认）：除以提供的关联三元组总数，**包括符号分离后的零贡献**。
- `mean_nonzero`：仅按非零贡献的数量计算均值。

在固定脑区数、完整枚举三元组时，均值与求和只相差常数倍，所得边向量的 Spearman 相关相同。
对非零贡献归一化或只选部分三元组时，这一结论一般不成立。没有关联项的位置返回零。
显式三元组索引须每行严格递增、行间唯一，输出保留行顺序；默认枚举按字典序排列。

### PED：部分熵分解

`ped_triplets` 默认逐脑区按时间均值二值化：大于均值为 1，否则为 0，包括恰好等于均值的观测。
这等价于对 Z 分数信号使用零阈值。已经是 0/1 数据时传入 `binary=True`。

对每个观测联合状态，共享熵定义为 `h_sx(alpha) = -log2 P(alpha 中匹配源事件的并集)`。
先按经验二值联合分布取期望，再对十八节点冗余格进行 Möbius 反演。
冗余分数取 `{1}{2}{3}`；协同分数为 `{3}{12}`、`{2}{13}`、`{1}{23}`、`{12}{13}{23}`、`{13}{23}`、`{12}{23}`、`{12}{13}` 七项之和，不按联合熵归一化。

`ped_from_probabilities(p)` 返回八状态分布的全部十八个原子，概率索引为 `X0 + 2*X1 + 4*X2`。
反链键使用源集合位掩码：`1=X0`、`2=X1`、`4=X2`、`3={X0,X1}`，依此类推。
这里分解的是**联合熵**，没有选定某个脑区作为 PID 的目标。独立公平比特仍可具有正共享熵，底部原子为 `log2(8/7)`。

### PhiID：整合信息分解

`phiid_pair(x[:, [i,j]], lag=1, method="gaussian")` 返回十六个 MMI 原子。
键以 `t` 连接过去和未来的标签 `r`、`u1`、`u2`、`s`，例如 `rtr`、`u1tu2`、`sts`。
`u1`、`u2` 分别对应输入的第一、第二个脑区的独有信息。所有原子之和等于 `I(pair_past; pair_future)`。
MMI 的 `rtr` 是四项单脑区过去／未来互信息中的最小值；其余累积冗余与原子由两个二源冗余格的乘积构造。单个原子不一定非负。

`phiid_matrices` 对两张图使用同一个估计器，默认是高斯估计。
高层接口 `compute_hoi` 默认沿用作者脚本的选择：`phiid_red_method="gc"`、`phiid_syn_method="gaussian"`。
若要在同一估计器下比较两个原子，请显式将两项设置为相同方法。
gc 先在完整扫描上进行高斯化，再构造滞后数据并中心化各协方差。离散 PhiID 也受支持，但不自动阈值化。
高斯 PhiID 至少需要五组保留的滞后观测，且四变量协方差必须非奇异。

### 拓扑指标

`correlation_scaffolds(x)` 返回频次、持久性、H1 区间和代表环的边列表。
边在 `1 - FC[i,j]` 时进入，每个三角形在三条边出生参数的最大值处进入，不对边权进行秩变换。实现使用 F2 上的边界矩阵约化。

`temporal_higher_order` 先在完整扫描上对脑区序列及三元乘积序列进行 Z 分数标准化。
乘积权重取标准化幅度的绝对值，仅当所有瞬时脑区值具有相同的非零符号时赋正号。

- `edge_mode="standardized"`（默认）：对二元乘积应用相同标准化和符号规则，对应作者的 **SimplicialTS 源码**。
- `edge_mode="raw"`：边权为 `z_i*z_j`，对应**文章的边权公式**。

按权重降序进入复形时，只有三条边都已进入的三角形才有效，即 `triangle_weight <= min(edge_weights)`。
无效三角形被删除，不延后加入。只有被删除的正权重计入 Triangles 特征。
有效复形的 H1 决定 TScaffold；将其替换为仅由边构造的 Vietoris-Rips 复形会改变指标。
常量乘积序列的标准化波动设为零，以避免除零；时间均值包含零贡献。

```python
# 显式选取输出帧，标准化统计仍使用完整扫描。
temporal = hoi.temporal_higher_order(x, frames=[0, 10, 20])
triangle_map = hoi.project_triplets(
    temporal.triangles.values, temporal.triangles.triplets, x.shape[1]
)
tscaffold = temporal.scaffold
# 省略 frames 即计算全部帧；只需 Triangles 时可设置 scaffold=False。
```

代表环**并不唯一**。本实现对同权边和三角形使用字典序，面先于其上维单纯形。
有限区间的代表环取约化后的三角形边界，不消亡区间取出生环，只保留正长度区间。
标准 H1 条形码一致不保证骨架边权与 Ripserer 选取的代表环逐边一致；同权节点重新编号也可能影响代表环。
不消亡的环计入频次，并对其边的持久性贡献 `+inf`；完整静态滤过没有不消亡的 H1 环。
`scaffold_from_filtration` 支持显式设置三角形集合及出生参数，以检查这些选择。`compare_connectomes` 拒绝包含无穷值的图。

## 连接图比较

`edge_vector(matrix)` 按 `np.triu_indices(N, 1)` 顺序提取严格上三角，排除对角线。
`compare_connectomes` 对这些对齐的边向量使用平均秩处理并列值，返回指标间的 Spearman 相关矩阵。
常量向量的相关未定义，返回 NaN，包括对应对角线。函数不假设各条边相互独立，也不返回逐边显著性 p 值。
相关衡量的是空间排序模式的一致性，不能据此验证估计器的正确性或认定生物学机制等价。

## 规模与验证

高斯三元组使用批量 3×3 矩阵计算，PED 和乘积统计采用分块处理，时间拓扑逐帧计算。
实现不分配完整的 `(时间点数, 全部三元组数)` 数组，但分块内存仍随时间点数及 `chunk_size` 增长。
拓扑每帧枚举 O(N^3) 个面，边界约化耗时取决于数据。完整 HCP 扫描的逐帧计算可能耗时较长；
探索时可先显式选择少量帧，正式分析再按研究设计计算完整帧集合。工具不会隐式抽样。

验证覆盖独立／复制／XOR 离散分布、高斯熵恒等式、十八个 PED 原子、十六个 PhiID 原子、环闭合性、有限与不消亡 H1 区间、帧平均、投影分母、并列秩、非法输入和全部十一种输出。
离线参考数据保存了 SxPID、PhiID 的数值及固定源码版本。另有四十个完整／部分三角形面的随机滤过与 GUDHI 3.13.0 对照，检验的是条形码，不要求代表环相同。测试未使用 HCP 数据集。

```powershell
python -m pytest tests/utils/test_utils_hoi.py -q
python -m pytest tests -k "not cuda" -q
```

## 来源与复现差异

- [论文](https://doi.org/10.1038/s41467-026-75959-w)。
- [作者分析脚本，固定版本 ce1a32c](https://github.com/nplresearch/HOI_lenses_analysis/tree/ce1a32ccba63dcc20c495b77be7352ad2993575d)：O-information／PhiID 调用、PED 原子选择、静态骨架及 SimplicialTS。
- [HOI 的 O-information 估计器](https://brainets.github.io/hoi/api/generated/hoi.metrics.Oinfo.html)与[高斯 copula 熵](https://brainets.github.io/hoi/api/generated/hoi.core.entropy_gc.html)。
- [SxPID 参考实现](https://github.com/nplresearch/HOI_lenses_analysis/blob/ce1a32ccba63dcc20c495b77be7352ad2993575d/Code/04_PED/sxpid/SxPID.py)。
- [MMI PhiID 参考实现，固定版本 6c5f2e9](https://github.com/Imperial-MIND-lab/integrated-info-decomp/tree/6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44)。
- [GUDHI 单纯形树文档](https://gudhi.inria.fr/python/latest/simplex_tree_ref.html)。

复现时应记录：均值／求和／非零归一化投影，原始／标准化时间边权，协方差中心化及偏差校正，
并列观测的平均秩处理（HOI 的 argsort 实现会赋予不同秩），常量乘积处理，滞后与帧选择，以及代表环约定。
论文描述平均投影，而作者部分笔记本使用求和，可选源码归一化还排除了零贡献。这些选择在本实现中显式公开，不默认视为等价。
当前工具不包含额外的 FDR、替代数据检验、个体指纹识别、任务解码或行为 PLS 分析。
