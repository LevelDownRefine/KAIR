"""十一种高阶脑连接指标的公共入口。

输入为单次扫描预处理后的 ROI 时间序列，形状为 (时间点数, 脑区数)。
本模块不执行影像预处理；估计方法与计算约定见 docs/README_HOI.md。
"""

from itertools import combinations

import numpy as np
from scipy.stats import rankdata

from utils._hoi_common import multiplets, positive_int, timeseries
from utils.utils_hoi_information import (
    PEDResult, TripletValues, o_information, oinfo_triplets, ped_from_probabilities,
    ped_triplets, phiid_matrices, phiid_pair,
)
from utils.utils_hoi_topology import (
    ScaffoldResult, TemporalHOIResult, correlation_scaffolds,
    scaffold_from_filtration, temporal_higher_order,
)

ALL_METRICS = (
    "FC", "O-Red", "O-Syn", "PED-Red", "PED-Syn", "PhiID-Red", "PhiID-Syn",
    "Triangles", "TScaffold", "Fscaff", "Pscaff",
)

__all__ = [
    "ALL_METRICS", "TripletValues", "PEDResult", "ScaffoldResult", "TemporalHOIResult",
    "functional_connectivity", "o_information", "oinfo_triplets",
    "ped_from_probabilities", "ped_triplets", "phiid_pair", "phiid_matrices",
    "scaffold_from_filtration", "correlation_scaffolds", "temporal_higher_order",
    "project_triplets", "split_oinfo", "edge_vector", "compare_connectomes", "compute_hoi",
]


def functional_connectivity(data):
    """计算脑区时间序列之间的 Pearson 功能连接。

    Args:
        data (array_like): 单次扫描的有限实数数据，形状为 (T, N)，T 和 N
            均至少为 2，各脑区序列不能为常量。

    Returns:
        numpy.ndarray: 形状为 (N, N) 的对称相关矩阵，对角线为 1。

    Raises:
        ValueError: 数据形状或数值不合法，或存在恒定脑区序列。
    """
    return np.corrcoef(timeseries(data), rowvar=False)


def project_triplets(values, triplets, n_regions, *, target="edges", reduction="mean"):
    """将与三元组逐行对应的分数投影到脑区对或单个脑区。

    Args:
        values (array_like): 形状为 (K,) 的有限分数，与 triplets 逐行对应。
        triplets (array_like): 形状为 (K, 3) 的整数索引，每行严格递增、
            行间不重复，取值范围为 [0, n_regions)。
        n_regions (int): 脑区总数，至少为 3。
        target (str): 投影目标。"edges" 表示脑区对，"nodes" 表示脑区；
            默认为 "edges"。
        reduction (str): 聚合方式。"sum" 求和；"mean" 按所有关联三元组
            的数量取均值，包括零分数；"mean_nonzero" 只按非零贡献计数。
            默认为 "mean"，分母仅统计显式提供的三元组。

    Returns:
        numpy.ndarray: 边投影为 (N, N) 对称矩阵且对角线为零，节点投影为
        (N,) 向量。没有关联三元组或均值分母为零的位置返回零。

    Raises:
        ValueError: 脑区数、索引、分数、投影目标或聚合方式不合法。
    """
    n_regions = positive_int(n_regions, "n_regions")
    if n_regions < 3:
        raise ValueError("at least three regions are required")
    idx = multiplets(n_regions, 3, triplets)
    scores = np.asarray(values, dtype=float)
    if scores.shape != (len(idx),) or not np.isfinite(scores).all():
        raise ValueError("values must be a finite vector aligned with triplets")
    if target not in ("edges", "nodes") or reduction not in ("sum", "mean", "mean_nonzero"):
        raise ValueError("target must be edges/nodes; reduction must be sum/mean/mean_nonzero")
    shape = (n_regions, n_regions) if target == "edges" else (n_regions,)
    result = np.zeros(shape)
    count = np.zeros(shape)
    weights = (scores != 0).astype(float) if reduction == "mean_nonzero" else np.ones(len(scores))
    if target == "edges":
        for a, b in combinations(range(3), 2):
            positions = (idx[:, a], idx[:, b])
            np.add.at(result, positions, scores)
            np.add.at(count, positions, weights)
        result += result.T
        count += count.T
    else:
        for a in range(3):
            np.add.at(result, idx[:, a], scores)
            np.add.at(count, idx[:, a], weights)
    if reduction != "sum":
        result = np.divide(result, count, out=np.zeros_like(result), where=count > 0)
    return result


def split_oinfo(result, *, target="edges", reduction="mean"):
    """分离 O-information 的正负贡献，并分别投影。

    Args:
        result (TripletValues): 原始三元组索引、O-information 分数及脑区数。
        target (str): "edges" 为边投影，"nodes" 为节点投影，默认为前者。
        reduction (str): "sum"、"mean" 或 "mean_nonzero"，含义与
            project_triplets 相同，默认为 "mean"。

    Returns:
        dict[str, numpy.ndarray]: 包含 "O-Red" 与 "O-Syn" 两张图。
        O-Red 非负，O-Syn 保留非正值；信息量单位为比特。

    Raises:
        ValueError: 原始分数或投影参数不符合 project_triplets 的要求。
    """
    return {
        "O-Red": project_triplets(np.maximum(result.values, 0), result.triplets, result.n_regions, target=target, reduction=reduction),
        "O-Syn": project_triplets(np.minimum(result.values, 0), result.triplets, result.n_regions, target=target, reduction=reduction),
    }


def edge_vector(matrix):
    """按固定顺序提取连接矩阵的严格上三角。

    Args:
        matrix (array_like): 形状为 (N, N) 的有限对称矩阵，N 至少为 2。

    Returns:
        numpy.ndarray: 长度为 N*(N-1)/2 的边向量，不含对角线，顺序与
        np.triu_indices(N, 1) 一致。116 个脑区对应 6,670 个分数。

    Raises:
        ValueError: 矩阵不是方阵、不对称、包含非有限值，或脑区数不足。
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or len(x) < 2:
        raise ValueError("matrix must be square with at least two regions")
    if not np.isfinite(x).all() or not np.allclose(x, x.T, rtol=1e-10, atol=1e-12):
        raise ValueError("matrix must be finite and symmetric")
    return x[np.triu_indices(len(x), 1)]


def compare_connectomes(matrices):
    """用 Spearman 秩相关比较各连接图的空间模式。

    并列数值使用平均秩。相关性描述边分数排序的一致程度，不返回依赖
    独立边假设的显著性检验结果。调用者须保证所有矩阵的脑区顺序一致。

    Args:
        matrices (Mapping[str, array_like]): 非空的名称到连接矩阵映射。
            所有矩阵须为同形状的有限对称方阵，至少包含两个脑区。

    Returns:
        tuple[tuple[str, ...], numpy.ndarray]: (指标名称, 相关矩阵)。名称
        保留映射的迭代顺序；M 个指标对应 (M, M) 矩阵。边向量为常量的
        指标，其整行和整列均为 NaN，包括对角线。

    Raises:
        ValueError: 映射为空、矩阵形状不同，或矩阵不符合 edge_vector
            的有限性与对称性要求。
    """
    if not matrices:
        raise ValueError("at least one connectome is required")
    names = tuple(matrices)
    vectors = [edge_vector(matrices[name]) for name in names]
    if len({len(v) for v in vectors}) != 1:
        raise ValueError("all connectomes must have the same shape and ROI order")
    ranks = np.array([rankdata(v, method="average") for v in vectors])
    ranks -= ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=1)
    denom = norms[:, None] * norms[None, :]
    result = np.divide(ranks @ ranks.T, denom, out=np.full_like(denom, np.nan), where=denom > 0)
    return names, np.clip(result, -1, 1)


def compute_hoi(data, *, metrics, reduction="mean", oinfo_method="gc",
                phiid_red_method="gc", phiid_syn_method="gaussian", lag=1,
                frames=None, edge_mode="standardized", chunk_size=256):
    """计算显式选定的脑连接指标，并统一返回连接矩阵。

    传入 metrics=ALL_METRICS 可计算全部十一种指标。默认估计器选择对应
    作者脚本：O-information 和 PhiID-Red 使用高斯 copula，PhiID-Syn
    使用高斯估计，PED 使用二值 h_sx。原始三元组分数及完整 PhiID 原子
    可通过各自的独立函数获取；复现约定见 docs/README_HOI.md。

    Args:
        data (array_like): 单次扫描的有限实数序列，形状为 (T, N)，各列
            不能恒定。不同指标所需的最少样本数与脑区数见对应函数。
        metrics (Sequence[str]): ALL_METRICS 中的非空、无重复名称序列。
            必须显式指定，不能直接传入单个字符串。
        reduction (str): 三元组投影方式，可选 "sum"、"mean"、
            "mean_nonzero"，默认为 "mean"。
        oinfo_method (str): O-information 估计器，可选 "gc"、"gaussian"、
            "discrete"，默认为 "gc"。离散估计不自动分箱。
        phiid_red_method (str): PhiID-Red 估计器，选项同上，默认为 "gc"。
        phiid_syn_method (str): PhiID-Syn 估计器，选项同上，默认为 "gaussian"。
        lag (int): PhiID 时间间隔，以保留的采样点计，默认为 1，须为正整数。
        frames (array_like | None): 时间拓扑指标使用的不重复帧索引。
            None 表示全部帧；不影响 FC、信息量估计或乘积标准化所用数据。
        edge_mode (str): 时间拓扑的边权定义，"standardized" 对应作者代码，
            "raw" 对应文章公式，默认为 "standardized"。
        chunk_size (int): 三元组与乘积统计的分块大小，默认为 256。

    Returns:
        dict[str, numpy.ndarray]: 按 metrics 顺序排列的 (N, N) 对称矩阵。
        FC 对角线为 1，其余为零；O-Syn 保留负号。各指标单位见对应函数。

    Raises:
        ValueError: 指标名称或所选指标使用的参数不合法，数据不符合要求，
            或信息量估计遇到样本不足、奇异协方差。
    """
    if isinstance(metrics, str):
        raise ValueError("metrics must be a sequence of names, not a string")
    requested = tuple(metrics)
    unknown = set(requested) - set(ALL_METRICS)
    if not requested or unknown or len(set(requested)) != len(requested):
        raise ValueError(f"metrics must be unique names from ALL_METRICS; unknown: {sorted(unknown)}")
    x = timeseries(data)
    maps = {}
    if "FC" in requested:
        maps["FC"] = functional_connectivity(x)
    if {"O-Red", "O-Syn"} & set(requested):
        maps.update(split_oinfo(oinfo_triplets(x, method=oinfo_method, chunk_size=chunk_size), reduction=reduction))
    if {"PED-Red", "PED-Syn"} & set(requested):
        ped = ped_triplets(x, chunk_size=chunk_size)
        for name, values in (("PED-Red", ped.redundancy), ("PED-Syn", ped.synergy)):
            maps[name] = project_triplets(values, ped.triplets, ped.n_regions, reduction=reduction)
    phi_cache = {}
    for name, method in (("PhiID-Red", phiid_red_method), ("PhiID-Syn", phiid_syn_method)):
        if name in requested:
            if method not in phi_cache:
                phi_cache[method] = phiid_matrices(x, lag=lag, method=method)
            maps[name] = phi_cache[method][name]
    if {"Fscaff", "Pscaff"} & set(requested):
        static = correlation_scaffolds(x)
        maps.update(Fscaff=static.frequency, Pscaff=static.persistence)
    if {"Triangles", "TScaffold"} & set(requested):
        temporal = temporal_higher_order(x, frames=frames, edge_mode=edge_mode,
                                        chunk_size=chunk_size, scaffold="TScaffold" in requested)
        raw = temporal.triangles
        maps["Triangles"] = project_triplets(raw.values, raw.triplets, raw.n_regions, reduction=reduction)
        maps["TScaffold"] = temporal.scaffold
    return {name: maps[name] for name in requested}
