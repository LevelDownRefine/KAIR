"""使用 NumPy 计算 H1 拓扑骨架和随时间变化的单纯形特征。

确定性的 F2 边界矩阵约化会选定一组代表环。骨架边权依赖代表环的选择，
不能仅凭持久同调条形码一致就保证与论文的 Julia/Ripserer 输出逐边一致。
"""

from dataclasses import dataclass

import numpy as np

from utils._hoi_common import multiplets, positive_int, standardize, timeseries
from utils.utils_hoi_information import TripletValues


@dataclass
class ScaffoldResult:
    """保存单个滤过的 H1 拓扑骨架与代表环。

    Attributes:
        frequency (numpy.ndarray): (N, N) 对称零对角矩阵，记录每条边参与
            正持久长度代表环的次数，包括不消亡的环。
        persistence (numpy.ndarray): (N, N) 对称零对角矩阵，累加每条边
            所属代表环的持久长度；不消亡的环贡献正无穷。
        intervals (numpy.ndarray): (M, 2) 数组，每行为一个环的出生与消亡
            参数，不消亡时第二列为正无穷。仅保留长度严格大于零的区间。
        cycles (tuple): 与 intervals 逐行对应的代表环。每个环是由脑区
            索引对 (i, j) 构成的边元组。此数据类自身不执行校验。
    """

    frequency: np.ndarray
    persistence: np.ndarray
    intervals: np.ndarray
    cycles: tuple


@dataclass
class TemporalHOIResult:
    """保存所选时间帧上的高阶拓扑特征均值。

    Attributes:
        triangles (TripletValues): 违反面顺序的三角形正权重的时间均值。
            所有三元组均保留，未贡献的帧按零计入平均。
        scaffold (numpy.ndarray | None): (N, N) 的 H1 环频次均值矩阵。
            未请求计算骨架时为 None。
        frames (numpy.ndarray): 实际参与时间平均的原始帧索引副本。
    """

    triangles: TripletValues
    scaffold: np.ndarray | None
    frames: np.ndarray


def _bit_indices(bits):
    """从低位到高位枚举整数中值为 1 的位索引。

    Args:
        bits (int): 调用者保证非负的 Python 整数，用作压缩位集合。

    Yields:
        int: 从零开始的位索引；输入为零时不产生元素。
    """
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


def scaffold_from_filtration(edge_filtration, *, triangles=None, triangle_filtration=None):
    """对带三角形面的完整图滤过计算 H1 频次骨架与持久性骨架。

    所有顶点从初始时刻存在。同权单纯形按字典序处理，面先于其上维
    单纯形。只统计持久长度严格大于零的环。有限区间采用约化后的三角形
    边界作为代表环；不消亡区间采用出生环。本 CPU 实现适合中等脑区规模。

    Args:
        edge_filtration (array_like): 形状为 (N, N) 的有限实数对称矩阵，
            N >= 2，非对角元素为边的出生参数；对角线不参与滤过。
        triangles (array_like | None): 形状为 (K, 3) 的三角形整数索引，
            每行严格递增、行间唯一且位于 [0, N)。None 表示全部三角形；
            (0, 3) 空数组表示不添加任何三角形面。
        triangle_filtration (array_like | None): 与 triangles 逐行对应的
            (K,) 有限出生参数，每项不得早于其三条边。None 表示取三条边
            的最大出生参数；同时省略两个三角形参数即为 Vietoris-Rips
            复形的二维骨架。

    Returns:
        ScaffoldResult: 频次矩阵、持久性矩阵、H1 区间和代表环。
        不消亡的环计入频次，并对其边的持久性贡献正无穷。

    Raises:
        ValueError: 边矩阵不是有限实数对称方阵、脑区数不足，或三角形
            索引、出生参数不合法。
    """
    if np.iscomplexobj(edge_filtration):
        raise ValueError("edge filtration must be real")
    matrix = np.asarray(edge_filtration, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 2:
        raise ValueError("edge filtration must be a square matrix with at least two nodes")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
        raise ValueError("edge filtration must be finite and symmetric")
    n = len(matrix)
    edge_array = multiplets(n, 2)
    edge_values = matrix[edge_array[:, 0], edge_array[:, 1]]
    # 稳定排序使同权边保留原始字典序。
    order = np.argsort(edge_values, kind="stable")
    edges = edge_array[order]
    births = edge_values[order]
    edge_index = {tuple(e): k for k, e in enumerate(edges)}

    faces = multiplets(n, 3, triangles)
    face_edges = np.array([[edge_index[(int(t[a]), int(t[b]))]
                            for a, b in ((0, 1), (0, 2), (1, 2))]
                           for t in faces], dtype=int).reshape(-1, 3)
    earliest = births[face_edges].max(axis=1)
    if triangle_filtration is None:
        deaths = earliest
    else:
        deaths = np.asarray(triangle_filtration, dtype=float)
        if deaths.shape != (len(faces),) or not np.isfinite(deaths).all() or np.any(deaths < earliest):
            raise ValueError("triangle births must be finite, aligned with triangles, and >= every edge birth")

    # 约化边界算子 d1，保留基变换列，用于构造不消亡的出生环。
    vertex_pivots = {}
    birth_cycles = {}
    for k, (i, j) in enumerate(edges):
        boundary = (1 << int(i)) ^ (1 << int(j))
        chain = 1 << k
        while boundary:
            pivot = boundary.bit_length() - 1
            if pivot not in vertex_pivots:
                vertex_pivots[pivot] = (boundary, chain)
                break
            old_boundary, old_chain = vertex_pivots[pivot]
            boundary ^= old_boundary
            chain ^= old_chain
        if not boundary:
            birth_cycles[k] = chain

    # 用 Python 整数表示压缩位集合，在 F2 上约化边界算子 d2。
    edge_pivots = {}
    records = []
    face_order = sorted(range(len(faces)), key=lambda k: (deaths[k], tuple(faces[k])))
    for k in face_order:
        boundary = sum(1 << int(e) for e in face_edges[k])
        while boundary:
            pivot = boundary.bit_length() - 1
            if pivot not in edge_pivots:
                edge_pivots[pivot] = boundary
                if deaths[k] > births[pivot]:
                    records.append((births[pivot], deaths[k], boundary))
                break
            boundary ^= edge_pivots[pivot]
    for pivot, chain in birth_cycles.items():
        if pivot not in edge_pivots:
            records.append((births[pivot], np.inf, chain))

    records.sort(key=lambda r: (r[0], r[1], r[2]))
    frequency = np.zeros((n, n), dtype=float)
    persistence = np.zeros_like(frequency)
    cycles = []
    for birth, death, chain in records:
        cycle = tuple(tuple(int(v) for v in edges[e]) for e in _bit_indices(chain))
        cycles.append(cycle)
        for i, j in cycle:
            frequency[i, j] += 1
            frequency[j, i] += 1
            persistence[i, j] += death - birth
            persistence[j, i] += death - birth
    intervals = np.array([(b, d) for b, d, _ in records], dtype=float).reshape(-1, 2)
    return ScaffoldResult(frequency, persistence, intervals, tuple(cycles))


def correlation_scaffolds(data):
    """基于 1 - Pearson FC 的滤过计算静态 Fscaff 和 Pscaff。

    Args:
        data (array_like): 单次扫描的有限实数序列，形状为 (T, N)，T 和 N
            均至少为 2，各脑区序列不能恒定。

    Returns:
        ScaffoldResult: frequency 为 Fscaff，persistence 为 Pscaff，并
        附带 H1 区间与代表环。边权使用 1-FC，不进行秩变换；完整静态
        滤过不存在不消亡的 H1 环。

    Raises:
        ValueError: 输入形状或数值不合法，或存在恒定脑区序列。
    """
    x = timeseries(data)
    corr = np.corrcoef(x, rowvar=False)
    return scaffold_from_filtration(1 - (corr + corr.T) / 2)


def _product_stats(z, indices, chunk_size):
    """分块计算脑区乘积序列的时间均值和标准差。

    Args:
        z (numpy.ndarray): 已标准化的 (T, N) 脑区序列。
        indices (numpy.ndarray): 形状为 (K, d) 的有效整数索引，指定 d 阶组合。
        chunk_size (int): 已校验的正整数分块大小。

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: (时间均值, 时间标准差)，均为
        (K,) 数组。标准差使用 ddof=0；常量乘积的标准差保留为零，供后续
        权重计算显式处理。
    """
    means = np.empty(len(indices))
    scales = np.empty(len(indices))
    for start in range(0, len(indices), chunk_size):
        block = np.prod(z[:, indices[start:start + chunk_size]], axis=2)
        means[start:start + len(block.T)] = block.mean(axis=0)
        scales[start:start + len(block.T)] = block.std(axis=0)
    # 保留常量乘积的零标准差，由权重函数将其标准化波动设为零。
    return means, scales


def _coherent_weights(z_frame, indices, means, scales):
    """由乘积幅度和脑区瞬时符号一致性生成单帧权重。

    Args:
        z_frame (numpy.ndarray): 单帧标准化脑区值，形状为 (N,)。
        indices (numpy.ndarray): 形状为 (K, d) 的有效组合索引。
        means (numpy.ndarray): 形状为 (K,) 的全时段乘积均值。
        scales (numpy.ndarray): 形状为 (K,) 的全时段乘积标准差。

    Returns:
        numpy.ndarray: 形状为 (K,) 的权重。先取标准化乘积的绝对值，
        仅当所有脑区值均为正或均为负时赋正号，其余赋负号。
        标准差为零的乘积返回零权重。
    """
    samples = z_frame[indices]
    products = np.prod(samples, axis=1)
    centered = np.divide(products - means, scales, out=np.zeros_like(products), where=scales > 0)
    coherent = np.all(samples > 0, axis=1) | np.all(samples < 0, axis=1)
    return np.where(coherent, 1.0, -1.0) * np.abs(centered)


def temporal_higher_order(data, *, frames=None, edge_mode="standardized", chunk_size=256, scaffold=True):
    """计算违反面顺序的 Triangles 特征及 TScaffold 频次的时间均值。

    先在完整扫描内标准化各脑区及其三元乘积，再按瞬时脑区符号是否一致
    设置乘积权重的正负号。按权重降序构造滤过，移除权重大于任一组成边
    的三角形。Triangles 只累加被移除三角形的正权重，其余贡献为零；
    TScaffold 为保留复形上 H1 环频次的时间均值。

    Args:
        data (array_like): 单次扫描的有限实数序列，形状为 (T, N)，T >= 2、
            N >= 3，各脑区序列不能恒定。
        frames (array_like | None): 非空、不重复且位于 [0, T) 的整数帧索引。
            None 表示全部帧。选择帧仅影响输出的时间平均；所有标准化和
            乘积统计仍使用完整扫描，无隐式下采样。
        edge_mode (str): "standardized" 对二元乘积使用相同标准化和符号
            规则，对应作者 SimplicialTS 代码；"raw" 使用 z_i*z_j，
            对应文章边权公式。默认为 "standardized"，两者可能得到不同复形。
        chunk_size (int): 乘积统计的正整数分块大小，默认为 256。
        scaffold (bool): 是否计算 TScaffold，默认为 True。False 时只计算
            Triangles，可减少拓扑约化开销。

    Returns:
        TemporalHOIResult: 原始三元组权重均值、可选的 (N, N) 骨架频次均值
        及所选帧索引。恒定乘积的标准化波动置零，零贡献计入时间均值。

    Raises:
        ValueError: 输入、帧索引、边权模式或分块大小不合法，或脑区序列
            无法标准化。
    """
    x = timeseries(data, min_regions=3)
    chunk_size = positive_int(chunk_size, "chunk_size")
    if edge_mode not in ("standardized", "raw"):
        raise ValueError("edge_mode must be 'standardized' or 'raw'")
    if frames is None:
        frames = np.arange(len(x))
    else:
        frames = np.asarray(frames)
        if frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer) or not len(frames):
            raise ValueError("frames must be a nonempty integer vector")
        if np.any(frames < 0) or np.any(frames >= len(x)) or len(np.unique(frames)) != len(frames):
            raise ValueError("frames must be unique and in range")
    z = standardize(x)
    n = x.shape[1]
    edges = multiplets(n, 2)
    triples = multiplets(n, 3)
    means, scales = _product_stats(z, triples, chunk_size)
    if edge_mode == "standardized":
        edge_means, edge_scales = _product_stats(z, edges, chunk_size)
    tri_sum = np.zeros(len(triples))
    scaffold_sum = np.zeros((n, n)) if scaffold else None
    for frame in frames:
        weights = _coherent_weights(z[frame], triples, means, scales)
        if edge_mode == "standardized":
            edge_weights = _coherent_weights(z[frame], edges, edge_means, edge_scales)
        else:
            edge_weights = np.prod(z[frame, edges], axis=1)
        matrix = np.zeros((n, n))
        matrix[edges[:, 0], edges[:, 1]] = edge_weights
        matrix += matrix.T
        face_min = np.minimum.reduce([matrix[triples[:, a], triples[:, b]] for a, b in ((0, 1), (0, 2), (1, 2))])
        valid = weights <= face_min
        tri_sum += np.where(~valid & (weights > 0), weights, 0)
        if scaffold:
            result = scaffold_from_filtration(-matrix, triangles=triples[valid], triangle_filtration=-weights[valid])
            scaffold_sum += result.frequency
    if scaffold:
        scaffold_sum /= len(frames)
    return TemporalHOIResult(TripletValues(triples, tri_sum / len(frames), n), scaffold_sum, frames.copy())
