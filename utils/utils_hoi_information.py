"""使用 NumPy/SciPy 计算基于信息论的脑连接指标。

输入形状为 (时间点数, 脑区数)，所有信息量均以比特为单位。
估计器、数学定义和参考实现见 docs/README_HOI.md。
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.special import digamma, ndtri
from scipy.stats import rankdata

from utils._hoi_common import multiplets, positive_int, timeseries


@dataclass
class TripletValues:
    """保存三元组索引及对应分数。

    Attributes:
        triplets (numpy.ndarray): 形状为 (K, 3) 的脑区索引，每行严格递增。
        values (numpy.ndarray): 形状为 (K,) 的分数，与 triplets 逐行对应。
            单位取决于生成函数：O-information 为比特，Triangles 为标准化
            乘积权重。此数据类自身不执行校验。
        n_regions (int): 原始脑区总数，用于构造完整投影图。
    """

    triplets: np.ndarray
    values: np.ndarray
    n_regions: int


@dataclass
class PEDResult:
    """保存三元组的部分熵分解（PED）汇总结果。

    Attributes:
        triplets (numpy.ndarray): 形状为 (K, 3) 的脑区索引。
        redundancy (numpy.ndarray): 形状为 (K,) 的底部冗余熵原子，单位为比特。
        synergy (numpy.ndarray): 形状为 (K,) 的七个指定协同熵原子之和，
            单位为比特，不按联合熵归一化。
        n_regions (int): 原始脑区总数。此数据类自身不执行校验。
    """

    triplets: np.ndarray
    redundancy: np.ndarray
    synergy: np.ndarray
    n_regions: int


def _prepare(data, method):
    """校验信息量估计所用的数据，并按需进行 copula 变换。

    Args:
        data (array_like): 形状为 (T, N) 的有限实数序列，T 至少为 2，
            N 至少为 1。
        method (str): "gaussian"、"gc" 或 "discrete"。只有 "gc" 执行
            逐列平均秩的高斯化；只有 "discrete" 允许常量列。

    Returns:
        numpy.ndarray: 形状为 (T, N) 的浮点数组。离散方法保留原类别值，
        不执行阈值划分或分箱。

    Raises:
        ValueError: 方法名称、数据形状或数值不合法，或连续估计存在常量列。
    """
    if method not in ("gaussian", "gc", "discrete"):
        raise ValueError("method must be 'gaussian', 'gc' or 'discrete'")
    x = timeseries(data, min_regions=1, nonconstant=method != "discrete")
    if method == "gc":
        # 并列观测使用平均秩，保留相等关系。
        x = ndtri(rankdata(x, axis=0, method="average") / (len(x) + 1))
    return x


def _bias(n, d):
    """计算高斯插件熵估计的有限样本偏差项。

    Args:
        n (int): 样本数，调用者须保证 n > d。
        d (int): 联合变量数。

    Returns:
        float: 以比特计的偏差项；从插件熵中减去该值可得到校正估计。
    """
    return (np.sum(digamma((n - np.arange(1, d + 1)) / 2))
            + d * np.log(2 / (n - 1))) / (2 * np.log(2))


def _entropy_function(x, method, biascorrect):
    """创建按列组合缓存结果的熵估计函数。

    Args:
        x (numpy.ndarray): 已经 _prepare 处理的 (T, N) 数组。
        method (str): "discrete" 使用经验概率；"gaussian" 和 "gc" 使用
            中心化样本协方差，后者要求输入已完成高斯化。
        biascorrect (bool): 是否校正高斯估计的有限样本偏差，离散方法忽略此项。

    Returns:
        Callable[[tuple[int, ...]], float]: 接收列索引元组并返回联合熵的
        函数，单位为比特。样本不足或协方差奇异时由该函数抛出 ValueError。
    """
    n = len(x)
    cov = None if method == "discrete" else np.atleast_2d(np.cov(x, rowvar=False))

    @lru_cache(None)
    def entropy(columns):
        """估计指定列的联合熵。

        Args:
            columns (tuple[int, ...]): 非空、不重复且在范围内的列索引。

        Returns:
            float: 以比特计的联合熵；连续方法返回微分熵。

        Raises:
            ValueError: 高斯估计的样本数不大于变量数，或协方差不正定。
        """
        d = len(columns)
        if method == "discrete":
            _, counts = np.unique(x[:, columns], axis=0, return_counts=True)
            p = counts / n
            return -np.dot(p, np.log2(p))
        if n <= d:
            raise ValueError("Gaussian entropy needs more samples than variables")
        try:
            chol = np.linalg.cholesky(cov[np.ix_(columns, columns)])
        except np.linalg.LinAlgError as exc:
            raise ValueError("singular covariance; remove duplicate/dependent signals") from exc
        result = (np.log(np.diag(chol)).sum() + d / 2 * np.log(2 * np.pi * np.e)) / np.log(2)
        return result - (_bias(n, d) if biascorrect else 0)

    return entropy


def o_information(data, *, method="gc", biascorrect=None):
    """将全部输入列视为一个变量组，计算其 O-information。

    定义为 (d-2)*H(X) + sum_i[H(X_i) - H(X_without_i)]。正值表示冗余
    占优，负值表示协同占优。高斯及高斯 copula 估计不能捕获任意非线性
    依赖；例如二值 XOR 应使用离散估计器。

    Args:
        data (array_like): 形状为 (T, d) 的有限实数数据，d 至少为 3。
            连续估计要求各列非常量且 T > d；离散估计允许常量列且 T >= 2。
        method (str): "gc" 为高斯 copula，"gaussian" 为高斯插件估计，
            "discrete" 将不同观测值视为不同类别而不自动分箱，默认为 "gc"。
        biascorrect (bool | None): 是否校正高斯熵的有限样本偏差。None 表示
            仅对 "gc" 开启；离散估计忽略此项。

    Returns:
        float: 全部输入变量的 O-information，单位为比特，保留正负号。

    Raises:
        ValueError: 数据或方法不合法、变量数或样本数不足，或协方差不正定。
    """
    x = _prepare(data, method)
    d = x.shape[1]
    if d < 3:
        raise ValueError("O-information needs at least three variables")
    correction = method == "gc" if biascorrect is None else bool(biascorrect)
    h = _entropy_function(x, method, correction)
    cols = tuple(range(d))
    return float((d - 2) * h(cols) + sum(h((i,)) - h(cols[:i] + cols[i + 1:]) for i in cols))


def oinfo_triplets(data, *, method="gc", biascorrect=None, triplets=None, chunk_size=4096):
    """计算全部或指定脑区三元组的 O-information。

    高斯估计按块处理 3×3 协方差矩阵，不构造时间点与全部三元组的乘积
    数组；不自动正则化协方差，也不截断负信息量。

    Args:
        data (array_like): 形状为 (T, N) 的有限实数数据，N 至少为 3。
            连续估计要求 T >= 4 且无常量列；离散估计要求 T >= 2。
        method (str): "gc"、"gaussian" 或 "discrete"，含义与
            o_information 相同，默认为 "gc"。
        biascorrect (bool | None): 是否校正高斯熵偏差。None 表示仅对 "gc"
            开启；离散估计忽略此项。
        triplets (array_like | None): 形状为 (K, 3) 的不重复整数索引，各行
            严格递增且位于 [0, N)。None 表示按字典序枚举全部三元组。
        chunk_size (int): 每批处理的高斯三元组数，默认为 4096，须为正整数。

    Returns:
        TripletValues: 三元组索引、以比特计的分数及原始脑区数。显式提供
        triplets 时保留其行顺序。

    Raises:
        ValueError: 输入或参数不合法、样本不足，或相关协方差块不正定。
    """
    x = _prepare(data, method)
    if x.shape[1] < 3:
        raise ValueError("at least three regions are required")
    chunk_size = positive_int(chunk_size, "chunk_size")
    idx = multiplets(x.shape[1], 3, triplets)
    correction = method == "gc" if biascorrect is None else bool(biascorrect)
    values = np.empty(len(idx))
    if method == "discrete":
        h = _entropy_function(x, method, False)
        for k, t in enumerate(idx):
            values[k] = h(tuple(t)) + sum(h((i,)) for i in t) - sum(h(p) for p in combinations(t, 2))
    else:
        if len(x) <= 3:
            raise ValueError("Gaussian triplets need at least four samples")
        corr = np.corrcoef(x, rowvar=False)
        adjust = (-_bias(len(x), 3) - 3 * _bias(len(x), 1) + 3 * _bias(len(x), 2)) if correction else 0
        for start in range(0, len(idx), chunk_size):
            t = idx[start:start + chunk_size]
            sub = corr[t[:, :, None], t[:, None, :]]
            try:
                chol = np.linalg.cholesky(sub)
            except np.linalg.LinAlgError as exc:
                raise ValueError("singular triplet covariance; remove dependent signals") from exc
            pair_dets = 1 - sub[:, (0, 0, 1), (1, 2, 2)] ** 2
            if np.any(pair_dets <= 0):
                raise ValueError("singular pair covariance")
            logdet = 2 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
            values[start:start + len(t)] = (logdet - np.log(pair_dets).sum(axis=1)) / (2 * np.log(2)) + adjust
    return TripletValues(idx, values, x.shape[1])


def _below(beta, alpha):
    """判断两个反链在冗余格中的偏序关系 beta <= alpha。

    Args:
        beta (tuple[int, ...]): 以位掩码集合表示的反链。
        alpha (tuple[int, ...]): 以位掩码集合表示的反链。

    Returns:
        bool: 若 alpha 的每个集合都包含 beta 中的至少一个集合，则为 True。
    """
    return all(any(b & a == b for b in beta) for a in alpha)


@lru_cache(None)
def _ped_lattice():
    """构造并缓存三变量 PED 的十八节点冗余格。

    Returns:
        tuple: (反链节点, 事件并集掩码, Möbius 逆矩阵, 协同原子索引)。
        节点使用位掩码集合；掩码形状为 (18, 8, 8)，依次对应反链、观测
        状态和可能状态；逆矩阵形状为 (18, 18)，协同索引包含七项。
        返回对象被缓存，调用者不得修改。
    """
    nodes = []
    for size in range(1, 8):
        for sets in combinations(range(1, 8), size):
            if all(a & b not in (a, b) for a, b in combinations(sets, 2)):
                nodes.append(sets)
    nodes = sorted(nodes, key=lambda a: (sum(_below(b, a) for b in nodes), a))
    zeta = np.array([[_below(b, a) for b in nodes] for a in nodes], dtype=float)
    states = np.arange(8)
    xor = states[:, None] ^ states[None, :]
    union = np.array([np.any([(xor & s) == 0 for s in a], axis=0) for a in nodes], dtype=float)
    # 源变量索引 0、1、2 分别对应位掩码 1、2、4。
    synergy = [(4, 3), (2, 5), (1, 6), (3, 5, 6), (5, 6), (3, 6), (3, 5)]
    syn_indices = [nodes.index(tuple(sorted(a))) for a in synergy]
    return tuple(nodes), union, np.linalg.inv(zeta), syn_indices


def ped_from_probabilities(probabilities):
    """从三变量二值分布计算全部十八个 h_sx 部分熵原子。

    h_sx 为源事件并集概率的负二进制对数的期望，再经冗余格的 Möbius
    反演得到熵原子。本函数分解联合熵，不指定某个脑区为 PID 目标。

    Args:
        probabilities (array_like): 形状为 (8,) 的联合概率向量，有限、非负，
            总和须在绝对误差 1e-10 内等于 1。索引为 X0 + 2*X1 + 4*X2。

    Returns:
        dict[tuple[int, ...], float]: 十八个熵原子，单位为比特。键为位掩码
        源集合组成的反链，例如 1=X0、2=X1、4=X2、3={X0,X1}，
        (1, 2, 4) 表示三个单源共同对应的底部原子。

    Raises:
        ValueError: 概率向量形状、有限性、非负性或归一化不满足要求。
    """
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (8,) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("probabilities must be eight nonnegative finite values summing to one")
    nodes, union, inverse, _ = _ped_lattice()
    cumulative = -np.log2(np.maximum(union @ p, np.finfo(float).tiny)) @ p
    return dict(zip(nodes, inverse @ cumulative))


def ped_triplets(data, *, binary=False, triplets=None, chunk_size=256):
    """计算脑区三元组的 PED 冗余熵和协同熵汇总。

    冗余取底部原子 {1}{2}{3}；协同取 {3}{12}、{2}{13}、{1}{23}、
    {12}{13}{23}、{13}{23}、{12}{23}、{12}{13} 七个原子之和。
    结果不按联合熵归一化。

    Args:
        data (array_like): 形状为 (T, N) 的有限实数数据，T >= 2，N >= 3。
        binary (bool): 是否已经为二值观测。False 时逐列按时间均值划分：
            大于均值为 1，其余为 0，并拒绝常量列；True 时只接受 0/1，
            允许常量列。默认为 False。
        triplets (array_like | None): 形状为 (K, 3) 的不重复整数索引，每行
            严格递增且位于 [0, N)。None 表示按字典序枚举全部三元组。
        chunk_size (int): 每批处理的三元组数，默认为 256，须为正整数。

    Returns:
        PEDResult: 索引、逐三元组的冗余/协同分数和脑区数，单位为比特。
        显式提供 triplets 时保留其行顺序。

    Raises:
        ValueError: 数据、二值编码、三元组索引或分块大小不符合要求。
    """
    x = timeseries(data, min_regions=3, nonconstant=not binary)
    if binary:
        if not np.isin(x, [0, 1]).all():
            raise ValueError("binary=True requires 0/1 data")
        bits = x.astype(np.uint8)
    else:
        bits = (x > x.mean(axis=0)).astype(np.uint8)
    idx = multiplets(x.shape[1], 3, triplets)
    chunk_size = positive_int(chunk_size, "chunk_size")
    nodes, union, inverse, syn_indices = _ped_lattice()
    red = np.empty(len(idx))
    syn = np.empty(len(idx))
    for start in range(0, len(idx), chunk_size):
        t = idx[start:start + chunk_size]
        states = (bits[:, t] * np.array([1, 2, 4], dtype=np.uint8)).sum(axis=2)
        p = np.stack([(states == s).mean(axis=0) for s in range(8)], axis=1)
        event_p = np.einsum("aij,bj->bai", union, p)
        cumulative = -np.einsum("bai,bi->ba", np.log2(np.maximum(event_p, np.finfo(float).tiny)), p)
        atoms = cumulative @ inverse.T
        red[start:start + len(t)] = atoms[:, nodes.index((1, 2, 4))]
        syn[start:start + len(t)] = atoms[:, syn_indices].sum(axis=1)
    return PEDResult(idx, red, syn, x.shape[1])


def _phiid_prepared(x, lag, method, biascorrect):
    """对已预处理的两个脑区计算 MMI PhiID 原子。

    Args:
        x (numpy.ndarray): 已由 _prepare 处理的 (T, 2) 数组。
        lag (int): 正整数时间间隔，调用者须保证至少保留两组滞后观测。
        method (str): "gaussian"、"gc" 或 "discrete"。
        biascorrect (bool): 是否校正高斯熵偏差，离散估计忽略此项。

    Returns:
        dict[str, float]: 十六个以比特计的原子，键格式与 phiid_pair 相同。

    Raises:
        ValueError: 高斯估计的滞后样本不足，或相关协方差不正定。
    """
    joint = np.column_stack((x[:-lag], x[lag:]))
    h = _entropy_function(joint, method, biascorrect)
    groups = (((0,), (1,)), ((0,),), ((1,),), ((0, 1),))
    cumulative = np.empty((4, 4))
    for i, alpha in enumerate(groups):
        for j, beta in enumerate(groups):
            cumulative[i, j] = min(
                h(a) + h(tuple(v + 2 for v in b)) - h(a + tuple(v + 2 for v in b))
                for a in alpha for b in beta
            )
    # 在过去、未来两个轴上分别进行二源菱形冗余格的 Möbius 反演。
    mobius = np.array([[1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 1, 0], [1, -1, -1, 1]])
    atoms = mobius @ cumulative @ mobius.T
    names = ("r", "u1", "u2", "s")
    return {f"{a}t{b}": float(atoms[i, j]) for i, a in enumerate(names) for j, b in enumerate(names)}


def phiid_pair(data, *, lag=1, method="gaussian", biascorrect=None):
    """计算两个脑区过去到未来的全部十六个 MMI PhiID 原子。

    全部原子之和等于 I(pair_past; pair_future)。gc 方法先对完整序列进行
    高斯化，再构造过去/未来配对并中心化协方差。不自动正则化或截断原子。
    不应拼接不同受试者或扫描；删帧后 lag 按保留样本计，不能自动识别时间缺口。

    Args:
        data (array_like): 形状为 (T, 2) 的有限实数序列。连续估计要求列
            非常量、T-lag >= 5 且四变量协方差正定；离散估计要求 T-lag >= 2。
        lag (int): 正整数时间间隔，单位为保留的采样点，默认为 1。
        method (str): "gaussian"、"gc" 或 "discrete"，默认为 "gaussian"。
            离散估计将观测值视为类别，不自动二值化。
        biascorrect (bool | None): 是否校正高斯熵偏差。None 表示仅对 "gc"
            开启；离散估计忽略此项。

    Returns:
        dict[str, float]: 十六个以比特计的原子。键以 "t" 连接过去和未来
        的标签 r、u1、u2、s，分别表示冗余、第一/第二列独有信息和协同。
        "rtr" 与 "sts" 分别为冗余到冗余、协同到协同；原子可为负值。

    Raises:
        ValueError: 数据形状、方法或时间间隔不合法，或出现常量列、滞后
            样本不足、非正定协方差。
    """
    lag = positive_int(lag, "lag")
    x = _prepare(data, method)
    if x.shape[1] != 2 or len(x) - lag < 2:
        raise ValueError("PhiID requires exactly two regions and at least two lagged observations")
    correction = method == "gc" if biascorrect is None else bool(biascorrect)
    return _phiid_prepared(x, lag, method, correction)


def phiid_matrices(data, *, lag=1, method="gaussian", biascorrect=None):
    """使用同一种估计器计算所有脑区对的 rtr 和 sts 连接矩阵。

    Args:
        data (array_like): 单次扫描的有限实数数组，形状为 (T, N)，N >= 2。
            样本与协方差要求同 phiid_pair，不应跨扫描拼接。
        lag (int): 正整数时间间隔，以保留采样点计，默认为 1。
        method (str): "gaussian"、"gc" 或 "discrete"，默认为 "gaussian"。
            两张图使用同一估计器；离散估计不自动二值化。
        biascorrect (bool | None): 是否校正高斯熵偏差。None 表示仅对 "gc"
            开启；离散估计忽略此项。

    Returns:
        dict[str, numpy.ndarray]: "PhiID-Red" 对应 rtr，"PhiID-Syn" 对应
        sts。两者均为 (N, N) 对称矩阵，对角线为零，单位为比特。

    Raises:
        ValueError: 参数或数据不合法、滞后样本不足，或相关协方差不正定。
    """
    lag = positive_int(lag, "lag")
    x = _prepare(data, method)
    if x.shape[1] < 2 or len(x) - lag < 2:
        raise ValueError("at least two regions and two lagged observations are required")
    correction = method == "gc" if biascorrect is None else bool(biascorrect)
    red = np.zeros((x.shape[1], x.shape[1]))
    syn = np.zeros_like(red)
    for i, j in combinations(range(x.shape[1]), 2):
        atoms = _phiid_prepared(x[:, [i, j]], lag, method, correction)
        red[i, j] = red[j, i] = atoms["rtr"]
        syn[i, j] = syn[j, i] = atoms["sts"]
    return {"PhiID-Red": red, "PhiID-Syn": syn}
