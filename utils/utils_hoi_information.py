"""Information-theoretic brain connectivity, using NumPy/SciPy only.

Inputs are (time, regions); all information values are in bits. See
docs/README_HOI.md for estimators, definitions and reference implementations.
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
    triplets: np.ndarray
    values: np.ndarray
    n_regions: int


@dataclass
class PEDResult:
    triplets: np.ndarray
    redundancy: np.ndarray
    synergy: np.ndarray
    n_regions: int


def _prepare(data, method):
    if method not in ("gaussian", "gc", "discrete"):
        raise ValueError("method must be 'gaussian', 'gc' or 'discrete'")
    x = timeseries(data, min_regions=1, nonconstant=method != "discrete")
    if method == "gc":
        # Average ranks preserve ties rather than inventing an ordering.
        x = ndtri(rankdata(x, axis=0, method="average") / (len(x) + 1))
    return x


def _bias(n, d):
    return (np.sum(digamma((n - np.arange(1, d + 1)) / 2))
            + d * np.log(2 / (n - 1))) / (2 * np.log(2))


def _entropy_function(x, method, biascorrect):
    n = len(x)
    cov = None if method == "discrete" else np.atleast_2d(np.cov(x, rowvar=False))

    @lru_cache(None)
    def entropy(columns):
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
    """O-information of ALL columns: sum H(Xi)+(d-2)H(X)-sum H(X without Xi).

    Positive means redundancy dominated; negative means synergy dominated.
    Gaussian/copula estimation cannot detect arbitrary nonlinear dependence.
    ``discrete`` treats each distinct input value as a category (no binning).
    Bias correction defaults to True for gc, False otherwise.
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
    """Compute every (or selected) triplet's O-information in lexicographic order.

    Gaussian estimates use batched 3x3 covariance determinants, bounding
    intermediate storage by chunk_size rather than samples times all triplets.
    No regularization or clipping of negative information estimates is applied.
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
    """Redundancy lattice order on antichains of bitmask source sets."""
    return all(any(b & a == b for b in beta) for a in alpha)


@lru_cache(None)
def _ped_lattice():
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
    # Source indices 0,1,2 map to bitmasks 1,2,4.
    synergy = [(4, 3), (2, 5), (1, 6), (3, 5, 6), (5, 6), (3, 6), (3, 5)]
    syn_indices = [nodes.index(tuple(sorted(a))) for a in synergy]
    return tuple(nodes), union, np.linalg.inv(zeta), syn_indices


def ped_from_probabilities(probabilities):
    """Exact three-binary-variable h_sx PED, returning all 18 entropy atoms.

    Input is eight probabilities; index = X0 + 2*X1 + 4*X2. Keys are
    antichains of bitmask source sets (1=X0, 2=X1, 4=X2, 3={X0,X1}, ...).
    h_sx is the expected -log2 probability of the UNION of source events.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (8,) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("probabilities must be eight nonnegative finite values summing to one")
    nodes, union, inverse, _ = _ped_lattice()
    cumulative = -np.log2(np.maximum(union @ p, np.finfo(float).tiny)) @ p
    return dict(zip(nodes, inverse @ cumulative))


def ped_triplets(data, *, binary=False, triplets=None, chunk_size=256):
    """PED Red (bottom atom) and Syn (seven joint-source atoms) in bits.

    By default binarize each ROI at its temporal mean (>mean -> 1, else 0),
    equivalent to thresholding z-scored BOLD at zero. ``binary=True`` accepts
    only 0/1 observations, including constants. No joint-entropy normalization.
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
    # Mobius transform of the two-source diamond, on past and future axes.
    mobius = np.array([[1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 1, 0], [1, -1, -1, 1]])
    atoms = mobius @ cumulative @ mobius.T
    names = ("r", "u1", "u2", "s")
    return {f"{a}t{b}": float(atoms[i, j]) for i, a in enumerate(names) for j, b in enumerate(names)}


def phiid_pair(data, *, lag=1, method="gaussian", biascorrect=None):
    """All 16 MMI PhiID atoms for two ROI time series, in bits.

    rtr and sts are redundancy-to-redundancy and synergy-to-synergy.
    Their sum with the other atoms equals I(pair_past; pair_future).
    ``lag`` is in retained samples: never concatenate subjects or runs.
    gc normalizes full series before forming past/future pairs; each covariance
    is then centered. There is no automatic regularization or atom clipping.
    """
    lag = positive_int(lag, "lag")
    x = _prepare(data, method)
    if x.shape[1] != 2 or len(x) - lag < 2:
        raise ValueError("PhiID requires exactly two regions and at least two lagged observations")
    correction = method == "gc" if biascorrect is None else bool(biascorrect)
    return _phiid_prepared(x, lag, method, correction)


def phiid_matrices(data, *, lag=1, method="gaussian", biascorrect=None):
    """Symmetric zero-diagonal rtr and sts matrices for every ROI pair."""
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
