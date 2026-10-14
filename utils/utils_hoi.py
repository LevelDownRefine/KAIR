"""Public entry point for the eleven higher-order brain connectivity maps.

No imaging preprocessing is performed here: supply one cleaned ROI time-series
matrix per run, shaped (time, regions). See docs/README_HOI.md for conventions.
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
    """Pearson FC, symmetric (N,N), including a unit diagonal."""
    return np.corrcoef(timeseries(data), rowvar=False)


def project_triplets(values, triplets, n_regions, *, target="edges", reduction="mean"):
    """Project aligned triplet scores onto edges or nodes.

    ``sum`` adds every incident triplet; ``mean`` divides by the number of
    supplied incident triplets, INCLUDING zeros; ``mean_nonzero`` reproduces
    the authors' optional nonzero-only normalization. Missing incidences are
    zero. Edge outputs are symmetric, diagonal zero. Order is never inferred.
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
    """Project positive and negative O-information separately; O-Syn <= 0."""
    return {
        "O-Red": project_triplets(np.maximum(result.values, 0), result.triplets, result.n_regions, target=target, reduction=reduction),
        "O-Syn": project_triplets(np.minimum(result.values, 0), result.triplets, result.n_regions, target=target, reduction=reduction),
    }


def edge_vector(matrix):
    """Strict upper triangle in np.triu_indices order (116 ROIs -> 6,670)."""
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or len(x) < 2:
        raise ValueError("matrix must be square with at least two regions")
    if not np.isfinite(x).all() or not np.allclose(x, x.T, rtol=1e-10, atol=1e-12):
        raise ValueError("matrix must be finite and symmetric")
    return x[np.triu_indices(len(x), 1)]


def compare_connectomes(matrices):
    """Return (names, Spearman rho matrix), comparing aligned ROI pairs.

    Input is a name->symmetric matrix mapping with identical ROI order/shape.
    Average ranks handle ties. Constant maps yield NaN, including their diagonal.
    These are descriptive similarities, not independent-edge significance tests.
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
    """Compute selected maps (or explicitly pass ``metrics=ALL_METRICS``).

    Raw triplet scores and full PhiID atoms are available from the individual
    functions. Defaults follow the authors' estimator choices (gc O-info/red,
    Gaussian PhiID synergy, binary h_sx PED); see documented reproduction limits.
    Topology is expensive: frames explicitly selects temporal output frames,
    never samples for FC, information measures or product normalization.
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
