"""H1 scaffolds and time-resolved simplicial features (NumPy only).

The deterministic F2 boundary reduction returns a particular cycle basis.
Scaffold weights depend on representatives, unlike persistence barcodes;
they are not guaranteed identical to the paper's Julia/Ripserer outputs.
"""

from dataclasses import dataclass

import numpy as np

from utils._hoi_common import multiplets, positive_int, standardize, timeseries
from utils.utils_hoi_information import TripletValues


@dataclass
class ScaffoldResult:
    frequency: np.ndarray
    persistence: np.ndarray
    intervals: np.ndarray
    cycles: tuple


@dataclass
class TemporalHOIResult:
    triangles: TripletValues
    scaffold: np.ndarray | None
    frames: np.ndarray


def _bit_indices(bits):
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


def scaffold_from_filtration(edge_filtration, *, triangles=None, triangle_filtration=None):
    """H1 frequency/persistence scaffolds of a filtered complete graph + faces.

    ``edge_filtration`` is a finite symmetric (N,N) birth-time matrix; diagonal
    ignored, all vertices present initially. By default add every triangle at
    its latest edge birth (Vietoris-Rips 2-skeleton). To omit faces, supply an
    explicit (K,3) ``triangles`` array, possibly empty. Custom triangle births
    must be no earlier than all their edges.

    Ties are ordered lexicographically, faces before cofaces. Only strictly
    positive-persistence H1 intervals contribute. Finite representatives are
    reduced triangle boundaries; essential representatives are birth cycles.
    Essential cycles count in frequency and contribute +inf to persistence.
    This reference implementation is intended for moderate N, not GPU scale.
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
    # Stable sort keeps the initial lexicographic order at equal weights.
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

    # Reduce d1, retaining change-of-basis columns for essential birth cycles.
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

    # Reduce d2 over F2 using Python integers as sparse packed bitsets.
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
    """Static Fscaff/Pscaff from the filtration 1 - Pearson FC."""
    x = timeseries(data)
    corr = np.corrcoef(x, rowvar=False)
    return scaffold_from_filtration(1 - (corr + corr.T) / 2)


def _product_stats(z, indices, chunk_size):
    means = np.empty(len(indices))
    scales = np.empty(len(indices))
    for start in range(0, len(indices), chunk_size):
        block = np.prod(z[:, indices[start:start + chunk_size]], axis=2)
        means[start:start + len(block.T)] = block.mean(axis=0)
        scales[start:start + len(block.T)] = block.std(axis=0)
    # A constant product has no temporal fluctuation: explicitly assign zero.
    return means, scales


def _coherent_weights(z_frame, indices, means, scales):
    samples = z_frame[indices]
    products = np.prod(samples, axis=1)
    centered = np.divide(products - means, scales, out=np.zeros_like(products), where=scales > 0)
    coherent = np.all(samples > 0, axis=1) | np.all(samples < 0, axis=1)
    return np.where(coherent, 1.0, -1.0) * np.abs(centered)


def temporal_higher_order(data, *, frames=None, edge_mode="standardized", chunk_size=256, scaffold=True):
    """Time-averaged violating Triangles and frequency TScaffold.

    Z-score each ROI over the full run. Z-score each triple product over time,
    then force its sign positive iff all three instantaneous ROI signs agree.
    ``edge_mode='standardized'`` applies the same rule to pair products, as in
    the authors' SimplicialTS code. ``'raw'`` uses z_i*z_j, as in the article's
    edge formula. These modes can produce different simplicial complexes.

    Descending weights define a filtration. Drop triangles whose weight is
    above any face edge. Triangles stores their positive weights (otherwise 0),
    averaged across selected frames. TScaffold averages H1 cycle frequencies
    of the remaining complexes. No threshold or frame subsampling is implicit.
    Product normalization always uses the full run, even when frames selected.
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
