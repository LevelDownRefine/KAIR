"""Scientific invariants, analytic distributions and filtration fixtures."""

from itertools import combinations, permutations, product
import ast
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import spearmanr

from utils import utils_hoi as hoi


REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "hoi_reference.json").read_text(encoding="utf-8"))


def test_ped_against_pinned_author_sxpid_reference():
    atoms = hoi.ped_from_probabilities(np.arange(1, 9, dtype=float) / 36)
    for key, expected in REFERENCE["ped_atoms_bits"].items():
        assert atoms[ast.literal_eval(key)] == pytest.approx(expected, abs=2e-14, rel=0)
    states = np.array([[(s >> b) & 1 for b in range(3)] for s in range(8)])
    result = hoi.ped_triplets(np.repeat(states, np.arange(1, 9), axis=0), binary=True)
    expected = REFERENCE["ped_atoms_bits"]
    synergy_keys = ["(3, 4)", "(2, 5)", "(1, 6)", "(3, 5, 6)", "(5, 6)", "(3, 6)", "(3, 5)"]
    assert_allclose(result.redundancy, expected["(1, 2, 4)"], atol=2e-14, rtol=0)
    assert_allclose(result.synergy, sum(expected[k] for k in synergy_keys), atol=2e-14, rtol=0)


def test_phiid_against_pinned_phyid_reference():
    x = np.random.default_rng(8).normal(size=(400, 2))
    for t in range(1, len(x)):
        x[t] += np.array([[.55, .2], [-.1, .4]]) @ x[t - 1]
    atoms = hoi.phiid_pair(x, lag=2)
    for key, expected in REFERENCE["phiid_atoms_bits"].items():
        assert atoms[key] == pytest.approx(expected, abs=2e-14, rel=0)


def exact_covariance(cov, samples=200):
    rng = np.random.default_rng(9)
    z = rng.normal(size=(samples, len(cov)))
    q, _ = np.linalg.qr(z - z.mean(axis=0))
    return q @ np.linalg.cholesky(cov).T * np.sqrt(samples - 1)


@pytest.mark.parametrize("kind, expected", [("independent", 0), ("copy", 1), ("xor", -1)])
def test_discrete_oinfo_analytic(kind, expected):
    if kind == "independent":
        x = np.array(list(product([0, 1], repeat=3)))
    elif kind == "copy":
        x = np.array([[0, 0, 0], [1, 1, 1]])
    else:
        x = np.array([[a, b, a ^ b] for a, b in product([0, 1], repeat=2)])
    assert hoi.o_information(x, method="discrete") == pytest.approx(expected)
    assert_allclose(hoi.oinfo_triplets(x, method="discrete").values, [expected])


@pytest.mark.parametrize("method", ["gaussian", "gc"])
def test_gaussian_triplets_match_entropy_identity(method):
    cov = np.full((5, 5), .4) + .6 * np.eye(5)
    x = exact_covariance(cov)
    result = hoi.oinfo_triplets(x, method=method, chunk_size=2)
    for triplet, value in zip(result.triplets, result.values):
        assert value == pytest.approx(hoi.o_information(x[:, triplet], method=method), abs=1e-12)
    if method == "gaussian":
        expected = .5 * np.log2(np.linalg.det(cov[:3, :3]) / (1 - .4 ** 2) ** 3)
        assert_allclose(result.values, expected, atol=1e-12)


def test_gc_monotonic_invariance_and_ties():
    x = np.random.default_rng(5).normal(size=(100, 4))
    assert_allclose(hoi.oinfo_triplets(x).values, hoi.oinfo_triplets(np.exp(x)).values)
    # Average ranks must not break identical columns into artificially distinct signals.
    tied = np.tile(np.arange(5), 10)
    with pytest.raises(ValueError, match="singular"):
        hoi.oinfo_triplets(np.column_stack((tied, tied, np.arange(50))))


def test_ped_copy_and_independent_entropy():
    p = np.zeros(8)
    p[[0, 7]] = .5
    atoms = hoi.ped_from_probabilities(p)
    assert len(atoms) == 18
    assert atoms[(1, 2, 4)] == pytest.approx(1)
    assert sum(atoms.values()) == pytest.approx(1)
    copy = hoi.ped_triplets([[0, 0, 0], [1, 1, 1]], binary=True)
    assert_allclose(copy.redundancy, 1)
    assert_allclose(copy.synergy, 0, atol=1e-12)
    independent = hoi.ped_from_probabilities(np.ones(8) / 8)
    assert sum(independent.values()) == pytest.approx(3)
    # h_sx is shared entropy, not PID redundancy about a separate target.
    assert independent[(1, 2, 4)] == pytest.approx(np.log2(8 / 7))


def test_ped_atoms_reconstruct_every_union_entropy():
    p = np.arange(1, 9, dtype=float) / 36
    atoms = hoi.ped_from_probabilities(p)
    states = np.arange(8)
    for alpha in atoms:
        cumulative = 0
        for v, probability in enumerate(p):
            event = np.zeros(8, dtype=bool)
            for subset in alpha:
                event |= (states & subset) == (v & subset)
            cumulative -= probability * np.log2(p[event].sum())
        below = sum(value for beta, value in atoms.items()
                    if all(any(b & a == b for b in beta) for a in alpha))
        assert below == pytest.approx(cumulative, abs=1e-12)


def test_ped_batch_permutation_and_threshold():
    x = np.random.default_rng(42).normal(size=(53, 5))
    result = hoi.ped_triplets(x, chunk_size=2)
    binary = hoi.ped_triplets(x > x.mean(axis=0), binary=True, chunk_size=50)
    assert_allclose(result.redundancy, binary.redundancy)
    assert_allclose(result.synergy, binary.synergy)
    for permutation in permutations(range(3)):
        perm = hoi.ped_triplets(x[:, permutation])
        assert_allclose(perm.redundancy, result.redundancy[:1])
        assert_allclose(perm.synergy, result.synergy[:1])


def test_phiid_discrete_perfect_redundant_memory():
    bit = np.arange(21) % 2
    result = hoi.phiid_pair(np.column_stack((bit, bit)), method="discrete")
    assert len(result) == 16
    assert result["rtr"] == pytest.approx(1)
    assert result["sts"] == pytest.approx(0)
    assert sum(result.values()) == pytest.approx(1)


def test_phiid_joint_information_conservation_and_region_swap():
    x = np.random.default_rng(8).normal(size=(400, 2))
    for t in range(1, len(x)):
        x[t] += np.array([[.55, .2], [-.1, .4]]) @ x[t - 1]
    lag = 2
    atoms = hoi.phiid_pair(x, lag=lag)
    joint = np.column_stack((x[:-lag], x[lag:]))
    cov = np.cov(joint, rowvar=False)
    expected = .5 * np.log2(np.linalg.det(cov[:2, :2]) * np.linalg.det(cov[2:, 2:]) / np.linalg.det(cov))
    assert sum(atoms.values()) == pytest.approx(expected, abs=1e-12)
    swapped = hoi.phiid_pair(x[:, ::-1], lag=lag)
    for key in ("rtr", "sts"):
        assert atoms[key] == pytest.approx(swapped[key], abs=1e-12)
    mi = [-.5 * np.log2(1 - np.corrcoef(joint[:, i], joint[:, j])[0, 1] ** 2)
          for i in (0, 1) for j in (2, 3)]
    assert atoms["rtr"] == pytest.approx(min(mi), abs=1e-12)
    matrices = hoi.phiid_matrices(x, lag=lag)
    assert_allclose(matrices["PhiID-Syn"], [[0, atoms["sts"]], [atoms["sts"], 0]])


def square_filtration():
    matrix = np.full((4, 4), 2.)
    np.fill_diagonal(matrix, 0)
    for i, j in ((0, 1), (1, 2), (2, 3), (0, 3)):
        matrix[i, j] = matrix[j, i] = 1
    return matrix


def test_static_square_cycle_and_filled_triangle():
    result = hoi.scaffold_from_filtration(square_filtration())
    assert_allclose(result.intervals, [[1, 2]])
    expected = (square_filtration() == 1).astype(float)
    assert_allclose(result.frequency, expected)
    assert_allclose(result.persistence, expected)
    filled = hoi.scaffold_from_filtration(np.ones((3, 3)) - np.eye(3))
    assert filled.intervals.shape == (0, 2)
    assert not filled.frequency.any()


def test_essential_cycles_and_delayed_filling():
    edges = np.ones((3, 3)) - np.eye(3)
    essential = hoi.scaffold_from_filtration(edges, triangles=np.empty((0, 3), dtype=int))
    assert_allclose(essential.intervals, [[1, np.inf]])
    assert_allclose(essential.frequency, edges)
    assert np.isinf(essential.persistence[np.triu_indices(3, 1)]).all()
    delayed = hoi.scaffold_from_filtration(edges, triangle_filtration=[3.])
    assert_allclose(delayed.intervals, [[1, 3]])
    assert_allclose(delayed.persistence, edges * 2)
    with pytest.raises(ValueError, match="triangle births"):
        hoi.scaffold_from_filtration(edges, triangle_filtration=[.5])


def test_scaffold_cycles_are_closed_and_barcode_permutation_invariant():
    rng = np.random.default_rng(19)
    matrix = rng.uniform(size=(8, 8))
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, 0)
    result = hoi.scaffold_from_filtration(matrix)
    for cycle, interval in zip(result.cycles, result.intervals):
        degree = np.zeros(8, dtype=int)
        for i, j in cycle:
            degree[i] += 1
            degree[j] += 1
            assert matrix[i, j] <= interval[0]
        assert np.all(degree % 2 == 0)
    permutation = rng.permutation(8)
    permuted = hoi.scaffold_from_filtration(matrix[np.ix_(permutation, permutation)])
    assert_allclose(permuted.intervals, result.intervals)


@pytest.mark.parametrize("edge_mode", ["raw", "standardized"])
def test_temporal_triangles_against_explicit_product_calculation(edge_mode):
    x = np.random.default_rng(22).normal(size=(40, 4))
    frames = [2, 4, 7]
    result = hoi.temporal_higher_order(x, frames=frames, edge_mode=edge_mode, chunk_size=1)
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    expected = []
    for t in result.triangles.triplets:
        prod = np.prod(z[:, t], axis=1)
        tw = np.abs((prod - prod.mean()) / prod.std())
        coherent = np.all(z[:, t] > 0, axis=1) | np.all(z[:, t] < 0, axis=1)
        tw *= np.where(coherent, 1, -1)
        ew = []
        for i, j in combinations(t, 2):
            e = z[:, i] * z[:, j]
            if edge_mode == "standardized":
                e = np.abs((e - e.mean()) / e.std()) * np.where(z[:, i] * z[:, j] > 0, 1, -1)
            ew.append(e)
        violating = tw > np.min(ew, axis=0)
        expected.append(np.where(violating & (tw > 0), tw, 0)[frames].mean())
    assert_allclose(result.triangles.values, expected)
    single = [hoi.temporal_higher_order(x, frames=[f], edge_mode=edge_mode) for f in frames]
    assert_allclose(result.scaffold, np.mean([s.scaffold for s in single], axis=0))
    assert_allclose(result.triangles.values, np.mean([s.triangles.values for s in single], axis=0))


def test_projection_preserves_signed_channels_and_denominators():
    triplets = np.array([[0, 1, 2], [0, 1, 3]])
    raw = hoi.TripletValues(triplets, np.array([2., -4.]), 4)
    maps = hoi.split_oinfo(raw)
    assert maps["O-Red"][0, 1] == 1
    assert maps["O-Syn"][0, 1] == -2
    nonzero = hoi.split_oinfo(raw, reduction="mean_nonzero")
    assert nonzero["O-Red"][0, 1] == 2
    assert nonzero["O-Syn"][0, 1] == -4
    summed = hoi.project_triplets(raw.values, triplets, 4, reduction="sum")
    assert summed[0, 1] == -2
    assert_allclose(np.diag(summed), 0)
    assert_allclose(hoi.project_triplets(raw.values, triplets, 4, target="nodes"), [-1, -1, 2, -4])


def test_edge_order_and_spearman_with_ties_and_constants():
    a = np.zeros((116, 116))
    a[np.triu_indices(116, 1)] = np.arange(6670)
    a += a.T
    assert_array_equal(hoi.edge_vector(a), np.arange(6670))
    x = np.array([[0, 1, 1], [1, 0, 3], [1, 3, 0.]])
    y = np.array([[0, 3, 2], [3, 0, 1], [2, 1, 0.]])
    names, rho = hoi.compare_connectomes({"x": x, "y": y, "constant": np.zeros((3, 3))})
    assert names == ("x", "y", "constant")
    assert rho[0, 1] == pytest.approx(spearmanr([1, 1, 3], [3, 2, 1]).statistic)
    assert np.isnan(rho[2]).all()
    assert np.isnan(rho[:, 2]).all()


def test_all_eleven_maps_and_input_not_mutated():
    x = np.random.default_rng(44).normal(size=(100, 5))
    original = x.copy()
    maps = hoi.compute_hoi(x, metrics=hoi.ALL_METRICS, frames=[0, 1], chunk_size=3)
    assert tuple(maps) == hoi.ALL_METRICS
    for name, matrix in maps.items():
        assert matrix.shape == (5, 5)
        assert np.isfinite(matrix).all()
        assert_allclose(matrix, matrix.T)
        assert_allclose(np.diag(matrix), 1 if name == "FC" else 0)
    assert_array_equal(x, original)
    assert_allclose(maps["Fscaff"], hoi.correlation_scaffolds(x).frequency)
    assert_allclose(maps["PhiID-Red"], hoi.phiid_matrices(x, method="gc")["PhiID-Red"])
    assert_allclose(maps["PhiID-Syn"], hoi.phiid_matrices(x, method="gaussian")["PhiID-Syn"])


@pytest.mark.parametrize("bad", [np.ones((10, 3)), [[1, np.nan], [2, 3]], [[1, np.inf], [2, 3]], [1, 2, 3], np.ones((2, 3, 4))])
def test_invalid_input_rejected(bad):
    with pytest.raises(ValueError):
        hoi.functional_connectivity(bad)


@pytest.mark.parametrize("lag", [0, -1, 1.5, True, 99])
def test_invalid_lag_rejected(lag):
    with pytest.raises(ValueError):
        hoi.phiid_pair(np.random.default_rng(1).normal(size=(20, 2)), lag=lag)


@pytest.mark.parametrize("triplets", [[[0, 0, 1]], [[0, 1, 5]], [[0., 1., 2.]], [[0, 1, 2], [0, 1, 2]]])
def test_invalid_triplets_rejected(triplets):
    with pytest.raises(ValueError):
        hoi.oinfo_triplets(np.random.default_rng(1).normal(size=(20, 4)), triplets=triplets)


def test_invalid_probability_matrix_and_metric_options():
    with pytest.raises(ValueError):
        hoi.ped_from_probabilities(np.ones(8))
    with pytest.raises(ValueError):
        hoi.edge_vector([[0, 1], [2, 0]])
    with pytest.raises(ValueError):
        hoi.compute_hoi(np.ones((20, 4)), metrics=["not-a-metric"])
    with pytest.raises(ValueError):
        hoi.compute_hoi(np.ones((20, 4)), metrics="FC")
    with pytest.raises(ValueError):
        hoi.phiid_pair(np.arange(12).reshape(4, 3))
    with pytest.raises(ValueError):
        hoi.temporal_higher_order(np.random.default_rng(1).normal(size=(20, 4)), frames=[0, 0])
