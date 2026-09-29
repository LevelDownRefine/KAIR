"""Independent optimization and data-flow checks for the projection experiment."""

import numpy as np
import pytest
from utils.utils_dynamics_projection import DynamicsProjection


@pytest.fixture
def rates():
    rng = np.random.default_rng(42)
    innovations = rng.normal(size=(250, 8))
    return np.cumsum(innovations * np.linspace(0.2, 2, 8), axis=0)


def test_zero_alpha_agrees_with_full_svd_pca(rates):
    model = DynamicsProjection(3, 0).fit(rates)
    _, _, right = np.linalg.svd(rates - rates.mean(axis=0), full_matrices=False)
    np.testing.assert_allclose(
        model.basis_ @ model.basis_.T, right[:3].T @ right[:3], atol=1e-12
    )


def test_projection_objective_and_monotonic_tradeoff(rates):
    previous_state, previous_increment = -np.inf, np.inf
    rng = np.random.default_rng(7)
    for alpha in (0, 0.1, 1, 10):
        model = DynamicsProjection(3, alpha).fit(rates)
        stats = model.errors(rates)
        state = stats["state_residual_fraction"]
        increment = stats["increment_residual_fraction"]
        assert state >= previous_state - 1e-12
        assert increment <= previous_increment + 1e-12
        previous_state, previous_increment = state, increment
        np.testing.assert_allclose(model.basis_.T @ model.basis_, np.eye(3), atol=1e-12)
        optimum = state + alpha * increment
        # Compare against independently computed reconstruction on random subspaces.
        states = rates - rates.mean(axis=0)
        differences = np.diff(rates, axis=0)
        for _ in range(8):
            basis, _ = np.linalg.qr(rng.normal(size=(8, 3)))
            candidate = np.sum((states - states @ basis @ basis.T) ** 2)
            candidate /= np.sum(states**2)
            candidate += (
                alpha
                * np.sum((differences - differences @ basis @ basis.T) ** 2)
                / np.sum(differences**2)
            )
            assert optimum <= candidate + 1e-12
        # Ky Fan optimum: total spectral mass minus the leading q eigenvalues.
        assert optimum == pytest.approx(1 + alpha - model.eigenvalues_[:3].sum())


def test_heldout_transform_does_not_refit_and_increments_commute(rates):
    model = DynamicsProjection(3, 1).fit(rates[:200])
    mean, basis = model.mean_.copy(), model.basis_.copy()
    heldout = rates[200:] + 1000
    embedding = model.transform(heldout)
    np.testing.assert_array_equal(model.mean_, mean)
    np.testing.assert_array_equal(model.basis_, basis)
    np.testing.assert_allclose(embedding, (heldout - mean) @ basis)
    np.testing.assert_allclose(
        np.diff(embedding, axis=0), np.diff(heldout, axis=0) @ basis, atol=1e-12
    )


def test_trace_normalization_is_invariant_to_uniform_units(rates):
    original = DynamicsProjection(3, 1).fit(rates)
    rescaled = DynamicsProjection(3, 1).fit(1000 * rates + 25)
    np.testing.assert_allclose(original.basis_, rescaled.basis_, atol=1e-11)
    assert original.effective_lambda_ == pytest.approx(rescaled.effective_lambda_)


def test_bad_projection_inputs_rejected():
    with pytest.raises(AssertionError):
        DynamicsProjection(3, -1)
    with pytest.raises(AssertionError):
        DynamicsProjection(3).fit(np.ones((10, 5)))
    with pytest.raises(AssertionError):
        DynamicsProjection(3).transform(np.ones((10, 5)))
