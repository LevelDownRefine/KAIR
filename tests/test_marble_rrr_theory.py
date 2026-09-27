"""Independent identities and counterexamples for the finite-dimensional proofs."""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from models.marble_rrr_theory import (
    choose_ridge,
    norm,
    ridge_perturbation_bound,
    rrr,
    spectral_certificate,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    torch.manual_seed(22)
    return request.param


def test_rrr_attains_independent_global_lower_bound(device):
    x = torch.randn(90, 8, dtype=torch.float64, device=device)
    y = torch.randn_like(x)
    covariance, cross = x.T @ x / len(x), x.T @ y / len(x)
    ridge = 0.07
    rank = 3
    solution = rrr(covariance, cross, rank, ridge)
    k = solution["coefficient"]
    actual = (x @ k - y).square().sum() / len(x) + ridge * k.square().sum()
    # Cholesky, rather than the solver's symmetric whitening, provides the oracle.
    a = covariance + ridge * torch.eye(8, dtype=x.dtype, device=device)
    cholesky = torch.linalg.cholesky(a)
    b = torch.linalg.solve_triangular(cholesky, cross, upper=False)
    optimum = y.square().sum() / len(y) - torch.linalg.svdvals(b)[:rank].square().sum()
    torch.testing.assert_close(actual, optimum, rtol=1e-11, atol=1e-11)
    arbitrary = torch.randn_like(k)
    objective = (x @ arbitrary - y).square().sum() / len(x)
    objective += ridge * arbitrary.square().sum()
    completed_square = (cholesky.T @ arbitrary - b).square().sum()
    completed_square += y.square().sum() / len(y) - b.square().sum()
    torch.testing.assert_close(objective, completed_square, rtol=1e-11, atol=1e-11)
    assert torch.linalg.matrix_rank(k) == rank


def test_ridge_condition_budget_even_with_rank_deficiency(device):
    c = torch.diag(
        torch.tensor([0.0, 1e-10, 1.0, 10.0], device=device, dtype=torch.float64)
    )
    ridge = choose_ridge(c)
    result = rrr(c, torch.eye(4, dtype=c.dtype, device=device), 2, ridge)
    assert float(result["condition"]) <= 100 * (1 + 1e-12)


def test_gradient_error_contracts_at_stated_rate(device):
    c = torch.diag(torch.tensor([0.1, 0.2, 1.0], device=device, dtype=torch.float64))
    a = c + 0.1 * torch.eye(3, device=device, dtype=c.dtype)
    t = torch.randn_like(a)
    optimum = torch.linalg.solve(a, t)
    k = torch.zeros_like(t)
    rate = 1 - torch.linalg.eigvalsh(a)[0] / torch.linalg.eigvalsh(a)[-1]
    for _ in range(30):
        previous = torch.linalg.vector_norm(k - optimum)
        k -= (a @ k - t) / torch.linalg.eigvalsh(a)[-1]
        assert torch.linalg.vector_norm(k - optimum) <= rate * previous + 1e-13


def test_ridge_resolvent_perturbation_bound(device):
    x = torch.randn(30, 5, device=device, dtype=torch.float64)
    y = x + 0.05 * torch.randn_like(x)
    c, changed_c = x.T @ x / len(x), y.T @ y / len(y)
    t = torch.randn_like(c)
    changed_t = t + 0.03 * torch.randn_like(t)
    identity = torch.eye(5, dtype=c.dtype, device=device)
    ridge = 0.2
    delta = norm(
        torch.linalg.solve(c + ridge * identity, t)
        - torch.linalg.solve(changed_c + ridge * identity, changed_t)
    )
    bound = ridge_perturbation_bound(c, t, changed_c, changed_t, ridge)
    assert delta <= bound


def test_spectral_certificate_and_degenerate_counterexample(device):
    h = torch.diag(
        torch.tensor([5.0, 3.0, 0.5, 0.1], device=device, dtype=torch.float64)
    )
    rotation, _ = torch.linalg.qr(
        torch.eye(4, device=device, dtype=h.dtype) + 1e-3 * torch.randn_like(h)
    )
    diagnostic = spectral_certificate(h, rotation @ h @ rotation.T, 2)
    assert diagnostic["nontrivial_certificate"]
    assert (
        diagnostic["actual_projector_distance"] <= diagnostic["projector_bound"] + 1e-12
    )
    repeated = torch.eye(4, device=device, dtype=h.dtype)
    assert not spectral_certificate(repeated, repeated, 2)["nontrivial_certificate"]


def test_subspace_iteration_rate(device):
    eigenvalues = torch.tensor([5.0, 3.0, 0.5, 0.1], device=device, dtype=torch.float64)
    h = torch.diag(eigenvalues)
    z = torch.cat(
        [
            torch.eye(2, device=device, dtype=h.dtype),
            0.1 * torch.randn(2, 2, device=device, dtype=h.dtype),
        ]
    )
    initial_tangent = norm(z[2:] @ torch.linalg.inv(z[:2]))
    for step in range(1, 6):
        z, _ = torch.linalg.qr(h @ z)
        tangent = norm(z[2:] @ torch.linalg.inv(z[:2]))
        assert (
            tangent
            <= (eigenvalues[2] / eigenvalues[1]) ** step * initial_tangent + 1e-13
        )


def test_dynamics_alone_cannot_identify_isotropic_coordinates(device):
    a = 0.8 * torch.eye(4, device=device, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn_like(a))
    torch.testing.assert_close(q @ a, a @ q, rtol=1e-12, atol=1e-12)
    assert float(norm(q - torch.eye(4, device=device, dtype=a.dtype))) > 0.1


def test_ols_predictions_rotate_but_uniform_r2_need_not():
    rng = np.random.default_rng(28)
    x = rng.normal(size=(100, 2))
    y = np.column_stack([10 * x[:, 0], rng.normal(size=len(x))])
    q = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    prediction = LinearRegression().fit(x, y).predict(x)
    rotated = LinearRegression().fit(x, y @ q).predict(x)
    np.testing.assert_allclose(rotated, prediction @ q, atol=1e-12)
    assert abs(r2_score(y, prediction) - r2_score(y @ q, rotated)) > 0.1
    np.testing.assert_allclose(
        r2_score(y, prediction, multioutput="variance_weighted"),
        r2_score(y @ q, rotated, multioutput="variance_weighted"),
        atol=1e-12,
    )
