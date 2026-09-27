"""GW against its explicit four-index sum and an independent autograd oracle."""

import torch

from models.network_marble_experiment import (
    MathematicalModel,
    gw_gradient,
    gw_loss,
    sinkhorn,
    solve_gw,
)


def test_gw_objective_and_gradient_match_four_index_definition():
    torch.manual_seed(2)
    left = torch.cdist(
        torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    )
    left = (left + left.T) / 2
    right = torch.cdist(
        torch.randn(6, 3, dtype=torch.float64), torch.randn(6, 3, dtype=torch.float64)
    )
    right = (right + right.T) / 2
    coupling = torch.rand(5, 6, dtype=torch.float64, requires_grad=True)
    literal = (
        (left[:, None, :, None] - right[None, :, None, :]).square()
        * coupling[:, :, None, None]
        * coupling[None, None, :, :]
    ).sum()
    literal.backward()
    torch.testing.assert_close(
        gw_loss(left, right, coupling), literal, rtol=1e-10, atol=1e-10
    )
    torch.testing.assert_close(
        gw_gradient(left, right, coupling), coupling.grad, rtol=1e-10, atol=1e-10
    )


def test_sinkhorn_has_both_prescribed_marginals():
    cost = torch.tensor([[0, 1], [1, 0], [0.4, 0.7]], dtype=torch.float64)
    p, q = (
        torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64),
        torch.tensor([0.4, 0.6], dtype=torch.float64),
    )
    coupling = sinkhorn(cost, p, q)
    torch.testing.assert_close(coupling.sum(1), p, rtol=0, atol=1e-8)
    torch.testing.assert_close(coupling.sum(0), q, rtol=0, atol=1e-8)


def test_sinkhorn_rounding_records_unconverged_inner_solve():
    cost = torch.tensor([[0, 1], [1, 0]], dtype=torch.float64)
    p = torch.tensor([0.2, 0.8], dtype=torch.float64)
    q = torch.tensor([0.6, 0.4], dtype=torch.float64)
    diagnostic = []
    coupling = sinkhorn(cost, p, q, max_steps=1, diagnostics=diagnostic)
    torch.testing.assert_close(coupling.sum(1), p, rtol=0, atol=1e-14)
    torch.testing.assert_close(coupling.sum(0), q, rtol=0, atol=1e-14)
    assert not diagnostic[0]["converged_before_rounding"]
    assert diagnostic[0]["rounding_l1"] > 0


def test_sinkhorn_matches_analytic_symmetric_solution():
    p = torch.full((2,), 0.5, dtype=torch.float64)
    cost = torch.tensor([[0, 0.1], [0.1, 0]], dtype=torch.float64)
    coupling = sinkhorn(cost, p, p)
    diagonal = 0.5 / (1 + torch.exp(torch.tensor(-2.0, dtype=torch.float64)))
    expected = torch.stack([diagonal, 0.5 - diagonal]).repeat(2, 1)
    expected[1] = expected[1].flip(0)
    torch.testing.assert_close(coupling, expected, rtol=1e-12, atol=1e-12)


def test_gw_damped_objective_is_nonincreasing():
    torch.manual_seed(3)
    data = torch.randn(8, 3, dtype=torch.float64)
    left = torch.cdist(data, data).square()
    left = left / left.mean()
    right = left.flip((0, 1))
    coupling, diagnostic = solve_gw(left, right)
    history = diagnostic["objective_history"]
    assert all(b <= a + 1e-12 for a, b in zip(history[:-1], history[1:]))
    assert diagnostic["marginal_max_error"] < 1e-8
    assert torch.isfinite(coupling).all()
    identity = torch.eye(8, dtype=torch.float64).flip(1) / 8
    assert abs(float(gw_loss(left, right, identity))) < 1e-12


def test_reference_keeps_identity_and_does_not_fit_test_data():
    data = torch.randn(40, 3, dtype=torch.float64)
    model = MathematicalModel(16).fit(data, 3, 0, reference=data.clone())
    query = torch.randn(11, 3, dtype=torch.float64)
    torch.testing.assert_close(model.transform(query), query, rtol=0, atol=0)
