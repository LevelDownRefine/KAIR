"""Independent projection, analytic solution, descent and device checks."""

import pytest
import torch

from models.marble_geometry_prior import GeometryGraph, make_graph, sample_pairs
from models.marble_provable_splitting import (
    METHODS,
    LatentObjective,
    SmoothResidual,
    feasibility,
    project_ball,
    split_step,
    stationarity,
    train_latent,
)


def test_projection_is_nearest_point_not_radial_cap():
    matrix = torch.tensor([[3.0, 0.0], [0.0, 0.4]], dtype=torch.float64)
    expected = torch.diag(torch.tensor([1.0, 0.4], dtype=torch.float64))
    actual = project_ball(matrix)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    assert (actual - matrix).norm() < (matrix / 3 - matrix).norm()
    # Variational inequality characterizes the convex projection independently.
    torch.manual_seed(20)
    for _ in range(20):
        feasible = torch.randn(2, 2, dtype=torch.float64)
        feasible = feasible / torch.linalg.matrix_norm(feasible, ord=2).clamp_min(1)
        assert ((matrix - actual) * (feasible - actual)).sum() <= 1e-14
    torch.testing.assert_close(
        project_ball(torch.tensor([3.0, 4.0])), torch.tensor([0.6, 0.8])
    )


class QuadraticReference:
    """F(x,v)=.5||x-a||² + .5||v||² + .5||v-x||² + .1||x||².

    Unconstrained minimizer x=a/1.7, v=x/2; reference also has a ball constraint.
    """

    auxiliary_step = 1 / 3

    def __init__(self, target):
        self.target = target
        self.nodes = target.numel()

    def smooth(self, parameters, auxiliary):
        value = parameters[0]
        return 0.5 * (
            (value - self.target).square().sum()
            + auxiliary.square().sum()
            + (auxiliary - value).square().sum()
        )

    def regularizer(self, parameters):
        return 0.1 * parameters[0].square().sum()

    def prox(self, parameters, step):
        return (project_ball(parameters[0] / (1 + 0.2 * step)),)


@pytest.mark.parametrize("method", ("palm", "joint"))
def test_splitting_reaches_analytic_quadratic_solution_with_descent(method):
    target = torch.tensor([0.6, -0.4], dtype=torch.float64)
    problem = QuadraticReference(target)
    parameters = (torch.tensor([-0.3, 0.2], dtype=torch.float64),)
    auxiliary = torch.ones_like(target)
    for _ in range(100):
        parameters, auxiliary, record = split_step(
            parameters, auxiliary, problem, method, maximum_step=0.4
        )
        assert record["majorization_margin"] >= 0
        assert record["descent_margin"] >= -1e-14
    torch.testing.assert_close(parameters[0], target / 1.7, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(auxiliary, target / 3.4, atol=1e-8, rtol=1e-8)
    assert (
        stationarity(parameters, auxiliary, problem)["parameter_proximal_mapping_norm"]
        < 1e-8
    )


def test_backtracking_does_not_hide_failure():
    problem = QuadraticReference(torch.ones(2, dtype=torch.float64))
    with pytest.raises(RuntimeError, match="no accepted step"):
        split_step(
            (torch.zeros(2, dtype=torch.float64),),
            torch.zeros(2, dtype=torch.float64),
            problem,
            "joint",
            maximum_step=1e6,
            max_backtracks=1,
        )


def test_dense_elimination_matches_moreau_envelope_and_auxiliary_stationarity():
    source = torch.tensor([0, 1])
    graph = GeometryGraph(source, source + 1, torch.ones(2, dtype=torch.float64), 3)
    laplacian = graph.laplacian.to_dense()
    values = torch.tensor([[0.2, 0.3], [0.6, -0.8], [-0.5, 0.1]], dtype=torch.float64)
    strength, eta = 0.7, 1.3
    system = eta * torch.eye(3, dtype=torch.float64) + strength * laplacian
    auxiliary = torch.linalg.solve(system, eta * values)
    force = strength * graph.laplace(auxiliary) + eta * (auxiliary - values)
    torch.testing.assert_close(force, torch.zeros_like(force), atol=1e-15, rtol=0)
    kernel = eta * strength * laplacian @ torch.linalg.inv(system)
    split = (
        strength * graph.energy(auxiliary, "quadratic")
        + eta * (auxiliary - values).square().sum() / 6
    )
    reduced = (values * (kernel @ values)).sum() / 6
    torch.testing.assert_close(split, reduced, atol=1e-15, rtol=1e-15)
    eigenvalues = torch.linalg.eigvalsh(laplacian)
    torch.testing.assert_close(
        torch.linalg.eigvalsh(kernel),
        eta * strength * eigenvalues / (eta + strength * eigenvalues),
        atol=1e-14,
        rtol=1e-14,
    )


def small_problem(device="cpu"):
    torch.manual_seed(30)
    anchor = torch.randn(18, 3, dtype=torch.float64, device="cpu").to(device)
    anchor = torch.nn.functional.normalize(anchor, dim=1)
    velocity = torch.randn(18, 3, dtype=torch.float64, device="cpu").to(device)
    graph, neighbors, _ = make_graph(anchor, velocity, neighbors=3)
    model = SmoothResidual(3).to(device)
    pairs = sample_pairs(neighbors, 0)
    return model, anchor, graph, pairs


def test_smooth_network_gradients_and_soft_normalization_at_zero():
    model = SmoothResidual(2)
    anchors = torch.tensor([[0.0, 0.0], [0.3, -0.5]], dtype=torch.float64)
    parameters = tuple(p.detach().requires_grad_() for p in model.parameters())
    assert torch.autograd.gradcheck(lambda *p: model.values(anchors, p), parameters)
    assert torch.isfinite(model(anchors)).all()
    assert torch.all(model(anchors).norm(dim=1) < 1)


def test_auxiliary_gradient_agrees_with_independent_formula():
    model, anchor, graph, pairs = small_problem()
    problem = LatentObjective(model, anchor, graph, pairs, 0.7, eta=1.3)
    auxiliary = (anchor * 0.8).requires_grad_()
    actual = torch.autograd.grad(
        problem.smooth(tuple(model.parameters()), auxiliary), auxiliary
    )[0]
    expected = (
        0.7 * graph.laplace(auxiliary) + 1.3 * (auxiliary - model(anchor))
    ) / len(anchor)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_latent_training_descends_and_preserves_constraints(method):
    model, anchor, graph, pairs = small_problem()
    saved = anchor.clone()
    before = model(anchor).detach()
    _, diagnostic = train_latent(model, anchor, graph, pairs, method, updates=20)
    assert not torch.equal(before, model(anchor))
    assert torch.equal(saved, anchor)
    assert feasibility(tuple(model.parameters()))["maximum_constraint_excess"] < 1e-12
    assert len(diagnostic["history"]) == 20
    assert all(
        r["majorization_margin"] >= 0 and r["descent_margin"] >= -1e-12
        for r in diagnostic["history"]
    )
    assert (
        diagnostic["history"][-1]["objective_after"]
        < diagnostic["history"][0]["objective_before"]
    )


@pytest.mark.parametrize("method", METHODS)
def test_cuda_splitting_matches_cpu(method):
    assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    outputs = []
    histories = []
    for device in ("cpu", "cuda"):
        model, anchor, graph, pairs = small_problem(device)
        auxiliary, diagnostic = train_latent(
            model, anchor, graph, pairs, method, updates=5
        )
        outputs.append((model(anchor).detach().cpu(), auxiliary.cpu()))
        histories.append([r["backtracks"] for r in diagnostic["history"]])
    assert histories[0] == histories[1]
    for cpu, gpu in zip(*outputs, strict=True):
        torch.testing.assert_close(cpu, gpu, atol=1e-11, rtol=1e-11)
