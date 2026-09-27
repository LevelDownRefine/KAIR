"""Independent dense/analytic oracles for geometric priors and proximal steps."""

import pytest
import torch

from models import marble_prior_solver
from models.marble_geometry_prior import (
    GeometryGraph,
    make_graph,
    pair_loss,
    sample_pairs,
)
from models.network_marble import MARBLEEncoder, contrastive_loss


def chain(device="cpu"):
    return GeometryGraph(
        torch.tensor([0, 1, 2], device=device),
        torch.tensor([1, 2, 3], device=device),
        torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64, device=device),
        4,
    )


def test_incidence_adjoint_and_dense_laplacian_agree():
    graph = chain()
    basis = torch.eye(4, dtype=torch.float64)
    dense = graph.laplace(basis)
    torch.testing.assert_close(
        graph.transpose.to_dense() @ graph.incidence.to_dense(), dense
    )
    torch.testing.assert_close(dense, dense.T)
    torch.testing.assert_close(dense.sum(1), torch.zeros(4, dtype=torch.float64))
    assert torch.linalg.eigvalsh(dense).max() <= 2
    value = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    graph.energy(value, "quadratic").backward()
    torch.testing.assert_close(value.grad, dense @ value / 4)


@pytest.mark.parametrize("strength", [0.0, 0.1, 1.0, 10.0])
def test_quadratic_prox_matches_independent_dense_solve(strength):
    graph = chain()
    torch.manual_seed(7)
    value = torch.randn(4, 3, dtype=torch.float64)
    actual, diagnostic = graph.quadratic_prox(value, strength, tolerance=1e-12)
    expected = torch.linalg.solve(
        torch.eye(4, dtype=torch.float64) + strength * graph.laplacian.to_dense(), value
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert diagnostic["linear_relative_residual"] < 1e-10


@pytest.mark.parametrize("strength", [0.05, 0.3, 4.0])
def test_vector_tv_matches_two_node_shrinkage(strength):
    graph = GeometryGraph(
        torch.tensor([0]), torch.tensor([1]), torch.ones(1, dtype=torch.float64), 2
    )
    value = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)
    actual, _, diagnostic = graph.tv_prox(value, strength, steps=500)
    delta = value[0] - value[1]
    shortened = delta * (1 - 2 * strength / delta.norm()).clamp_min(0)
    expected = torch.stack(
        [value.mean(0) + shortened / 2, value.mean(0) - shortened / 2]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    assert abs(diagnostic["tv_gap_per_node"]) < 1e-9


def test_convex_admm_reference_reaches_direct_map_solution():
    graph = chain()
    torch.manual_seed(8)
    observation = torch.randn(4, 3, dtype=torch.float64)
    auxiliary, dual = observation.clone(), torch.zeros_like(observation)
    for _ in range(100):
        primal = (observation + auxiliary - dual) / 2
        auxiliary, _ = graph.quadratic_prox(primal + dual, 0.7, tolerance=1e-12)
        dual += primal - auxiliary
    direct = torch.linalg.solve(
        torch.eye(4, dtype=torch.float64) + 0.7 * graph.laplacian.to_dense(),
        observation,
    )
    torch.testing.assert_close(primal, direct, atol=1e-10, rtol=1e-10)
    assert (primal - auxiliary).norm() < 1e-10


def test_pnp_preserves_constants_and_is_rotation_equivariant():
    graph = chain()
    constants = torch.ones(4, 3, dtype=torch.float64)
    torch.testing.assert_close(graph.adaptive_denoise(constants, 0.5), constants)
    values = torch.randn(4, 3, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    torch.testing.assert_close(
        graph.adaptive_denoise(values @ rotation, 0.5),
        graph.adaptive_denoise(values, 0.5) @ rotation,
    )


def test_pair_loss_matches_original_marble_logistic_loss():
    torch.manual_seed(12)
    values = torch.randn(12, 4, dtype=torch.float64)
    neighbors = torch.randint(12, (12, 3))
    positive, negative = sample_pairs(neighbors, 4)
    triples = torch.cat(
        [
            values.repeat_interleave(4, dim=0),
            values[positive].flatten(0, 1),
            values[negative].flatten(0, 1),
        ]
    )
    torch.testing.assert_close(
        pair_loss(values, positive, negative), contrastive_loss(triples)
    )


def test_neural_graph_is_invariant_to_block_rotation():
    torch.manual_seed(9)
    pos, velocity = (
        torch.randn(30, 3, dtype=torch.float64),
        torch.randn(30, 3, dtype=torch.float64),
    )
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    first, near, _ = make_graph(pos, velocity, neighbors=5)
    second, rotated_near, _ = make_graph(
        pos @ rotation, velocity @ rotation, neighbors=5
    )
    assert torch.equal(near, rotated_near)
    torch.testing.assert_close(first.laplacian.to_dense(), second.laplacian.to_dense())


def test_cuda_proximal_operators_match_cpu():
    assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    torch.manual_seed(1)
    values = torch.randn(4, 3, dtype=torch.float64)
    cpu, gpu = chain(), chain("cuda")
    first, _ = cpu.quadratic_prox(values, 1.0, tolerance=1e-12)
    second, _ = gpu.quadratic_prox(values.cuda(), 1.0, tolerance=1e-12)
    torch.testing.assert_close(first, second.cpu(), atol=1e-10, rtol=1e-10)
    first, _, _ = cpu.tv_prox(values, 0.1)
    second, _, _ = gpu.tv_prox(values.cuda(), 0.1)
    torch.testing.assert_close(first, second.cpu(), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", marble_prior_solver.METHODS)
def test_encoder_updates_keep_bn_frozen_and_queries_batch_independent(
    method, monkeypatch
):
    monkeypatch.setattr(marble_prior_solver, "OUTER_STEPS", 2)
    monkeypatch.setattr(marble_prior_solver, "INNER_STEPS", 2)
    torch.manual_seed(41)
    params = {
        "order": 1,
        "diffusion": False,
        "inner_product_features": False,
        "include_positions": True,
        "include_self": True,
        "vec_norm": False,
        "emb_norm": True,
        "bias": True,
        "frac_sampled_nb": -1,
        "batch_norm": "batch_norm",
        "dropout": 0.5,
        "dim_emb": 2,
        "dim_signal": 2,
        "hidden_channels": [8],
        "out_channels": 3,
    }
    model = MARBLEEncoder(params)
    features = torch.randn(30, 8)
    graph, neighbors, _ = make_graph(features[:, :2], features[:, 2:4], neighbors=5)
    buffers = {key: value.clone() for key, value in model.named_buffers()}
    before = model.enc.lins[0].weight.detach().clone()
    marble_prior_solver.optimize(
        model, features, graph, sample_pairs(neighbors, 0), method
    )
    assert not torch.equal(before, model.enc.lins[0].weight)
    for key, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)
    query = torch.randn(11, 8)
    torch.testing.assert_close(
        model(query),
        torch.cat([model(query[:5]), model(query[5:])]),
        atol=1e-6,
        rtol=1e-6,
    )
