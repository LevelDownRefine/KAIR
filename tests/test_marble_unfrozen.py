"""Check exact BN folding, complete gradient paths and matched training scopes."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.marble_geometry_prior import make_graph, sample_pairs
from models.marble_unfrozen import METHODS, FullObjective, TrainableMARBLE, train_full
from models.network_marble import LegacyEncoder


class Reference(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = LegacyEncoder([8, 5, 3]).double()
        with torch.no_grad():
            bn = self.enc.norms[0].module
            bn.running_mean.copy_(torch.linspace(-0.4, 0.3, 5))
            bn.running_var.copy_(torch.linspace(0.2, 2.0, 5))
            bn.weight.copy_(torch.tensor([1.0, -0.7, 0.3, 2.0, -1.0]))
            bn.bias.copy_(torch.linspace(-0.2, 0.5, 5))

    def forward(self, value):
        return F.normalize(self.enc(value), dim=-1)


def example(device="cpu"):
    torch.manual_seed(52)
    features = (torch.randn(20, 8, dtype=torch.float64) * torch.arange(1, 9) + 3).to(
        device
    )
    base = Reference().eval().to(device)
    model = TrainableMARBLE(base, features)
    graph, neighbors, _ = make_graph(features[:, :3], features[:, 3:], neighbors=3)
    return base, model, features, graph, sample_pairs(neighbors, 0)


def test_bn_and_feature_scaling_fold_exact_function_and_input_gradient():
    base, model, features, _, _ = example()
    features = features.requires_grad_()
    parameters = tuple(model.parameters())
    actual = model.encode(features, parameters, smooth=False)
    expected = base(features)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    weights = torch.randn_like(actual)
    expected_gradient = torch.autograd.grad((expected * weights).sum(), features)[0]
    actual_gradient = torch.autograd.grad((actual * weights).sum(), features)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12
    )


def test_training_statistics_are_immutable_and_inference_is_batch_independent():
    _, model, features, _, _ = example()
    saved = {name: value.clone() for name, value in model.named_buffers()}
    model.train()
    first = model(features[:4])
    combined = model(torch.cat([features[:4], features[4:] * 100]))[:4]
    torch.testing.assert_close(first, combined, atol=1e-14, rtol=1e-14)
    for name, value in model.named_buffers():
        assert torch.equal(value, saved[name])
    assert not any(isinstance(module, nn.BatchNorm1d) for module in model.modules())


def test_every_encoder_parameter_has_a_valid_smooth_gradient():
    _, model, features, _, _ = example()
    parameters = tuple(p.detach().requires_grad_() for p in model.parameters())
    assert torch.autograd.gradcheck(
        lambda *p: model.values(features[:2], p), parameters
    )
    loss = (model.values(features, parameters) * torch.arange(1, 4)).sum()
    gradients = torch.autograd.grad(loss, parameters)
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients[:4])


def test_frozen_control_is_restriction_of_identical_objective():
    _, model, features, graph, pairs = example()
    full = FullObjective(model, features, graph, pairs)
    frozen = FullObjective(model, features, graph, pairs, frozen=True)
    parameters = tuple(model.parameters())
    auxiliary = model(features).detach() * 0.9
    assert full.count == frozen.count
    torch.testing.assert_close(
        full.smooth(parameters, auxiliary),
        frozen.smooth(parameters[4:], auxiliary),
        atol=1e-14,
        rtol=1e-14,
    )
    torch.testing.assert_close(
        full.regularizer(parameters),
        frozen.regularizer(parameters[4:]),
        atol=1e-14,
        rtol=1e-14,
    )
    full_grad = torch.autograd.grad(full.smooth(parameters, auxiliary), parameters)[4:]
    frozen_grad = torch.autograd.grad(
        frozen.smooth(parameters[4:], auxiliary), parameters[4:]
    )
    for first, second in zip(full_grad, frozen_grad, strict=True):
        torch.testing.assert_close(first, second, atol=1e-13, rtol=1e-13)


def test_nonunit_ball_prox_matches_analytic_shrinkage_and_clipping():
    _, model, features, graph, pairs = example()
    problem = FullObjective(model, features, graph, pairs)
    problem.radii = (2.0, 3.0)
    matrix = torch.diag(torch.tensor([8.0, 0.4], dtype=torch.float64))
    vector = torch.tensor([6.0, 8.0], dtype=torch.float64)
    step = 0.2
    factor = 1 + step * problem.beta / problem.count
    result = problem.prox((matrix, vector), step)
    torch.testing.assert_close(
        result[0], torch.diag(torch.tensor([2.0, 0.4 / factor], dtype=torch.float64))
    )
    torch.testing.assert_close(result[1], torch.tensor([1.8, 2.4], dtype=torch.float64))


@pytest.mark.parametrize("method", METHODS)
def test_palm_updates_all_effective_weights_only_when_unfrozen(method):
    _, model, features, graph, pairs = example()
    _, diagnostic = train_full(model, features, graph, pairs, method, updates=12)
    changes = diagnostic["parameter_changes"]
    if method == "frozen_palm":
        assert changes[:4] == [0.0] * 4
    else:
        assert all(value > 0 for value in changes[:4])
        assert (
            diagnostic["optimized_parameter_count"]
            == diagnostic["total_parameter_count"]
        )
    assert all(
        row["majorization_margin"] >= 0 and row["descent_margin"] >= -1e-12
        for row in diagnostic["history"]
    )
    assert diagnostic["feasibility"]["maximum_relative_constraint_excess"] < 1e-12


@pytest.mark.parametrize("method", METHODS)
def test_cuda_full_encoder_updates_match_cpu(method):
    assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    _, cpu, features, graph, pairs = example()
    gpu = copy.deepcopy(cpu).cuda()
    graph_gpu, neighbors, _ = make_graph(
        features[:, :3].cuda(), features[:, 3:].cuda(), neighbors=3
    )
    _, cpu_d = train_full(cpu, features, graph, pairs, method, updates=4)
    _, gpu_d = train_full(
        gpu, features.cuda(), graph_gpu, sample_pairs(neighbors, 0), method, updates=4
    )
    assert [r["backtracks"] for r in cpu_d["history"]] == [
        r["backtracks"] for r in gpu_d["history"]
    ]
    torch.testing.assert_close(
        cpu(features), gpu(features.cuda()).cpu(), atol=1e-10, rtol=1e-10
    )
