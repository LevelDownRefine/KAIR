"""Analytic checks for the spectral operator and anchored fixed-point network."""

import pytest
import torch

import main_explore_marble_spectral as experiment
from models.marble_geometry_prior import make_graph, sample_pairs
from models.network_marble_spectral import FrozenMARBLESpectral, SpectralLatentLayer


def test_hard_cap_uses_true_operator_norm():
    model = SpectralLatentLayer(4).double()
    with torch.no_grad():
        model.first.weight.mul_(7)
        model.second.weight.mul_(3)
    assert model.bounds()["operator_lipschitz_upper_bound"] > 2
    model.cap_weights()
    bounds = model.bounds()
    assert bounds["operator_lipschitz_upper_bound"] < 1
    assert max(bounds["weight_norms"]) <= 0.999 + 1e-12


def test_soft_penalty_has_gradient_but_does_not_enforce_constraint():
    model = SpectralLatentLayer(3).double()
    with torch.no_grad():
        model.first.weight.mul_(2)
    penalty = model.spectral_penalty()
    penalty.backward()
    assert penalty > 0 and model.first.weight.grad.norm() > 0
    assert not model.bounds()["half_strict_pseudocontractive_certified"]


def test_strict_pseudocontractive_inequality_and_nonexpansiveness_are_distinct():
    # N=-tanh gives D=2N-I with derivative close to -3, yet k=1/2 works.
    model = SpectralLatentLayer(2).double()
    with torch.no_grad():
        model.second.weight.neg_()
    torch.manual_seed(1)
    x, y = torch.randn(40, 2) * 0.1, torch.randn(40, 2) * 0.1
    x, y = x.double(), y.double()
    delta = x - y
    mapped = model.denoise(x) - model.denoise(y)
    left = mapped.square().sum(1)
    right = delta.square().sum(1) + 0.5 * (delta - mapped).square().sum(1)
    assert torch.all(left <= right + 1e-12)
    assert torch.any(left > delta.square().sum(1))


def test_analytic_jacobian_matches_autograd():
    torch.manual_seed(2)
    model = SpectralLatentLayer(3).double()
    point = torch.randn(3, dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(model.operator, point)
    reported = model.jacobian_diagnostics(point[None])
    assert reported["sampled_N_jacobian_norm_max"] == pytest.approx(
        float(torch.linalg.matrix_norm(jacobian, ord=2)), abs=1e-12
    )


def test_fixed_point_certificate_and_bilipschitz_bound():
    torch.manual_seed(3)
    model = SpectralLatentLayer(3).double()
    anchors = torch.randn(30, 3, dtype=torch.float64)
    solution, diagnostic = model.solve(anchors, tolerance=1e-8)
    reference, _ = model.solve(anchors, tolerance=1e-13)
    actual = (solution - reference).norm(dim=1).max()
    assert actual <= diagnostic["a_posteriori_error_bound"] + 1e-12
    before = (anchors[:15] - anchors[15:]).norm(dim=1)
    after = (reference[:15] - reference[15:]).norm(dim=1)
    assert torch.all(after <= before + 1e-12)
    assert torch.all(after >= before / 3 - 1e-12)
    split, _ = model.solve(anchors[:15], tolerance=1e-13)
    torch.testing.assert_close(split, reference[:15], atol=1e-12, rtol=1e-12)


def test_mann_form_equals_anchored_iteration():
    model = SpectralLatentLayer(3).double()
    values, anchor = torch.randn(2, 5, 3, dtype=torch.float64)
    full_map = model.denoise(values) - 2 * (values - anchor)
    torch.testing.assert_close(
        0.75 * values + 0.25 * full_map, model.step(values, anchor)
    )


def test_solver_reports_cap_without_claiming_convergence():
    model = SpectralLatentLayer(3).double()
    _, diagnostic = model.solve(torch.ones(4, 3).double(), tolerance=1e-14, max_steps=1)
    assert not diagnostic["tolerance_met"]
    assert diagnostic["steps"] == 1


def test_frozen_wrapper_trains_latent_only():
    base = torch.nn.Linear(4, 3)
    model = FrozenMARBLESpectral(base, SpectralLatentLayer(3))
    features = torch.randn(10, 4)
    model.train()
    model(features).square().mean().backward()
    assert not base.training
    assert all(p.grad is None for p in base.parameters())
    assert model.latent.first.weight.grad.norm() > 0


def test_cuda_spectral_network_matches_cpu():
    assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    torch.manual_seed(4)
    cpu = SpectralLatentLayer(3).double()
    gpu = SpectralLatentLayer(3).double().cuda()
    gpu.load_state_dict(cpu.state_dict())
    values = torch.randn(20, 3, dtype=torch.float64)
    first, _ = cpu.solve(values, tolerance=1e-10)
    second, _ = gpu.solve(values.cuda(), tolerance=1e-10)
    torch.testing.assert_close(first, second.cpu(), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", experiment.METHODS)
def test_fit_updates_latent_weights_and_preserves_anchors(method, monkeypatch):
    monkeypatch.setattr(experiment, "UPDATES", 10)
    torch.manual_seed(11)
    anchor = torch.nn.functional.normalize(torch.randn(25, 3), dim=1)
    saved = anchor.clone()
    graph, neighbors, _ = make_graph(anchor, torch.randn_like(anchor), neighbors=4)
    model = SpectralLatentLayer(3)
    before = model.first.weight.detach().clone()
    history = experiment.train_latent(
        model, anchor, graph, sample_pairs(neighbors, 0), method
    )
    assert len(history) == 1
    assert not torch.equal(before, model.first.weight)
    assert torch.equal(anchor, saved)
    if method == "hard_spectral":
        assert model.bounds()["half_strict_pseudocontractive_certified"]
