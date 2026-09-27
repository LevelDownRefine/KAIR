"""All effective encoder weights participate in a smooth, bounded PALM model."""

import torch
from torch import nn
from torch.nn import functional as F

from models.marble_geometry_prior import pair_loss
from models.marble_provable_splitting import (
    SmoothResidual,
    project_ball,
    split_step,
    stationarity,
)

METHODS = ("frozen_palm", "full_no_prior", "full_palm")


def soft_normalize(value, epsilon):
    return value / (value.square().sum(-1, keepdim=True) + epsilon**2).sqrt()


def parameter_norm(value):
    return torch.linalg.matrix_norm(value, ord=2) if value.ndim == 2 else value.norm()


class TrainableMARBLE(nn.Module):
    def __init__(self, base, features, delta=0.05, epsilon=0.1):
        super().__init__()
        assert not base.training and delta > 0 and epsilon > 0
        assert features.dtype == torch.float64
        self.delta, self.epsilon = delta, epsilon
        mean = features.mean(0)
        scale = (features - mean).square().mean(0).sqrt().clamp_min(1e-6)
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_scale", scale)
        with torch.no_grad():
            norm = base.enc.norms[0].module
            gain = norm.weight.double() / (norm.running_var.double() + norm.eps).sqrt()
            weight = gain[:, None] * base.enc.lins[0].weight.double()
            bias = gain * (base.enc.lins[0].bias.double() - norm.running_mean.double())
            bias = bias + norm.bias.double()
            self.encoder = nn.ParameterList(
                [
                    nn.Parameter(weight * scale[None, :]),
                    nn.Parameter(bias + weight @ mean),
                    nn.Parameter(base.enc.lins[1].weight.detach().double().clone()),
                    nn.Parameter(base.enc.lins[1].bias.detach().double().clone()),
                ]
            )
        self.latent = SmoothResidual(self.encoder[2].shape[0], epsilon).to(
            features.device
        )
        self.radii = (
            tuple(
                max(1.0, 1.25 * float(parameter_norm(p.detach()))) for p in self.encoder
            )
            + (1.0,) * 4
        )

    def preactivation(self, features, parameters, standardized=False):
        values = (
            features
            if standardized
            else (features - self.input_mean) / self.input_scale
        )
        return F.linear(values, *parameters[:2])

    def encode(self, features, parameters, smooth=True, standardized=False):
        hidden = self.preactivation(features, parameters, standardized)
        if smooth:
            hidden = (hidden + (hidden.square() + self.delta**2).sqrt()) / 2
        else:
            hidden = F.relu(hidden)
        raw = F.linear(hidden, *parameters[2:4])
        return soft_normalize(raw, self.epsilon) if smooth else F.normalize(raw, dim=-1)

    def values(self, features, parameters, standardized=False):
        anchor = self.encode(features, parameters, standardized=standardized)
        return self.latent.values(anchor, parameters[4:])

    def forward(self, features):
        return self.values(features, tuple(self.parameters()))


class FullObjective:
    def __init__(self, model, features, graph, pairs, frozen=False, strength=1.0):
        self.model, self.features, self.graph, self.pairs = (
            model,
            features,
            graph,
            pairs,
        )
        self.frozen, self.strength = frozen, strength
        self.nodes, self.eta, self.beta = len(features), 1.0, 1e-4
        self.count = sum(p.numel() for p in model.parameters())
        self.fixed = tuple(p.detach().clone() for p in model.encoder)
        self.radii = model.radii[4:] if frozen else model.radii
        self.inputs = (features - model.input_mean) / model.input_scale
        with torch.no_grad():
            self.frozen_anchors = model.encode(features, self.fixed) if frozen else None

    def expand(self, parameters):
        return self.fixed + tuple(parameters) if self.frozen else parameters

    def terms(self, parameters, auxiliary):
        if self.frozen:
            values = self.model.latent.values(self.frozen_anchors, parameters)
        else:
            values = self.model.values(self.inputs, parameters, standardized=True)
        contrastive = pair_loss(values, *self.pairs)
        geometry = self.strength * self.graph.energy(auxiliary, "quadratic")
        coupling = self.eta * (auxiliary - values).square().sum() / (2 * self.nodes)
        return contrastive, geometry, coupling

    def smooth(self, parameters, auxiliary):
        return sum(self.terms(parameters, auxiliary))

    def regularizer(self, parameters):
        return (
            self.beta
            * sum(p.square().sum() for p in self.expand(parameters))
            / (2 * self.count)
        )

    @torch.no_grad()
    def prox(self, parameters, step):
        return tuple(
            radius * project_ball(p / ((1 + step * self.beta / self.count) * radius))
            for p, radius in zip(parameters, self.radii, strict=True)
        )

    @property
    def auxiliary_step(self):
        return self.nodes / (self.eta + 2 * self.strength + 1)


@torch.no_grad()
def full_feasibility(parameters, radii):
    ratios = [
        float(parameter_norm(p)) / r for p, r in zip(parameters, radii, strict=True)
    ]
    return {
        "parameter_ball_ratios": ratios,
        "maximum_relative_constraint_excess": max(0.0, max(ratios) - 1),
    }


def train_full(model, features, graph, pairs, method, updates=300):
    assert method in METHODS and updates > 0
    frozen = method == "frozen_palm"
    problem = FullObjective(
        model, features, graph, pairs, frozen, float(method != "full_no_prior")
    )
    original = tuple(p.detach().clone() for p in model.parameters())
    parameters = original[4:] if frozen else original
    auxiliary = model(features).detach()
    initial_stationarity = stationarity(parameters, auxiliary, problem)
    history = []
    for iteration in range(updates):
        parameters, auxiliary, record = split_step(
            parameters, auxiliary, problem, "palm"
        )
        record["iteration"] = iteration + 1
        record["constraint_excess"] = full_feasibility(parameters, problem.radii)[
            "maximum_relative_constraint_excess"
        ]
        assert record["constraint_excess"] < 1e-12
        history.append(record)
    final = problem.expand(parameters)
    with torch.no_grad():
        for target, value in zip(model.parameters(), final, strict=True):
            target.copy_(value)
        changes = [
            float((before - after).norm())
            for before, after in zip(original, final, strict=True)
        ]
        if frozen:
            assert changes[:4] == [0.0] * 4
        else:
            assert all(change > 0 for change in changes[:4])
        terms = [float(v) for v in problem.terms(parameters, auxiliary)]
    return auxiliary, {
        "history": history,
        "initial_stationarity": initial_stationarity,
        "final_stationarity": stationarity(parameters, auxiliary, problem),
        "feasibility": full_feasibility(final, model.radii),
        "parameter_names": [name for name, _ in model.named_parameters()],
        "parameter_changes": changes,
        "parameter_radii": list(model.radii),
        "total_parameter_count": problem.count,
        "optimized_parameter_count": sum(p.numel() for p in parameters),
        "encoder_frozen": frozen,
        "final_contrastive": terms[0],
        "final_geometry": terms[1],
        "final_coupling": terms[2],
        "finite_budget_convergence_claimed": False,
    }
