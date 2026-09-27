"""Smooth latent model and exact proximal splits of a fixed penalized objective.

See docs/MARBLE_SPLITTING_THEORY.md for the exact-arithmetic convergence scope.
Float64 diagnostics check each accepted step; a finite budget is not convergence.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.marble_geometry_prior import pair_loss

METHODS = ("palm_no_prior", "joint_quadratic", "palm_quadratic")


@torch.no_grad()
def project_ball(value):
    """Euclidean projection: singular-value clipping, not radial matrix scaling."""
    if value.ndim == 2:
        left, singular, right = torch.linalg.svd(value, full_matrices=False)
        return (left * singular.clamp_max(1)) @ right
    assert value.ndim == 1
    return value / value.norm().clamp_min(1)


class SmoothResidual(nn.Module):
    def __init__(self, channels, epsilon=0.1):
        super().__init__()
        assert channels > 0 and epsilon > 0
        self.epsilon = epsilon
        self.first = nn.Linear(channels, 2 * channels, dtype=torch.float64)
        self.second = nn.Linear(2 * channels, channels, dtype=torch.float64)
        with torch.no_grad():
            eye = torch.eye(channels, dtype=torch.float64)
            self.first.weight.copy_(torch.cat([eye, -eye]) * (0.5 / math.sqrt(2)))
            self.first.bias.zero_()
            self.second.weight.zero_()
            self.second.bias.zero_()

    def values(self, anchors, parameters):
        first, bias, second, offset = parameters
        raw = anchors + F.linear(
            torch.tanh(F.linear(anchors, first, bias)), second, offset
        )
        return raw / (raw.square().sum(-1, keepdim=True) + self.epsilon**2).sqrt()

    def forward(self, anchors):
        return self.values(anchors, tuple(self.parameters()))


class LatentObjective:
    def __init__(self, model, anchors, graph, pairs, strength, eta=1.0, beta=1e-4):
        assert strength >= 0 and eta > 0 and beta > 0
        assert anchors.dtype == torch.float64
        self.model, self.anchors, self.graph, self.pairs = model, anchors, graph, pairs
        self.strength, self.eta, self.beta = strength, eta, beta
        self.count = sum(p.numel() for p in model.parameters())
        self.nodes = len(anchors)

    def terms(self, parameters, auxiliary):
        values = self.model.values(self.anchors, parameters)
        contrastive = pair_loss(values, *self.pairs)
        geometry = self.strength * self.graph.energy(auxiliary, "quadratic")
        coupling = self.eta * (auxiliary - values).square().sum() / (2 * self.nodes)
        return contrastive, geometry, coupling

    def smooth(self, parameters, auxiliary):
        return sum(self.terms(parameters, auxiliary))

    def regularizer(self, parameters):
        return self.beta * sum(p.square().sum() for p in parameters) / (2 * self.count)

    @torch.no_grad()
    def prox(self, parameters, step):
        return tuple(
            project_ball(p / (1 + step * self.beta / self.count)) for p in parameters
        )

    @property
    def auxiliary_step(self):
        return self.nodes / (self.eta + 2 * self.strength + 1)


def squared_norm(values):
    return sum(value.square().sum() for value in values)


def inner(first, second):
    return sum((a * b).sum() for a, b in zip(first, second, strict=True))


def split_step(
    parameters,
    auxiliary,
    problem,
    method,
    maximum_step=16.0,
    margin=0.1,
    max_backtracks=40,
):
    """One PALM or synchronous PFB sweep; retain every acceptance/descent margin.

    problem supplies smooth, regularizer, prox and a valid auxiliary_step.
    No finite nonlinear subproblem solver is hidden inside either update.
    """
    assert method in ("palm", "joint") and maximum_step > 0 and 0 < margin < 1
    parameters = tuple(p.detach().requires_grad_() for p in parameters)
    auxiliary = auxiliary.detach().requires_grad_()
    before = problem.smooth(parameters, auxiliary)
    gradients = torch.autograd.grad(before, (*parameters, auxiliary))
    parameter_gradient, auxiliary_gradient = gradients[:-1], gradients[-1]
    assert all(torch.isfinite(g).all() for g in gradients)
    with torch.no_grad():
        objective_before = before + problem.regularizer(parameters)
        for backtracks in range(max_backtracks):
            step = maximum_step * 0.5**backtracks
            auxiliary_step = problem.auxiliary_step * (
                0.5**backtracks if method == "joint" else 1
            )
            candidate = problem.prox(
                tuple(
                    p - step * g
                    for p, g in zip(parameters, parameter_gradient, strict=True)
                ),
                step,
            )
            delta = tuple(
                new - old for new, old in zip(candidate, parameters, strict=True)
            )
            square = squared_norm(delta)
            trial_auxiliary = (
                auxiliary - auxiliary_step * auxiliary_gradient
                if method == "joint"
                else auxiliary
            )
            auxiliary_delta = trial_auxiliary - auxiliary
            auxiliary_square = auxiliary_delta.square().sum()
            linear = inner(parameter_gradient, delta)
            if method == "joint":
                linear = linear + (auxiliary_gradient * auxiliary_delta).sum()
            quadratic = square / step + auxiliary_square / auxiliary_step
            trial_value = problem.smooth(candidate, trial_auxiliary)
            acceptance = before + linear + (1 - margin) * quadratic / 2 - trial_value
            if torch.isfinite(trial_value) and acceptance >= 0:
                break
        else:
            raise RuntimeError(
                f"{method}: no accepted step after {max_backtracks} trials"
            )
    if method == "palm":
        fresh_gradient = torch.autograd.grad(
            problem.smooth(candidate, auxiliary), auxiliary
        )[0]
        with torch.no_grad():
            trial_auxiliary = auxiliary - auxiliary_step * fresh_gradient
            auxiliary_square = (trial_auxiliary - auxiliary).square().sum()
    with torch.no_grad():
        objective_after = problem.smooth(
            candidate, trial_auxiliary
        ) + problem.regularizer(candidate)
        bound = margin * square / (2 * step)
        bound = bound + (margin if method == "joint" else 1) * auxiliary_square / (
            2 * auxiliary_step
        )
        descent_margin = objective_before - objective_after - bound
        assert torch.isfinite(objective_after)
        # The theorem is in exact arithmetic. Keep and report sub-ulp deviations.
        if descent_margin < -1e-12:
            raise RuntimeError(
                f"{method}: sufficient descent violated: {float(descent_margin)}"
            )
        record = {
            "objective_before": float(objective_before),
            "objective_after": float(objective_after),
            "majorization_margin": float(acceptance),
            "descent_lower_bound": float(bound),
            "descent_margin": float(descent_margin),
            "parameter_step": step,
            "auxiliary_step": auxiliary_step,
            "backtracks": backtracks,
            "parameter_displacement": float(square.sqrt()),
            "auxiliary_displacement": float(auxiliary_square.sqrt()),
        }
    return candidate, trial_auxiliary.detach(), record


def stationarity(parameters, auxiliary, problem):
    """Unit-step proximal gradient map and scaled auxiliary stationarity force."""
    parameters = tuple(p.detach().requires_grad_() for p in parameters)
    auxiliary = auxiliary.detach().requires_grad_()
    gradients = torch.autograd.grad(
        problem.smooth(parameters, auxiliary), (*parameters, auxiliary)
    )
    with torch.no_grad():
        projected = problem.prox(
            tuple(p - g for p, g in zip(parameters, gradients[:-1], strict=True)), 1
        )
        mapping = squared_norm(
            tuple(a - b for a, b in zip(parameters, projected, strict=True))
        ).sqrt()
        return {
            "parameter_proximal_mapping_norm": float(mapping),
            "auxiliary_gradient_norm": float(gradients[-1].norm()),
            "auxiliary_force_rms": float(
                gradients[-1].norm() * math.sqrt(problem.nodes)
            ),
        }


@torch.no_grad()
def feasibility(parameters):
    norms = [
        float(torch.linalg.matrix_norm(p, ord=2) if p.ndim == 2 else p.norm())
        for p in parameters
    ]
    return {
        "parameter_ball_norms": norms,
        "maximum_constraint_excess": max(0.0, max(norms) - 1),
    }


def train_latent(model, anchors, graph, pairs, method, updates=300):
    assert method in METHODS and updates > 0
    problem = LatentObjective(
        model, anchors, graph, pairs, float(method != "palm_no_prior")
    )
    parameters = tuple(p.detach().clone() for p in model.parameters())
    auxiliary = model(anchors).detach()
    initial_stationarity = stationarity(parameters, auxiliary, problem)
    history = []
    for iteration in range(updates):
        parameters, auxiliary, record = split_step(
            parameters,
            auxiliary,
            problem,
            "joint" if method == "joint_quadratic" else "palm",
        )
        record["iteration"] = iteration + 1
        record["constraint_excess"] = feasibility(parameters)[
            "maximum_constraint_excess"
        ]
        assert record["constraint_excess"] < 1e-12
        history.append(record)
    with torch.no_grad():
        for target, value in zip(model.parameters(), parameters, strict=True):
            target.copy_(value)
        terms = [float(term) for term in problem.terms(parameters, auxiliary)]
    return auxiliary, {
        "history": history,
        "initial_stationarity": initial_stationarity,
        "final_stationarity": stationarity(parameters, auxiliary, problem),
        "feasibility": feasibility(parameters),
        "final_contrastive": terms[0],
        "final_geometry": terms[1],
        "final_coupling": terms[2],
        "finite_budget_convergence_claimed": False,
    }
