"""Matched encoder continuation and inexact nonconvex ADMM experiments."""

import torch

from models.marble_geometry_prior import pair_loss, representation_diagnostics

METHODS = ("control", "direct_quadratic", "admm_quadratic", "admm_tv", "pnp")
OUTER_STEPS = 30
INNER_STEPS = 10
LEARNING_RATE = 1e-3
RHO = 1.0
QUADRATIC_WEIGHT = 1.0
TV_WEIGHT = 0.1
BASE_PRIOR_WEIGHT = 1e-4


def base_prior(model):
    parameters = list(model.enc.parameters())
    return (
        0.5
        * BASE_PRIOR_WEIGHT
        * sum(p.square().sum() for p in parameters)
        / sum(p.numel() for p in parameters)
    )


def optimize(model, features, graph, pairs, method):
    assert method in METHODS
    model.eval()  # Freeze BN buffers and disable dropout in every arm.
    positive, negative = pairs
    optimizer = torch.optim.Adam(model.enc.parameters(), lr=LEARNING_RATE)
    with torch.no_grad():
        initial = model(features).detach()
        auxiliary = initial.clone()
        bandwidth = (initial[graph.source] - initial[graph.target]).square().sum(1)
        bandwidth = float(bandwidth.median().clamp_min(1e-6))
        initial_diagnostics = representation_diagnostics(initial)
    dual = torch.zeros_like(auxiliary)
    tv_dual = None
    history = []
    for outer in range(OUTER_STEPS):
        for _ in range(INNER_STEPS):
            values = model(features)
            likelihood = pair_loss(values, positive, negative) + base_prior(model)
            if method == "control":
                objective = likelihood
            elif method == "direct_quadratic":
                objective = likelihood + QUADRATIC_WEIGHT * graph.energy(
                    values, "quadratic"
                )
            else:
                penalty = (
                    0.5 * RHO * (values - auxiliary + dual).square().sum() / len(values)
                )
                objective = likelihood + penalty
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
        with torch.no_grad():
            values = model(features).detach()
            record = {
                "outer": outer + 1,
                "contrastive": float(pair_loss(values, positive, negative)),
            }
            record["quadratic_energy"] = float(graph.energy(values, "quadratic"))
            record["tv_energy"] = float(graph.energy(values, "tv"))
            record["base_prior_energy"] = float(base_prior(model))
            if method in ("admm_quadratic", "admm_tv", "pnp"):
                previous = auxiliary.clone()
                query = values + dual
                if method == "admm_quadratic":
                    auxiliary, diagnostic = graph.quadratic_prox(
                        query, QUADRATIC_WEIGHT / RHO
                    )
                elif method == "admm_tv":
                    auxiliary, tv_dual, diagnostic = graph.tv_prox(
                        query, TV_WEIGHT / RHO, tv_dual
                    )
                else:
                    auxiliary = graph.adaptive_denoise(query, bandwidth)
                    diagnostic = {"denoiser_bandwidth_squared": bandwidth}
                dual += values - auxiliary
                record.update(diagnostic)
                record["primal_relative"] = float(
                    (values - auxiliary).norm() / values.norm().clamp_min(1e-12)
                )
                record["dual_relative"] = float(
                    RHO * (auxiliary - previous).norm() / values.norm().clamp_min(1e-12)
                )
            if method in ("direct_quadratic", "admm_quadratic"):
                record["explicit_objective"] = (
                    record["contrastive"]
                    + record["base_prior_energy"]
                    + QUADRATIC_WEIGHT * record["quadratic_energy"]
                )
            elif method == "admm_tv":
                record["explicit_objective"] = (
                    record["contrastive"]
                    + record["base_prior_energy"]
                    + TV_WEIGHT * record["tv_energy"]
                )
            assert torch.isfinite(values).all()
            history.append(record)
    # Always deploy the inductive encoder, never auxiliary denoised train codes.
    diagnostics = {
        "initial": initial_diagnostics,
        "final": representation_diagnostics(values),
        "history": history,
        "fixed_budget_reached": True,
        "global_optimality_or_convergence_claimed": False,
        "deployed_output": "encoder(values), not ADMM auxiliary variable",
    }
    return diagnostics
