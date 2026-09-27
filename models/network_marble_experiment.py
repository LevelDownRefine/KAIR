"""Entropic Gromov-Wasserstein alignment of learned MARBLE metric spaces.

Squared relational loss, log-Sinkhorn marginal projection, damped updates, and
a barycentric-displacement extension. No behavioral correspondence is supplied.
Equations: https://pythonot.github.io/gen_modules/ot.gromov.html
"""

import torch

from models.marble_math_common import landmark_ids, squared_distances

NAME = "transport"
INPUT_KIND = "marble_embeddings"


def round_coupling(coupling, p, q):
    """Feasible marginal rounding by scaling down and a rank-one residual."""
    coupling = coupling * (p / coupling.sum(1)).clamp(max=1)[:, None]
    coupling = coupling * (q / coupling.sum(0)).clamp(max=1)[None, :]
    row = (p - coupling.sum(1)).clamp_min(0)
    column = (q - coupling.sum(0)).clamp_min(0)
    if float(row.sum()) > 1e-15:
        coupling = coupling + row[:, None] * column[None, :] / row.sum()
    return coupling


def sinkhorn(cost, p, q, epsilon=0.05, max_steps=2000, diagnostics=None):
    assert epsilon > 0 and torch.all(p > 0) and torch.all(q > 0)
    assert abs(float(p.sum() - q.sum())) < 1e-10 and max_steps > 0
    log_kernel = -cost / epsilon
    log_u, log_v = torch.zeros_like(p), torch.zeros_like(q)
    for step in range(max_steps):
        log_u = p.log() - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = q.log() - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
        if (step + 1) % 20 == 0 or step + 1 == max_steps:
            coupling = torch.exp(log_u[:, None] + log_kernel + log_v[None, :])
            error = max(
                float((coupling.sum(1) - p).abs().max()),
                float((coupling.sum(0) - q).abs().max()),
            )
            if error < 1e-8:
                break
    rounded = round_coupling(coupling, p, q)
    correction = float((rounded - coupling).abs().sum())
    assert torch.isfinite(rounded).all()
    if diagnostics is not None:
        diagnostics.append(
            {
                "steps": step + 1,
                "pre_rounding_marginal_error": error,
                "rounding_l1": correction,
                "converged_before_rounding": error < 1e-8,
            }
        )
    return rounded


def gw_loss(left, right, coupling):
    p, q = coupling.sum(1), coupling.sum(0)
    constant = (left.square() @ p)[:, None] + (right.square() @ q)[None, :]
    return ((constant - 2 * left @ coupling @ right.T) * coupling).sum()


def gw_gradient(left, right, coupling):
    p, q = coupling.sum(1), coupling.sum(0)
    constant = (left.square() @ p)[:, None] + (right.square() @ q)[None, :]
    return 2 * (constant - 2 * left @ coupling @ right.T)


def solve_gw(left, right, epsilon=0.05, iterations=60):
    inner = []
    p = torch.full((len(left),), 1 / len(left), device=left.device, dtype=left.dtype)
    q = torch.full(
        (len(right),), 1 / len(right), device=right.device, dtype=right.dtype
    )
    # Quantiles of within-space distances provide a rotation-invariant warm start.
    levels = torch.linspace(0.1, 0.9, 9, device=left.device, dtype=left.dtype)
    signatures0 = torch.quantile(left, levels, dim=1).T
    signatures1 = torch.quantile(right, levels, dim=1).T
    cost = squared_distances(signatures0, signatures1)
    coupling = sinkhorn(cost, p, q, epsilon, diagnostics=inner)

    def objective(value):
        entropy = (value * (value.clamp_min(1e-300).log() - 1)).sum()
        return gw_loss(left, right, value) + epsilon * entropy

    history = [float(objective(coupling))]
    stop = "iteration_limit"
    for _ in range(iterations):
        proposed = sinkhorn(
            gw_gradient(left, right, coupling), p, q, epsilon, diagnostics=inner
        )
        direction = proposed - coupling
        if float(direction.abs().max()) < 1e-8:
            stop = "fixed_point"
            break
        accepted = False
        for step in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = coupling + step * direction
            value = float(objective(candidate))
            if value <= history[-1] + 1e-12:
                coupling = candidate
                history.append(value)
                accepted = True
                break
        if not accepted:
            stop = "line_search_stalled"
            break
    marginal_error = max(
        float((coupling.sum(1) - p).abs().max()),
        float((coupling.sum(0) - q).abs().max()),
    )
    assert marginal_error < 1e-8
    return coupling, {
        "objective_history": history,
        "gw_distortion": float(gw_loss(left, right, coupling)),
        "marginal_max_error": marginal_error,
        "stop_reason": stop,
        "iterations": len(history) - 1,
        "entropy_regularization": epsilon,
        "sinkhorn_calls": inner,
        "row_max_mass_mean": float((coupling / p[:, None]).max(1).values.mean()),
    }


class MathematicalModel:
    def __init__(self, landmarks=128):
        self.landmarks = landmarks

    @torch.no_grad()
    def fit(self, data, dimensions, seed, reference=None):
        assert (
            reference is not None and dimensions == data.shape[1] == reference.shape[1]
        )
        self.identity = torch.equal(data, reference)
        if self.identity:
            self.diagnostics = {
                "identity_reference": True,
                "reason": "Reference animal/seed fixes the common coordinate system",
            }
            return self
        self.source_ids = landmark_ids(
            len(data), min(self.landmarks, len(data)), seed, data.device
        )
        self.target_ids = landmark_ids(
            len(reference), min(self.landmarks, len(reference)), seed, data.device
        )
        self.anchors = data[self.source_ids].clone()
        target = reference[self.target_ids]
        left = squared_distances(self.anchors, self.anchors)
        right = squared_distances(target, target)
        left, right = left / left.mean(), right / right.mean()
        self.coupling, self.diagnostics = solve_gw(left, right)
        mapped = self.coupling @ target / self.coupling.sum(1, keepdim=True)
        self.displacement = mapped - self.anchors
        return self

    @torch.no_grad()
    def transform(self, data):
        if self.identity:
            return data.clone()
        distances = squared_distances(data, self.anchors)
        values, ids = torch.topk(distances, k=min(8, len(self.anchors)), largest=False)
        # Local Shepard interpolation of the displacement, preserving fine detail.
        weights = 1 / values.clamp_min(1e-12)
        weights = weights / weights.sum(1, keepdim=True)
        return data + (weights[:, :, None] * self.displacement[ids]).sum(1)

    def state(self):
        if self.identity:
            return {"identity_reference": True}
        return {
            name: getattr(self, name).cpu()
            for name in (
                "source_ids",
                "target_ids",
                "anchors",
                "coupling",
                "displacement",
            )
        }
