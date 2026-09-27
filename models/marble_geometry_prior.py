"""Fixed neural graphs, explicit proximal priors, and a nonlinear PnP denoiser.

The quadratic and vectorial-TV proximal problems are convex. The surrounding
encoder training is not; generic convex ADMM convergence does not apply to it.
"""

import torch
from torch.nn import functional as F


def sparse_matrix(indices, values, shape):
    return torch.sparse_coo_tensor(
        indices, values, shape, device=values.device, check_invariants=True
    ).coalesce()


class GeometryGraph:
    def __init__(self, source, target, weights, nodes):
        assert source.shape == target.shape == weights.shape
        assert torch.all(source < target) and torch.all(weights > 0)
        self.source, self.target, self.weights = source, target, weights
        self.nodes = nodes
        degree = torch.zeros(nodes, dtype=weights.dtype, device=weights.device)
        degree.scatter_add_(0, source, weights)
        degree.scatter_add_(0, target, weights)
        self.weights = weights / degree.max()
        self.degree = degree / degree.max()
        diagonal = torch.arange(nodes, device=weights.device)
        indices = torch.stack(
            [
                torch.cat([source, target, diagonal]),
                torch.cat([target, source, diagonal]),
            ]
        )
        self.laplacian = sparse_matrix(
            indices,
            torch.cat([-self.weights, -self.weights, self.degree]),
            (nodes, nodes),
        )
        edges = torch.arange(len(source), device=weights.device)
        self.incidence = sparse_matrix(
            torch.stack([torch.cat([edges, edges]), torch.cat([source, target])]),
            torch.cat([self.weights.sqrt(), -self.weights.sqrt()]),
            (len(source), nodes),
        )
        self.transpose = self.incidence.transpose(0, 1).coalesce()

    def laplace(self, values):
        return torch.sparse.mm(self.laplacian, values)

    def differences(self, values):
        return torch.sparse.mm(self.incidence, values)

    def adjoint(self, values):
        return torch.sparse.mm(self.transpose, values)

    def energy(self, values, kind):
        assert kind in ("quadratic", "tv")
        if kind == "quadratic":
            return 0.5 * (values * self.laplace(values)).sum() / self.nodes
        return self.differences(values).norm(dim=1).sum() / self.nodes

    @torch.no_grad()
    def quadratic_prox(self, values, strength, tolerance=2e-6, steps=60):
        """Solve (I + strength L) v = values by global conjugate gradients."""
        assert strength >= 0 and tolerance > 0 and steps > 0
        result = values.clone()
        residual = values - result - strength * self.laplace(result)
        direction = residual.clone()
        squared = residual.square().sum()
        threshold = tolerance**2 * values.square().sum().clamp_min(1e-30)
        used = 0
        for used in range(steps):
            if squared <= threshold:
                break
            applied = direction + strength * self.laplace(direction)
            denominator = (direction * applied).sum()
            assert denominator > 0
            alpha = squared / denominator
            result = result + alpha * direction
            residual = residual - alpha * applied
            updated = residual.square().sum()
            direction = residual + updated / squared * direction
            squared = updated
        relative = (result + strength * self.laplace(result) - values).norm()
        relative = float(relative / values.norm().clamp_min(1e-30))
        return result, {"linear_relative_residual": relative, "cg_steps": used}

    @torch.no_grad()
    def tv_prox(self, values, strength, dual=None, steps=100):
        """Dual projected gradient for .5||v-q||² + strength sum_e ||Bv_e||.

        ||B||² <= 2 by degree normalization, so dual step .49 is conservative.
        The returned primal/dual gap certifies the convex subproblem accuracy.
        """
        assert strength > 0 and steps > 0
        if dual is None:
            dual = torch.zeros_like(self.differences(values))
        for _ in range(steps):
            primal = values - self.adjoint(dual)
            dual = dual + 0.49 * self.differences(primal)
            dual = dual / (dual.norm(dim=1, keepdim=True) / strength).clamp_min(1)
        correction = self.adjoint(dual)
        result = values - correction
        primal_value = 0.5 * correction.square().sum()
        primal_value += strength * self.differences(result).norm(dim=1).sum()
        dual_value = (values * correction).sum() - 0.5 * correction.square().sum()
        gap = float((primal_value - dual_value) / self.nodes)
        assert gap >= -1e-5 and torch.isfinite(result).all()
        return result, dual, {"tv_gap_per_node": gap}

    @torch.no_grad()
    def adaptive_denoise(self, values, bandwidth):
        """One adaptive graph diffusion step; not asserted to be a proximal map."""
        assert bandwidth > 0
        delta = values[self.source] - values[self.target]
        gates = torch.exp(-delta.square().sum(1) / bandwidth)
        increments = 0.5 * (self.weights * gates)[:, None] * delta
        result = values.clone()
        result.index_add_(0, self.source, -increments)
        result.index_add_(0, self.target, increments)
        return result


@torch.no_grad()
def make_graph(positions, velocities, neighbors=15):
    """Symmetrized Gaussian kNN graph, fit entirely on neural training data."""
    blocks = []
    for block in (positions, velocities):
        centered = block - block.mean(0)
        scale = centered.square().mean().sqrt()
        assert scale > 0
        blocks.append(centered / scale)
    data = torch.cat(blocks, dim=1)
    n = len(data)
    assert 0 < neighbors < n
    distances, targets = [], []
    for start in range(0, n, 512):
        part = torch.cdist(data[start : start + 512], data).square()
        rows = torch.arange(len(part), device=data.device)
        part[rows, rows + start] = float("inf")
        distance, target = part.topk(neighbors, largest=False)
        distances.append(distance)
        targets.append(target)
    distances, targets = torch.cat(distances), torch.cat(targets)
    bandwidth = distances[:, -1].median().clamp_min(1e-12)
    values = torch.exp(-distances / bandwidth).flatten()
    source = torch.arange(n, device=data.device).repeat_interleave(neighbors)
    target = targets.flatten()
    indices = torch.stack(
        [torch.minimum(source, target), torch.maximum(source, target)]
    )
    symmetric = sparse_matrix(indices, values / 2, (n, n))
    graph = GeometryGraph(*symmetric.indices(), symmetric.values(), n)
    return (
        graph,
        targets,
        {
            "neighbors": neighbors,
            "edges": len(graph.weights),
            "phase_bandwidth_squared": float(bandwidth),
            "normalization": "combinatorial Laplacian / maximum weighted degree",
        },
    )


def sample_pairs(neighbors, seed, count=4):
    n, k = neighbors.shape
    generator = torch.Generator().manual_seed(seed)
    columns = torch.randint(k, (n, count), generator=generator).to(neighbors.device)
    positives = neighbors.gather(1, columns)
    negatives = torch.randint(n, (n, count), generator=generator).to(neighbors.device)
    return positives, negatives


def pair_loss(values, positive, negative):
    center = values[:, None, :]
    return (
        F.softplus(-(center * values[positive]).sum(-1)).mean()
        + F.softplus((center * values[negative]).sum(-1)).mean()
    )


def representation_diagnostics(values):
    centered = values.double() - values.double().mean(0)
    singular = torch.linalg.svdvals(centered)
    return {
        "centered_rms": float(centered.square().mean().sqrt()),
        "effective_rank": int((singular > singular[0] * 1e-4).sum()),
        "singular_values": singular.cpu().tolist(),
        "mean_norm": float(values.mean(0).norm()),
    }
