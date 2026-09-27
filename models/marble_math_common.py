"""Label-free phase-space kernels for structural MARBLE alternatives."""

import torch


def squared_distances(left, right):
    return torch.cdist(left, right).square()


def landmark_ids(length, count, seed, device):
    assert 2 <= count <= length
    return torch.randperm(length, generator=torch.Generator().manual_seed(seed))[
        :count
    ].to(device)


class PhaseDictionary:
    """Orthogonally invariant block scaling, then a Gaussian landmark dictionary."""

    def __init__(self, count=256):
        self.count = count

    def fit(self, data, seed):
        assert data.ndim == 2 and data.shape[1] % 2 == 0
        assert torch.isfinite(data).all()
        self.mean = data.mean(0)
        centered = data - self.mean
        half = data.shape[1] // 2
        scales = [part.square().mean().sqrt() for part in centered.split(half, dim=1)]
        assert all(float(s) > 0 for s in scales)
        self.scale = torch.cat([s.expand(half) for s in scales])
        normalized = centered / self.scale
        self.ids = landmark_ids(
            len(data), min(self.count, len(data)), seed, data.device
        )
        self.anchors = normalized[self.ids].clone()
        distances = squared_distances(self.anchors, self.anchors)
        distances.fill_diagonal_(float("inf"))
        k = min(15, len(self.anchors) - 1)
        self.bandwidth = distances.kthvalue(k, dim=1).values.median()
        assert self.bandwidth > 0 and torch.isfinite(self.bandwidth)
        return self

    def kernel(self, data):
        values = (data - self.mean) / self.scale
        return torch.exp(-squared_distances(values, self.anchors) / self.bandwidth)

    def state(self):
        return {
            name: getattr(self, name).detach().cpu()
            for name in ("mean", "scale", "ids", "anchors", "bandwidth")
        }


def embedding_diagnostics(values):
    singular = torch.linalg.svdvals(values - values.mean(0))
    assert torch.isfinite(values).all() and singular[0] > 0
    return {
        "effective_rank_relative_1e-4": int((singular > singular[0] * 1e-4).sum()),
        "centered_rms": float((values - values.mean(0)).square().mean().sqrt()),
        "singular_values": singular.cpu().tolist(),
    }
