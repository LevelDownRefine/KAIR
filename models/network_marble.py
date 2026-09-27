"""First-order, nondiffusive MARBLE on native PyTorch sparse operators.

Adapted from LevelDownRefine/MARBLE reproduction/src/modern_gpu.py at
872e46bd6dff2d092f8554a8c084701450e84904. See licenses/MARBLE.txt.
Only the explicitly validated protocol is supported; no image tensor conversion.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LegacyNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.module = nn.BatchNorm1d(channels)

    def forward(self, values):
        return self.module(values)


class LegacyEncoder(nn.Module):
    """Preserve the PyG 2.1 MLP's architecture and checkpoint keys."""

    def __init__(self, channels):
        super().__init__()
        self.lins = nn.ModuleList(
            nn.Linear(left, right) for left, right in zip(channels[:-1], channels[1:])
        )
        self.norms = nn.ModuleList([LegacyNorm(channels[1])])

    def forward(self, values):
        return self.lins[1](F.relu(self.norms[0](self.lins[0](values))))


class MARBLEEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        required = {
            "order": 1,
            "diffusion": False,
            "inner_product_features": False,
            "include_positions": True,
            "include_self": True,
            "vec_norm": False,
            "emb_norm": True,
            "dropout": 0.0,
            "bias": True,
            "frac_sampled_nb": -1,
            "batch_norm": "batch_norm",
        }
        for key, value in required.items():
            assert key in params and params[key] == value, (
                f"Unsupported MARBLE configuration: {key} must be {value!r}"
            )
        assert all(
            key in params
            for key in ("dim_emb", "dim_signal", "hidden_channels", "out_channels")
        )
        assert len(params["hidden_channels"]) == 1
        d, s = params["dim_emb"], params["dim_signal"]
        self.enc = LegacyEncoder(
            [d + s + d * s, *params["hidden_channels"], params["out_channels"]]
        )
        self.diffusion = nn.Module()
        # Unused in this protocol, retained for original checkpoint compatibility.
        self.diffusion.register_parameter(
            "diffusion_time", nn.Parameter(torch.tensor(0.0))
        )

    def forward(self, features):
        return F.normalize(self.enc(features), dim=-1)


class GraphFeatures:
    def __init__(self, graph, device):
        assert all(
            key in graph for key in ("pos", "x", "kernel_indices", "kernel_values")
        )
        self.x = graph["x"].to(device)
        self.pos = graph["pos"].to(device)
        self.n, self.d = self.pos.shape
        self.s = self.x.shape[1]
        self.kernel = torch.sparse_coo_tensor(
            graph["kernel_indices"].to(device),
            graph["kernel_values"].to(device),
            (self.n * self.d, self.n),
            device=device,
            check_invariants=True,
        ).coalesce()

    @torch.no_grad()
    def __call__(self, node_ids=None, targets=None):
        if node_ids is None:
            assert targets is None
            weighted = self.x
        else:
            assert targets is not None and 0 < targets <= len(node_ids)
            node_ids = node_ids.to(device=self.x.device, dtype=torch.long)
            # K[:, ids] @ x[ids] includes duplicate columns. Do not deduplicate.
            counts = torch.bincount(node_ids, minlength=self.n)
            weighted = self.x * counts[:, None]
        gradients = torch.sparse.mm(self.kernel, weighted)
        gradients = gradients.reshape(self.n, self.d, self.s).transpose(1, 2)
        features = torch.cat([self.pos, self.x, gradients.reshape(self.n, -1)], dim=1)
        if node_ids is not None:
            features = features[node_ids[:targets]]
        return features


def contrastive_loss(embedding):
    assert embedding.ndim == 2 and len(embedding) > 0 and len(embedding) % 3 == 0
    anchor, positive, negative = embedding.chunk(3)
    return (
        -F.logsigmoid((anchor * positive).sum(-1)).mean()
        - F.logsigmoid(-(anchor * negative).sum(-1)).mean()
    )
