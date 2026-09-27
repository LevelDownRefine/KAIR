"""Anchored implicit MARBLE layer with a spectrally bounded latent operator.

If Lip(N) <= 1, D=2N-I is 1/2-strictly pseudocontractive and the anchored
iteration S_q(z)=(N(z)+q)/2 is contractive. This certifies inference with fixed
weights; it makes no claim about convergence of Adam or the base encoder.
"""

import torch
from torch import nn
from torch.nn import functional as F

SPECTRAL_CAP = 0.999
UNROLL_STEPS = 16
METHODS = ("unconstrained", "soft_spectral", "hard_spectral")


class SpectralLatentLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        assert channels > 0
        self.first = nn.Linear(channels, 2 * channels)
        self.second = nn.Linear(2 * channels, channels)
        with torch.no_grad():
            self.first.weight.zero_()
            self.second.weight.zero_()
            eye = torch.eye(channels) * (SPECTRAL_CAP / 2**0.5)
            self.first.weight[:channels].copy_(eye)
            self.first.weight[channels:].copy_(-eye)
            self.second.weight[:, :channels].copy_(eye)
            self.second.weight[:, channels:].copy_(-eye)
            self.first.bias.zero_()
            self.second.bias.zero_()

    def operator(self, values):
        # 2*tanh(x/2) is smooth and 1-Lipschitz, without a norm singularity.
        return self.second(2 * torch.tanh(self.first(values) / 2))

    def denoise(self, values):
        return 2 * self.operator(values) - values

    def step(self, values, anchor):
        return (self.operator(values) + anchor) / 2

    def forward(self, anchor, steps=UNROLL_STEPS):
        assert steps > 0
        values = anchor
        for _ in range(steps):
            values = self.step(values, anchor)
        return values

    def spectral_penalty(self):
        return sum(
            F.relu(torch.linalg.matrix_norm(layer.weight, ord=2) - 1).square()
            for layer in (self.first, self.second)
        )

    @torch.no_grad()
    def cap_weights(self):
        # Exact small-matrix SVD in double, with slack for float32 rounding.
        # Radial rescaling is a hard cap, not the nearest Frobenius projection.
        for layer in (self.first, self.second):
            norm = torch.linalg.matrix_norm(layer.weight.double(), ord=2)
            layer.weight.div_((norm / SPECTRAL_CAP).clamp_min(1).to(layer.weight))

    @torch.no_grad()
    def bounds(self):
        norms = [
            float(torch.linalg.matrix_norm(layer.weight.double(), ord=2))
            for layer in (self.first, self.second)
        ]
        bound = norms[0] * norms[1]
        return {
            "weight_norms": norms,
            "operator_lipschitz_upper_bound": bound,
            "iteration_contraction_upper_bound": bound / 2,
            "half_strict_pseudocontractive_certified": bound <= 1,
            "inference_contraction_certified": bound < 2,
            "certificate_scope": (
                "fixed weights; floating-point SVD, not interval arithmetic"
            ),
        }

    @torch.no_grad()
    def solve(self, anchor, tolerance=1e-6, max_steps=100):
        assert tolerance > 0 and max_steps > 0
        diagnostic = self.bounds()
        contraction = diagnostic["iteration_contraction_upper_bound"]
        threshold = tolerance * (1 - contraction) if contraction < 1 else tolerance
        values = anchor
        residual = float("inf")
        for used in range(1, max_steps + 1):
            values = self.step(values, anchor)
            residual = float((self.step(values, anchor) - values).norm(dim=1).max())
            assert torch.isfinite(values).all()
            if residual <= threshold:
                break
        diagnostic.update(
            steps=used,
            fixed_point_residual_max=residual,
            a_posteriori_error_bound=(
                residual / (1 - contraction) if contraction < 1 else None
            ),
            tolerance_met=residual <= threshold,
            requested_tolerance=tolerance,
        )
        return values, diagnostic

    @torch.no_grad()
    def jacobian_diagnostics(self, points):
        derivative = 1 - torch.tanh(self.first(points) / 2).square()
        jacobian = torch.einsum(
            "ah,bh,hc->bac", self.second.weight, derivative, self.first.weight
        )
        identity = torch.eye(points.shape[1], device=points.device, dtype=points.dtype)
        denoiser = 2 * jacobian - identity
        return {
            "sample_count": len(points),
            "sampled_N_jacobian_norm_max": float(
                torch.linalg.matrix_norm(jacobian, ord=2).max()
            ),
            "sampled_D_jacobian_norm_max": float(
                torch.linalg.matrix_norm(denoiser, ord=2).max()
            ),
            "sampled_D_symmetric_eigenvalue_max": float(
                torch.linalg.eigvalsh((denoiser + denoiser.transpose(1, 2)) / 2).max()
            ),
            "sampled_checks_are_global_proof": False,
        }


class FrozenMARBLESpectral(nn.Module):
    """Deploy the frozen feature encoder followed by the implicit latent layer."""

    def __init__(self, base, latent):
        super().__init__()
        self.base, self.latent = base, latent
        self.base.requires_grad_(False)
        self.base.eval()

    def forward(self, features):
        self.base.eval()
        with torch.no_grad():
            anchor = self.base(features)
        return self.latent(anchor)

    @torch.no_grad()
    def solve(self, features, **kwargs):
        self.base.eval()
        return self.latent.solve(self.base(features), **kwargs)
