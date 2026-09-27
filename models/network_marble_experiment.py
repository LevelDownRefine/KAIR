"""Kernel VAMP: variational singular functions of a time-lagged Koopman map.

Wu & Noe, https://arxiv.org/abs/1707.04659. Fixed one-bin lag; no label tuning.
"""

import torch

from models.marble_math_common import PhaseDictionary

NAME = "koopman"
INPUT_KIND = "phase_space"


def inverse_sqrt(covariance, ridge):
    values, vectors = torch.linalg.eigh(covariance)
    assert ridge > 0
    return (vectors * (values.clamp_min(0) + ridge).rsqrt()) @ vectors.T


def vamp_projection(features, dimensions, lag=1):
    assert 0 < lag < len(features) - 1 and dimensions <= features.shape[1]
    left, right = features[:-lag], features[lag:]
    mean = left.mean(0)
    left, right = left - mean, right - right.mean(0)
    c00, c11 = left.T @ left / len(left), right.T @ right / len(left)
    c01 = left.T @ right / len(left)
    ridge = (c00.trace() + c11.trace()) / (2 * len(c00)) * 1e-6
    w0, w1 = inverse_sqrt(c00, ridge), inverse_sqrt(c11, ridge)
    whitened = w0 @ c01 @ w1
    u, singular, vh = torch.linalg.svd(whitened, full_matrices=False)
    projection = w0 @ u[:, :dimensions] * singular[:dimensions]
    diagnostics = {
        "singular_values": singular[:dimensions].cpu().tolist(),
        "vamp2_retained": float(singular[:dimensions].square().sum()),
        "svd_residual": float((whitened - (u * singular) @ vh).abs().max()),
        "ridge": float(ridge),
        "lag_bins": lag,
        "covariance_effective_rank": int((torch.linalg.eigvalsh(c00) > ridge).sum()),
    }
    return mean, projection, diagnostics


class MathematicalModel:
    def __init__(self, landmarks=256):
        self.dictionary = PhaseDictionary(landmarks)

    @torch.no_grad()
    def fit(self, data, dimensions, seed, reference=None):
        self.dictionary.fit(data, seed)
        values = self.dictionary.kernel(data)
        values = values / values.sum(1, keepdim=True)
        self.mean, self.projection, self.diagnostics = vamp_projection(
            values, dimensions
        )
        return self

    @torch.no_grad()
    def transform(self, data):
        values = self.dictionary.kernel(data)
        values = values / values.sum(1, keepdim=True)
        return (values - self.mean) @ self.projection

    def state(self):
        return {
            "dictionary": self.dictionary.state(),
            "mean": self.mean.cpu(),
            "projection": self.projection.cpu(),
        }
