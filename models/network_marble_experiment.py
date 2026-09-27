"""Density-corrected landmark diffusion maps with a frozen Nyström extension.

Coifman & Lafon (2006), https://doi.org/10.1016/j.acha.2006.04.006.
This is a structural alternative, not a claim to reproduce the MARBLE algorithm.
"""

import torch

from models.marble_math_common import PhaseDictionary

NAME = "diffusion"
INPUT_KIND = "phase_space"


class MathematicalModel:
    def __init__(self, landmarks=256):
        self.dictionary = PhaseDictionary(landmarks)

    @torch.no_grad()
    def fit(self, data, dimensions, seed, reference=None):
        self.dictionary.fit(data, seed)
        kernel = self.dictionary.kernel(data[self.dictionary.ids])
        self.density = kernel.sum(1)
        corrected = kernel / self.density[:, None] / self.density[None, :]
        degree = corrected.sum(1)
        symmetric = corrected / degree.sqrt()[:, None] / degree.sqrt()[None, :]
        eigenvalues, vectors = torch.linalg.eigh(symmetric)
        order = torch.argsort(eigenvalues, descending=True)
        assert dimensions + 1 <= len(order)
        self.eigenvalues = eigenvalues[order[1 : dimensions + 1]]
        assert self.eigenvalues.min() > 0
        self.basis = vectors[:, order[1 : dimensions + 1]] / degree.sqrt()[:, None]
        self.diagnostics = {
            "stationary_eigenvalue": float(eigenvalues[order[0]]),
            "eigenvalues": self.eigenvalues.cpu().tolist(),
            "eigen_residual": float(
                (symmetric @ vectors - vectors * eigenvalues).abs().max()
            ),
            "density_correction_alpha": 1,
            "diffusion_time": 1,
        }
        return self

    @torch.no_grad()
    def transform(self, data):
        kernel = self.dictionary.kernel(data) / self.density[None, :]
        transition = kernel / kernel.sum(1, keepdim=True)
        # P_new psi = lambda * psi_new: diffusion coordinates at time one.
        return transition @ self.basis

    def state(self):
        return {
            "dictionary": self.dictionary.state(),
            "density": self.density.cpu(),
            "basis": self.basis.cpu(),
            "eigenvalues": self.eigenvalues.cpu(),
        }
