"""Orthogonal state/increment projection fitted without behavioral labels.

Samples are rows in arrays. The mathematical basis U has shape (D, q), so
column-vector states transform as U.T @ (r - mean). Alpha is dimensionless:
minimize A / trace(S) + alpha * B / trace(D), with uncentered increments.
"""

import numpy as np


def _matrix(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 2 and min(values.shape) > 0
    assert np.isfinite(values).all()
    return values


def _basis(scatter, dimensions):
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    basis = eigenvectors[:, -dimensions:][:, ::-1].copy()
    # Use the largest loading to fix each arbitrary eigenvector sign.
    pivots = np.argmax(np.abs(basis), axis=0)
    basis *= np.sign(basis[pivots, np.arange(dimensions)])
    return basis, eigenvalues[::-1]


class DynamicsProjection:
    def __init__(self, n_components=20, alpha=1.0):
        assert isinstance(n_components, int) and n_components > 0
        assert np.isfinite(alpha) and alpha >= 0
        self.n_components = n_components
        self.alpha = float(alpha)

    def fit(self, values):
        values = _matrix(values)
        assert len(values) > 1 and self.n_components <= values.shape[1]
        self.mean_ = values.mean(axis=0)
        states = values - self.mean_
        increments = np.diff(values, axis=0)
        state_scatter = states.T @ states
        increment_scatter = increments.T @ increments
        self.state_energy_ = float(np.trace(state_scatter))
        self.increment_energy_ = float(np.trace(increment_scatter))
        assert self.state_energy_ > 0 and self.increment_energy_ > 0
        objective = state_scatter / self.state_energy_
        objective += self.alpha * increment_scatter / self.increment_energy_
        basis, self.eigenvalues_ = _basis(objective, self.n_components)
        reference, _ = _basis(state_scatter / self.state_energy_, self.n_components)
        # Rotate within the selected subspace to its nearest PCA orientation.
        # This changes neither the projector nor the minimization objective.
        if self.alpha > 0:
            left, _, right = np.linalg.svd(basis.T @ reference)
            basis = basis @ (left @ right)
        self.basis_ = basis
        self.reference_basis_ = reference
        self.effective_lambda_ = (
            self.alpha * self.state_energy_ / self.increment_energy_
        )
        return self

    def transform(self, values):
        assert hasattr(self, "basis_"), "Fit on training data first"
        values = _matrix(values)
        assert values.shape[1] == len(self.mean_)
        return (values - self.mean_) @ self.basis_

    def fit_transform(self, values):
        return self.fit(values).transform(values)

    def errors(self, values, reference=False):
        assert hasattr(self, "basis_"), "Fit on training data first"
        values = _matrix(values)
        assert len(values) > 1 and values.shape[1] == len(self.mean_)
        basis = self.reference_basis_ if reference else self.basis_
        result = {}
        for name, matrix in (
            ("state", values - self.mean_),
            ("increment", np.diff(values, axis=0)),
        ):
            energy = float(np.sum(matrix**2))
            residual = matrix - (matrix @ basis) @ basis.T
            error = float(np.sum(residual**2))
            assert energy > 0
            result[f"{name}_squared_error"] = error
            result[f"{name}_energy"] = energy
            result[f"{name}_residual_fraction"] = error / energy
        return result
