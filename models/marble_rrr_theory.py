"""Finite-dimensional RRR and deterministic, conditional stability certificates.

Proofs and applicability limits: docs/MARBLE_MATH_FOUNDATIONS.md.
This module never loads behavioral labels or asserts statistical consistency.
"""

import torch


def norm(matrix):
    return torch.linalg.matrix_norm(matrix, ord=2)


def choose_ridge(covariance, condition_budget=100.0):
    assert condition_budget > 1
    values = torch.linalg.eigvalsh(covariance)
    assert values[-1] > 0
    floor = values[-1] * torch.finfo(covariance.dtype).eps * len(values)
    needed = (values[-1] - condition_budget * values[0]) / (condition_budget - 1)
    return torch.maximum(needed, floor)


def rrr(covariance, cross_covariance, rank, ridge):
    assert covariance.shape == cross_covariance.shape
    assert 1 <= rank < len(covariance) and ridge > 0
    values, vectors = torch.linalg.eigh(covariance)
    assert torch.all(values + ridge > 0)
    inverse_sqrt = (vectors * (values + ridge).rsqrt()) @ vectors.T
    b = inverse_sqrt @ cross_covariance
    u, singular, vh = torch.linalg.svd(b, full_matrices=False)
    truncated = (u[:, :rank] * singular[:rank]) @ vh[:rank]
    return {
        "coefficient": inverse_sqrt @ truncated,
        "b": b,
        "h": b @ b.T,
        "singular": singular,
        "projector": u[:, :rank] @ u[:, :rank].T,
        "condition": (values[-1] + ridge) / (values[0] + ridge),
        "ridge": ridge,
    }


def spectral_certificate(h, perturbed_h, rank):
    values, vectors = torch.linalg.eigh(h)
    _, changed = torch.linalg.eigh(perturbed_h)
    gap = float(values[-rank] - values[-rank - 1])
    epsilon = float(norm(perturbed_h - h))
    certified = gap > 2 * epsilon
    projection = vectors[:, -rank:] @ vectors[:, -rank:].T
    other = changed[:, -rank:] @ changed[:, -rank:].T
    return {
        "gap": gap,
        "operator_perturbation": epsilon,
        "twice_perturbation_over_gap": 2 * epsilon / gap if gap > 0 else None,
        "nontrivial_certificate": certified,
        "projector_bound": 2 * epsilon / gap if certified else None,
        "actual_projector_distance": float(norm(projection - other)),
    }


def ridge_perturbation_bound(covariance, cross, other_covariance, other_cross, ridge):
    assert ridge > 0
    return (
        norm(cross - other_cross) / ridge
        + norm(covariance - other_covariance) * norm(other_cross) / ridge**2
    )
