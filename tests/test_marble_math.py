"""VAMP covariance/SVD identities and no test-batch adaptation."""

import torch

from models.network_marble_experiment import (
    MathematicalModel,
    inverse_sqrt,
    vamp_projection,
)


def test_covariance_inverse_sqrt():
    matrix = torch.tensor([[3.0, 0.4], [0.4, 1.0]], dtype=torch.float64)
    ridge = torch.tensor(1e-6, dtype=torch.float64)
    whitening = inverse_sqrt(matrix, ridge)
    torch.testing.assert_close(
        whitening @ (matrix + ridge * torch.eye(2)) @ whitening,
        torch.eye(2, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )


def test_vamp_recovers_order_of_known_ar_timescales():
    torch.manual_seed(10)
    rho = torch.tensor([0.98, 0.8, 0.3], dtype=torch.float64)
    data = torch.zeros(8000, 3, dtype=torch.float64)
    noise = torch.randn_like(data) * (1 - rho.square()).sqrt()
    for index in range(1, len(data)):
        data[index] = rho * data[index - 1] + noise[index]
    _, _, diagnostic = vamp_projection(data[500:], 3)
    torch.testing.assert_close(
        torch.tensor(diagnostic["singular_values"]), rho.float(), rtol=0, atol=0.025
    )
    assert diagnostic["svd_residual"] < 1e-10


def test_transform_does_not_refit_to_test_batch():
    torch.manual_seed(1)
    data = torch.randn(80, 6, dtype=torch.float64)
    model = MathematicalModel(32).fit(data, 3, 0)
    mean = model.mean.clone()
    query = torch.randn(20, 6, dtype=torch.float64) + 0.3
    torch.testing.assert_close(
        model.transform(query),
        torch.cat([model.transform(query[:7]), model.transform(query[7:])]),
        rtol=1e-9,
        atol=1e-9,
    )
    torch.testing.assert_close(model.mean, mean, rtol=0, atol=0)
