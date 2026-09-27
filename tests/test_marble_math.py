"""Mathematical identities and frozen out-of-sample transforms, not score targets."""

import torch

from models.marble_math_common import PhaseDictionary
from models.network_marble_experiment import MathematicalModel


def test_phase_kernel_invariant_to_orthogonal_coordinates():
    torch.manual_seed(15)
    data = torch.randn(70, 6, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    rotated = torch.cat([data[:, :3] @ rotation, data[:, 3:] @ rotation], dim=1)
    first, second = (
        PhaseDictionary(32).fit(data, 2),
        PhaseDictionary(32).fit(rotated, 2),
    )
    torch.testing.assert_close(
        first.kernel(data), second.kernel(rotated), rtol=1e-10, atol=1e-10
    )


def test_diffusion_nystrom_matches_landmark_eigenfunctions():
    torch.manual_seed(8)
    data = torch.randn(80, 6, dtype=torch.float64)
    model = MathematicalModel(32).fit(data, 3, 0)
    actual = model.transform(data[model.dictionary.ids])
    torch.testing.assert_close(
        actual, model.basis * model.eigenvalues, rtol=1e-9, atol=1e-9
    )
    assert abs(model.diagnostics["stationary_eigenvalue"] - 1) < 1e-10
    assert model.diagnostics["eigen_residual"] < 1e-10


def test_transform_does_not_refit_to_test_batch():
    torch.manual_seed(1)
    data = torch.randn(80, 6, dtype=torch.float64)
    model = MathematicalModel(32).fit(data, 3, 0)
    mean = model.dictionary.mean.clone()
    query = torch.randn(20, 6, dtype=torch.float64) + 0.3
    torch.testing.assert_close(
        model.transform(query),
        torch.cat([model.transform(query[:7]), model.transform(query[7:])]),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(model.dictionary.mean, mean, rtol=0, atol=0)
