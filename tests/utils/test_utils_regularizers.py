"""Unit tests for utils.utils_regularizers.

The three regularizers mutate Conv/Linear weights in place via
``module.apply``. ``regularizer_clip`` nudges out-of-range weights by ``eps``;
the SVD regularizers (``regularizer_orth`` / ``regularizer_orth2``) re-factor
the weight through its SVD and are idempotent (applying twice is a no-op).
"""
import torch
import torch.nn as nn

import utils.utils_regularizers as reg


def test_regularizer_clip_nudges_outliers():
    conv_hi = nn.Conv2d(3, 3, 3)
    conv_hi.weight.data.fill_(2.0)   # > c_max (1.5)
    conv_lo = nn.Conv2d(3, 3, 3)
    conv_lo.weight.data.fill_(-2.0)  # < c_min (-1.5)
    lin = nn.Linear(4, 4)
    lin.weight.data.fill_(2.0)
    net = nn.Sequential(conv_hi, conv_lo, lin)

    # the documented usage is model.apply(regularizer_clip)
    net.apply(reg.regularizer_clip)

    assert torch.allclose(conv_hi.weight.data, torch.full((3, 3, 3, 3), 2.0 - 1e-4))
    assert torch.allclose(conv_lo.weight.data, torch.full((3, 3, 3, 3), -2.0 + 1e-4))
    assert torch.allclose(lin.weight.data, torch.full((4, 4), 2.0 - 1e-4))


def _tiny_net():
    return nn.Sequential(nn.Conv2d(3, 4, 3))


def test_regularizer_orth_keeps_shape_and_idempotent():
    net = _tiny_net()
    net.apply(reg.regularizer_orth)
    assert net[0].weight.shape == (4, 3, 3, 3)
    assert torch.isfinite(net[0].weight).all()
    w0 = net[0].weight.data.clone()
    net.apply(reg.regularizer_orth)  # second apply must be a no-op
    assert torch.allclose(net[0].weight.data, w0, atol=1e-4)


def test_regularizer_orth2_keeps_shape_and_idempotent():
    net = _tiny_net()
    net.apply(reg.regularizer_orth2)
    assert net[0].weight.shape == (4, 3, 3, 3)
    assert torch.isfinite(net[0].weight).all()
    w0 = net[0].weight.data.clone()
    net.apply(reg.regularizer_orth2)
    assert torch.allclose(net[0].weight.data, w0, atol=1e-4)
