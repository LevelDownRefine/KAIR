"""Unit tests for utils.utils_bnorm.

``deleteLayer`` removes a layer type, ``merge_bn`` folds Conv+BN into Conv
(numerically identical in eval mode), ``add_bn`` wraps convs with a BN, and
``tidy_sequential`` collapses single-element Sequentials.
"""
import torch
import torch.nn as nn
from collections import OrderedDict

import utils.utils_bnorm as bnorm


def _count(net, cls):
    return sum(isinstance(m, cls) for m in net.modules())


def test_deleteLayer_removes_bn():
    net = nn.Sequential(nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3))
    bnorm.deleteLayer(net, nn.BatchNorm2d)
    assert _count(net, nn.BatchNorm2d) == 0
    assert _count(net, nn.Conv2d) == 1


def test_merge_bn_matches_original_in_eval():
    x = torch.rand(1, 3, 8, 8)
    original = nn.Sequential(
        OrderedDict([('conv', nn.Conv2d(3, 4, 3)),
                     ('bn', nn.BatchNorm2d(4))]))
    original.bn.running_mean.fill_(0.3)
    original.bn.running_var.fill_(1.2)
    original.eval()

    merged = nn.Sequential(
        OrderedDict([('conv', nn.Conv2d(3, 4, 3)),
                     ('bn', nn.BatchNorm2d(4))]))
    merged.conv.load_state_dict(original.conv.state_dict())
    merged.bn.load_state_dict(original.bn.state_dict())
    merged.eval()
    bnorm.merge_bn(merged)

    with torch.no_grad():
        out_orig = original(x)
        out_merged = merged(x)
    assert _count(merged, nn.BatchNorm2d) == 0
    assert torch.allclose(out_orig, out_merged, atol=1e-5)


def test_add_bn_inserts_bn():
    net = nn.Sequential(nn.Conv2d(3, 4, 3))
    bnorm.add_bn(net)
    assert _count(net, nn.BatchNorm2d) == 1
    # forward still works
    out = net(torch.rand(1, 3, 8, 8))
    assert out.shape[:2] == (1, 4)


def test_tidy_sequential_collapses_singletons():
    net = nn.Sequential(nn.Sequential(nn.Conv2d(3, 4, 3)))
    bnorm.tidy_sequential(net)
    assert isinstance(net[0], nn.Conv2d)
