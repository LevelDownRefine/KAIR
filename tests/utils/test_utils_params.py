"""Unit tests for utils.utils_params.

``rgb2gray_net`` collapses the input conv's 3 channels into 1 using the
Rec.601 luma weights; ``show_kv`` just enumerates state_dict keys.
"""
import torch
from collections import OrderedDict

import utils.utils_params as params


def test_rgb2gray_net_collapses_input_channels():
    # in_filter: shape (out, in, h, w); make the combine deterministic
    w = torch.zeros(3, 3, 3, 3)
    w[:, 0, 0, 0] = 1.0   # R
    w[:, 1, 0, 0] = 2.0   # G
    w[:, 2, 0, 0] = 3.0   # B
    net = OrderedDict({'0.weight': w})

    out = params.rgb2gray_net(net, only_input=True)
    got = out['0.weight']
    assert got.shape == (3, 1, 3, 3)
    expected = 0.2989 * 1.0 + 0.587 * 2.0 + 0.114 * 3.0
    assert torch.allclose(got[:, 0, 0, 0], torch.full((3,), expected))


def test_show_kv_prints_keys(capsys):
    net = OrderedDict({'a.weight': torch.rand(2), 'b.bias': torch.rand(2)})
    params.show_kv(net)
    out = capsys.readouterr().out
    assert 'a.weight' in out
    assert 'b.bias' in out
