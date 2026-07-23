"""Unit tests for utils.utils_model.

Exercises model description helpers and the ``test_mode`` dispatch (normal /
pad / split / x8 / split+x8) against a deterministic identity model so the
numerical outcome is known.
"""
import torch
import torch.nn as nn

import utils.utils_model as model_util


class _Identity(nn.Module):
    def forward(self, x):
        return x


def _count_modules(net, cls):
    return sum(isinstance(m, cls) for m in net.modules())


def test_find_last_checkpoint(tmp_path):
    for i in (10, 30, 20):
        open(tmp_path / '{}_G.pth'.format(i), 'w').close()
    init_iter, init_path = model_util.find_last_checkpoint(str(tmp_path), net_type='G')
    assert init_iter == 30
    assert str(init_path).endswith('30_G.pth')


def test_describe_model_reports_name_and_params():
    net = nn.Sequential(nn.Conv2d(3, 3, 3))
    msg = model_util.describe_model(net)
    assert 'Sequential' in msg
    assert 'Params number' in msg
    # 3*3*3*3 weights + 3 bias = 84
    assert '84' in msg
    assert model_util.info_model(net) == msg


def test_describe_params_lists_weights_and_skips_bn_trackers():
    net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
    msg = model_util.describe_params(net)
    assert '0.weight' in msg
    assert 'num_batches_tracked' not in msg
    assert model_util.info_params(net) == msg


def test_print_model_and_params_run(capsys):
    net = nn.Conv2d(3, 3, 3)
    model_util.print_model(net)
    out = capsys.readouterr().out
    assert 'Params number' in out
    model_util.print_params(net)
    out = capsys.readouterr().out
    assert 'mean' in out


def test_test_mode_dispatches_all_modes():
    net = _Identity().eval()
    L = torch.rand(1, 3, 64, 64)
    # every mode must run and preserve the input tensor shape
    # (default min_size=256 keeps the small input in the pad branch, so the
    # split recursion never triggers and the call terminates)
    for mode in range(5):
        E = model_util.test_mode(net, L, mode=mode)
        assert E.shape == L.shape


def test_test_mode_normal_and_split_are_identity():
    net = _Identity().eval()
    L = torch.rand(1, 3, 64, 64)
    # mode 0: direct; mode 2: pad branch on a small image -> both return L
    assert torch.allclose(model_util.test_mode(net, L, mode=0), L)
    assert torch.allclose(model_util.test_mode(net, L, mode=2), L)


def test_test_pad_crops_back_to_original_size():
    net = _Identity().eval()
    L = torch.rand(1, 3, 65, 65)
    E = model_util.test_pad(net, L, modulo=16, sf=1)
    assert E.shape == (1, 3, 65, 65)
    assert torch.allclose(E, L)
