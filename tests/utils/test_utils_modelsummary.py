"""Unit tests for utils.utils_modelsummary.

FLOPs / activations / parameter counts are checked against a hand-computed
single-conv reference (Conv2d(3,8,3) on a 64x64 input => 884736 MACs, 224
params, 32768 activations, 1 conv), plus the human-readable string formatters.
"""
import torch
import torch.nn as nn
import pytest

import utils.utils_modelsummary as ms


def _conv():
    return nn.Conv2d(3, 8, 3, padding=1)


def test_get_model_flops_single_conv():
    # default print_per_layer_stat=True must not crash after the
    # __batch_counter__ fix (it previously raised AttributeError)
    flops = ms.get_model_flops(_conv(), (3, 64, 64))
    assert isinstance(flops, int)
    assert flops == 884736  # 64*64 * (3*3*3*8)


def test_batch_counter_set_by_start_flops_count():
    net = ms.add_flops_counting_methods(_conv())
    net.eval().start_flops_count()
    net(torch.rand(2, 3, 64, 64))  # batch size 2
    # __batch_counter__ records the forward batch size, not 0 / not 1
    assert net.__batch_counter__ == 2
    # print_model_with_flops must run without error now
    ms.print_model_with_flops(net)
    net.stop_flops_count()


def test_print_model_with_flops_requires_forward():
    net = ms.add_flops_counting_methods(_conv())
    # never started / never forwarded -> counter absent -> explicit error
    try:
        net.start_flops_count()
        with pytest.raises(RuntimeError):
            ms.print_model_with_flops(net)
    finally:
        net.stop_flops_count()


def test_get_model_activation_single_conv():
    act, nconv = ms.get_model_activation(_conv(), (3, 64, 64))
    assert act == 32768  # 1*8*64*64
    assert nconv == 1


def test_get_model_complexity_info_tuple():
    flops, params = ms.get_model_complexity_info(_conv(), (3, 64, 64), as_strings=False)
    assert flops == 884736
    assert params == 224  # 3*3*3*8 weights + 8 bias


def test_get_model_parameters_number():
    assert ms.get_model_parameters_number(_conv()) == 224


def test_flops_to_string():
    assert ms.flops_to_string(2_000_000, units='MMac') == '2.0 MMac'
    assert ms.flops_to_string(0, units=None) == '0 Mac'


def test_params_to_string():
    assert ms.params_to_string(1_500_000) == '1.5 M'
    assert ms.params_to_string(224) == '224'
