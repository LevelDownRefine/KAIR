"""Unit tests for utils.utils_receptivefield.

``outFromIn`` is a pure recurrence over a single conv layer; checked against
hand-computed values. ``printLayer`` just renders to stdout.
"""
import math

import utils.utils_receptivefield as rf


def test_outFromIn_unit_stride_same_padding():
    # [k, s, p] = [3,1,1] on a 128px input with j=1, r=1, start=0.5
    out = rf.outFromIn([3, 1, 1], [128, 1, 1, 0.5])
    assert out == (128, 1, 3, 0.5)


def test_outFromIn_no_padding_shrinks():
    # [k,s,p]=[3,1,0] on a 10px input
    n_out, j_out, r_out, start_out = rf.outFromIn([3, 1, 0], [10, 1, 1, 0.5])
    assert n_out == 8
    assert j_out == 1
    assert r_out == 3
    assert math.isclose(start_out, 1.5)


def test_printLayer_renders(capsys):
    rf.printLayer([64, 1, 3, 0.5], 'conv1')
    out = capsys.readouterr().out
    assert 'conv1' in out
    assert '64' in out
