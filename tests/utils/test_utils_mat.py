"""Unit tests for utils.utils_mat.

``loadmat`` wraps scipy.io with recursive mat-struct -> dict conversion and a
NoneDict fallback for missing keys. ``mat2json`` needs the optional ``pandas``
dependency and is intentionally NOT covered here (pandas is not installed in
the test env); ``loadmat`` is fully exercised using a synthetic .mat written by
the shared ``save_mat`` fixture.
"""
import numpy as np

import utils.utils_mat as mat


def test_loadmat_roundtrip(save_mat):
    p = save_mat('x.mat', a=np.arange(10, dtype=np.float64),
                 m=np.array([[1.0, 2.0], [3.0, 4.0]]))
    d = mat.loadmat(str(p))
    assert np.allclose(d['a'], np.arange(10, dtype=np.float64))
    assert np.allclose(d['m'], [[1.0, 2.0], [3.0, 4.0]])


def test_loadmat_missing_key_is_none(save_mat):
    p = save_mat('y.mat', a=np.ones(3))
    d = mat.loadmat(str(p))
    assert d['nope'] is None


def test_dict_to_nonedict_fallback():
    nd = mat.dict_to_nonedict({'k': {'j': 1}})
    assert nd['k']['j'] == 1
    assert nd['k']['missing'] is None
