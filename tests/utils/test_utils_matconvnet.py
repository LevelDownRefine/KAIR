"""Unit tests for utils.utils_matconvnet.

``weights2tensor`` only reshapes/permutes a numpy array into the torch
layout; ``save_model`` dumps a state_dict that reloads losslessly.
"""
import numpy as np
import torch
import torch.nn as nn

import utils.utils_matconvnet as mcn


def test_weights2tensor_permutes_4d_to_torch_layout():
    x = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(np.float32)
    t = mcn.weights2tensor(x)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (5, 4, 2, 3)
    assert torch.allclose(t, torch.from_numpy(np.ascontiguousarray(x.transpose((3, 2, 0, 1)))))


def test_weights2tensor_3d_gets_trailing_channel():
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    t = mcn.weights2tensor(x)
    assert t.shape == (1, 4, 2, 3)


def test_weights2tensor_squeeze_reshapes_2d():
    x = np.arange(6).reshape(2, 3).astype(np.float32)
    t = mcn.weights2tensor(x, squeeze=True, in_features=3, out_features=2)
    assert t.shape == (2, 3)
    assert torch.allclose(t, torch.from_numpy(x))


def test_save_model_roundtrip(tmp_path):
    net = nn.Conv2d(3, 4, 3)
    path = tmp_path / 'm.pth'
    mcn.save_model(net, str(path))
    reloaded = torch.load(str(path), weights_only=True)
    assert torch.allclose(reloaded['weight'], net.weight.data.cpu())
    assert torch.allclose(reloaded['bias'], net.bias.data.cpu())
