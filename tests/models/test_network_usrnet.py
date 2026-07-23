"""Unit tests for models.network_usrnet (FFT migration).

Exercises the migrated torch.fft wrappers through the real forward path of
DataNet, which internally calls rfft/irfft/cmul/cdiv on the interleaved-real
(N, C, H, W, 2) complex layout.
"""
import torch

import models.network_usrnet as usrnet


def test_datanet_forward_runs():
    torch.manual_seed(0)
    N, C, H, W, sf = 1, 3, 16, 16, 2
    Up = H * sf
    x0 = torch.randn(N, C, H, W)
    x = torch.nn.functional.interpolate(x0, scale_factor=sf, mode='nearest')
    FB = torch.randn(N, C, Up, Up, 2)
    FBC = usrnet.cconj(FB)
    F2B = usrnet.r2c(usrnet.cabs2(FB))
    STy = torch.nn.functional.interpolate(x0, scale_factor=sf, mode='nearest')
    FBFy = usrnet.cmul(FBC, usrnet.rfft(STy))
    out = usrnet.DataNet()(x, FB, FBC, F2B, FBFy, torch.tensor(0.1), sf)
    assert out.shape == (N, C, Up, Up)
    assert torch.isfinite(out).all()
