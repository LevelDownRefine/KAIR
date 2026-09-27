"""Verify the CUDA 13 runtime and training on the target RTX 5070 Ti.

These checks fail if CUDA is unavailable. CPU-only CI explicitly excludes this
module; a local verification must never silently skip a broken GPU environment.
PyTorch wheels include the CUDA runtime, so these tests do not require nvcc.
"""

import logging

import torch
from torchvision.ops import nms

from models.network_dncnn import DnCNN

logger = logging.getLogger(__name__)


def test_torch_ones_cuda():
    assert torch.version.cuda == "13.0", (
        f"Expected CUDA 13.0, got {torch.version.cuda}. "
        "Run `uv sync --locked --all-groups` to install the official cu130 wheels."
    )
    assert torch.cuda.is_available(), (
        "CUDA is unavailable. Check `nvidia-smi` for a CUDA 13.0-compatible "
        "NVIDIA driver and run `uv sync --locked --all-groups`."
    )

    x = torch.ones(3, 3, device="cuda")
    assert x.device.type == "cuda"
    assert x.shape == (3, 3)
    assert x.dtype == torch.float32
    torch.testing.assert_close(x.cpu(), torch.ones(3, 3))

    device_name = torch.cuda.get_device_name(0)
    logger.info("CUDA tensor allocated on: %s", device_name)
    assert "5070" in device_name, f"Expected RTX 5070 Ti, got: {device_name}"


def test_dncnn_cuda_training_step():
    """Exercise KAIR convolutions, batch norm, autograd and an optimizer update."""
    torch.manual_seed(0)
    model = DnCNN(in_nc=1, out_nc=1, nc=8, nb=3).cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    noisy = torch.randn(2, 1, 16, 16, device="cuda")
    parameter = next(model.parameters())
    before = parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    prediction = model(noisy)
    assert prediction.shape == noisy.shape
    assert prediction.device.type == "cuda"
    loss = prediction.square().mean()
    assert torch.isfinite(loss)
    loss.backward()

    for weight in model.parameters():
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
    assert torch.count_nonzero(parameter.grad) > 0
    optimizer.step()
    torch.cuda.synchronize()
    assert torch.isfinite(parameter).all()
    assert not torch.equal(before, parameter.detach())


def test_torchvision_cuda_nms():
    """Verify that TorchVision's compiled CUDA operator matches PyTorch."""
    boxes = torch.tensor(
        [[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]],
        dtype=torch.float32,
        device="cuda",
    )
    scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
    kept = nms(boxes, scores, iou_threshold=0.5)
    assert kept.device.type == "cuda"
    torch.testing.assert_close(kept.cpu(), torch.tensor([0, 2]))
