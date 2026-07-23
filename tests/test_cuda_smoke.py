"""Smoke test: confirm PyTorch can allocate a tensor on the CUDA device.

Running this on the target machine (NVIDIA RTX 5070 Ti) verifies that the
cu128 build of PyTorch is correctly installed and the GPU driver is new
enough (R570+) to initialize CUDA.

This is a *verification* test, not a compatibility guard: if CUDA is not
usable it must FAIL loudly so the problem is visible (CPU-only torch build,
missing CUDA toolkit, or an outdated driver). It must never silently skip.
"""
import torch


def test_torch_ones_cuda():
    # Fail loudly when CUDA is unusable instead of skipping. A skipped test
    # would hide a broken GPU setup; a failure surfaces it immediately.
    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() returned False — CUDA is not usable. "
        "Likely causes: (a) the installed torch is a CPU-only wheel instead "
        "of a CUDA build (e.g. torch 2.11.0+cu128); (b) the NVIDIA driver is "
        "too old for the CUDA version (cu128 needs R570+, cu130 needs R580+). "
        "Re-run `uv sync` to ensure the cu128 torch is installed, then check "
        "`nvidia-smi` for the driver version."
    )

    # Allocate a (3, 3) tensor filled with 1.0 directly on the CUDA device.
    x = torch.ones(3, 3, device="cuda")

    # It must live on the CUDA device, keep the expected shape/dtype,
    # and contain exactly ones.
    assert x.device.type == "cuda"
    assert x.shape == (3, 3)
    assert x.dtype == torch.float32
    assert torch.allclose(x, torch.ones(3, 3, device="cuda"))

    # Report which GPU actually served the tensor (e.g. RTX 5070 Ti).
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA tensor allocated on: {device_name}")
    assert "5070" in device_name, f"Expected RTX 5070 Ti, got: {device_name}"
