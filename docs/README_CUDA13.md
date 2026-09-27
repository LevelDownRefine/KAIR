# CUDA 13.0 environment

This fork pins the official stable PyTorch 2.14.0 and TorchVision 0.29.0
CUDA 13.0 wheels in `pyproject.toml` and `uv.lock`. Only these packages use
the explicit `https://download.pytorch.org/whl/cu130` index; other packages
come from PyPI. Python 3.12 is the local validation and CI version.

Official releases: [PyTorch 2.14.0](https://github.com/pytorch/pytorch/releases/tag/v2.14.0),
[TorchVision 0.29.0](https://github.com/pytorch/vision/releases/tag/v0.29.0).

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and an NVIDIA
driver supporting CUDA 13.0. From the repository root:

```sh
nvidia-smi
uv sync --locked --all-groups --python 3.12
uv run --locked python -m pytest tests/test_cuda_smoke.py -v
uv run --locked python -m pytest tests -q
```

`uv sync` creates or updates the repository's `.venv`. It can upgrade an existing
CUDA 12.8 environment; there is no need to delete it. The committed lock file
keeps installations repeatable. Avoid installing the legacy `requirement.txt`
over this environment, since that bypasses the pinned CUDA wheel selection.

Run a training script through the same environment, for example:

```sh
uv run --locked python main_train_psnr.py --opt options/train_msrresnet_psnr.json
```

Configure the dataset paths and GPU IDs in the options file first. In PowerShell,
`. ./.venv/Scripts/Activate.ps1` also activates the environment for plain `python`
commands; `env.bat` is the equivalent entry point for Command Prompt.

## What is verified

Local validation on 2026-09-28 used Windows, Python 3.12.10, an NVIDIA GeForce
RTX 5070 Ti Laptop GPU and driver 616.64. Installed versions were
`torch==2.14.0+cu130` and `torchvision==0.29.0+cu130`, with
`torch.version.cuda == "13.0"`. `uv pip check` passed, and the full test suite
passed all 280 tests (including the three GPU checks) in 28.76 seconds, with
52 deprecation/future warnings from existing code and dependencies. Ruff lint
and format checks passed for both modified Python test files.

`tests/test_cuda_smoke.py` checks the CUDA 13.0 runtime on the target RTX 5070 Ti,
allocates a GPU tensor, runs a KAIR DnCNN forward/backward pass and SGD update,
and checks TorchVision's compiled CUDA NMS operator. These tests fail when CUDA
is unavailable. GitHub's CPU runner explicitly excludes them with `-k "not cuda"`
and runs the remaining tests with the same locked dependencies.

The CUDA runtime shipped inside the PyTorch wheel is sufficient for these
checks and ordinary PyTorch models. `nvidia-smi` reports the driver's supported
CUDA level; `torch.version.cuda` reports the runtime used by PyTorch.

VRT, RVRT and face enhancement also use custom JIT-compiled extensions
(`deform_attn`, `upfirdn2d`, `fused_act`). Building those additionally requires a
CUDA 13.0 toolkit (`nvcc`) and a compatible C++ compiler. They are excluded from
the core import tests, and this environment migration does not establish their
CUDA 13 build compatibility or reproduce any paper's numerical results.
