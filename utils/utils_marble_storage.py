"""Lossless compressed storage for new MARBLE reference graphs and sampling plans."""

import gzip
import io
from pathlib import Path

import torch


def save_compressed(value, path):
    path = Path(path)
    assert path.suffix == ".gz"
    if path.exists():
        raise FileExistsError(path)
    buffer = io.BytesIO()
    torch.save(value, buffer)
    with gzip.open(path, "wb", compresslevel=1) as handle:
        handle.write(buffer.getbuffer())


def load_tensor_file(path, *, weights_only=True):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            content = io.BytesIO(handle.read())
        return torch.load(content, map_location="cpu", weights_only=weights_only)
    return torch.load(path, map_location="cpu", weights_only=weights_only)
