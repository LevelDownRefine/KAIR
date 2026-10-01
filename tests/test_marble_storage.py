"""Compressed plans must preserve exact tensors and remain hash checked."""

import pytest
import torch
from data.dataset_marble import DatasetMARBLE
from utils.utils_marble import save_json, sha256
from utils.utils_marble_storage import load_tensor_file, save_compressed

from tests.test_marble import bundle, graph, params  # noqa: F401


def test_compressed_plan_dataset_equivalence(bundle):  # noqa: F811
    dataset = DatasetMARBLE({"dataroot": str(bundle)})
    before, _, _ = dataset.sampling(0)
    path = bundle / "seed-0/sampling.pt.gz"
    save_compressed(before, path)
    save_json(
        path.parent / "sampling_receipt.json",
        {"file": path.name, "sha256": sha256(path)},
    )
    after, _, _ = dataset.sampling(0)
    for key, tensor in before["initial_state"].items():
        assert key in after["initial_state"]
        torch.testing.assert_close(tensor, after["initial_state"][key], rtol=0, atol=0)
    for key in ("train", "val"):
        torch.testing.assert_close(
            before["epochs"][0][key][0][1],
            after["epochs"][0][key][0][1],
            rtol=0,
            atol=0,
        )
    with pytest.raises(FileExistsError):
        save_compressed(before, path)
    with path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(AssertionError):
        dataset.sampling(0)


def test_lossless_compressed_graph_aliases(tmp_path):
    graph_value = {"x": torch.randn(10, 3), "ids": torch.tensor([1, 1, 2])}
    path = tmp_path / "graphs.pt.gz"
    save_compressed({"train": graph_value, "test": graph_value}, path)
    result = load_tensor_file(path)
    assert "train" in result and "test" in result
    assert result["train"] is result["test"]
    for name, value in graph_value.items():
        assert name in result["train"]
        torch.testing.assert_close(value, result["train"][name], rtol=0, atol=0)
