"""Whole-graph MARBLE export, including original sampled IDs and fresh weights.

This is an experiment bundle, not an image-pair or per-node DataLoader dataset.
Positions used by the encoder are neural PCA coordinates; behavioral labels are
only exposed to the downstream decoder.
"""

from pathlib import Path

import torch

from utils.utils_marble import read_json, sha256, tensor_digest


def validate_graph(graph):
    assert all(k in graph for k in ("pos", "x", "kernel_indices", "kernel_values"))
    pos, signal = graph["pos"], graph["x"]
    assert pos.ndim == signal.ndim == 2 and len(pos) == len(signal) > 1
    assert pos.shape[1] > 0 and signal.shape[1] > 0
    indices, values = graph["kernel_indices"], graph["kernel_values"]
    assert indices.ndim == 2 and indices.shape[0] == 2
    assert values.ndim == 1 and indices.shape[1] == len(values) > 0
    assert indices.dtype == torch.int64
    assert pos.dtype == signal.dtype == values.dtype == torch.float32
    assert all(torch.isfinite(t).all() for t in (pos, signal, values))
    assert indices.min() >= 0
    assert indices[0].max() < pos.numel() and indices[1].max() < len(pos)


class DatasetMARBLE:
    def __init__(self, options):
        assert "dataroot" in options
        self.root = Path(options["dataroot"]).resolve()
        self.pack = torch.load(
            self.root / "training_input.pt", map_location="cpu", weights_only=True
        )
        required = (
            "params",
            "seeds",
            "train_graph",
            "test_graph",
            "train_labels",
            "test_labels",
            "validation",
        )
        assert all(key in self.pack for key in required)
        self.params = self.pack["params"]
        self.seeds = self.pack["seeds"]
        assert self.seeds and len(set(self.seeds)) == len(self.seeds)
        assert all(isinstance(seed, int) and seed >= 0 for seed in self.seeds)
        for split in ("train", "test"):
            graph, labels = self.pack[f"{split}_graph"], self.pack[f"{split}_labels"]
            validate_graph(graph)
            assert "dim_emb" in self.params and "dim_signal" in self.params
            assert graph["pos"].shape[1] == self.params["dim_emb"]
            assert graph["x"].shape[1] == self.params["dim_signal"]
            assert labels.ndim == 2 and len(labels) == len(graph["pos"])
            assert labels.shape[1] >= 1 and torch.isfinite(labels).all()
        self.protocol = read_json(self.root / "protocol.json")
        self.provenance = read_json(self.root / "provenance.json")
        assert "model_parameters" in self.protocol
        assert self.protocol["model_parameters"] == self.params

    def sampling(self, seed):
        assert seed in self.seeds
        directory = self.root / f"seed-{seed}"
        path = directory / "sampling.pt"
        receipt = read_json(directory / "sampling_receipt.json")
        assert "sha256" in receipt and sha256(path) == receipt["sha256"]
        plan = torch.load(path, map_location="cpu", weights_only=True)
        assert all(key in plan for key in ("initial_state", "epochs", "test"))
        assert "epochs" in self.params and len(plan["epochs"]) == self.params["epochs"]
        initialization = read_json(directory / "initialization.json")
        assert all(
            key in initialization
            for key in (
                "initial_state_sha256",
                "author_state_sha256",
                "optimizer_state_loaded",
            )
        )
        digest = tensor_digest(plan["initial_state"].items())
        assert digest == initialization["initial_state_sha256"]
        assert digest != initialization["author_state_sha256"]
        assert initialization["optimizer_state_loaded"] is False
        n = len(self.pack["train_graph"]["pos"])
        groups = [plan["test"]]
        for epoch in plan["epochs"]:
            assert "train" in epoch and "val" in epoch
            groups.extend((epoch["train"], epoch["val"]))
        for batches in groups:
            assert batches
            for targets, ids in batches:
                assert isinstance(targets, int) and targets > 0 and targets % 3 == 0
                assert ids.ndim == 1 and len(ids) >= targets
                assert ids.dtype in (torch.int16, torch.int32, torch.int64)
                assert ids.min() >= 0 and ids.max() < n
        return plan, initialization, receipt["sha256"]
