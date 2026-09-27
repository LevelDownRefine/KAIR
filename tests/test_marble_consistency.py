"""Cross-animal protocol checks and independent PyG dropout regression fixtures."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from main_test_marble_consistency import evaluate_modes, load_bundle
from models.model_marble import ModelMARBLE, cpu_state
from models.network_marble import MARBLEEncoder, contrastive_loss
from utils.utils_marble import read_json, save_json, sha256
from utils.utils_marble_consistency import align_positions, consistency_scores


def rat_params():
    return {
        "order": 1,
        "diffusion": False,
        "inner_product_features": False,
        "include_positions": True,
        "include_self": True,
        "vec_norm": False,
        "emb_norm": True,
        "dropout": 0.5,
        "bias": True,
        "frac_sampled_nb": -1,
        "batch_norm": "batch_norm",
        "dim_emb": 2,
        "dim_signal": 2,
        "hidden_channels": [8],
        "out_channels": 3,
    }


def test_rat_dropout_eval_matches_zero_dropout():
    params = rat_params()
    network = MARBLEEncoder(params).eval()
    reference_params = copy.deepcopy(params)
    reference_params["dropout"] = 0.0
    reference = MARBLEEncoder(reference_params).eval()
    reference.load_state_dict(network.state_dict(), strict=True)
    values = torch.randn(12, 8)
    torch.testing.assert_close(network(values), reference(values), rtol=0, atol=0)


def test_fixed_dropout_mask_rejected_in_eval():
    network = MARBLEEncoder(rat_params()).eval()
    with pytest.raises(AssertionError, match="training"):
        network(torch.ones(12, 8), dropout_mask=torch.ones(12, 8, dtype=torch.bool))


def tensor(value):
    return torch.tensor(value["values"], dtype=getattr(torch, value["dtype"]))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_original_pyg_dropout_gradients_and_three_sgd_steps(device):
    if device.startswith("cuda"):
        assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fixture = read_json(
        Path(__file__).parent / "fixtures/marble_dropout_reference.json"
    )
    assert fixture["native_cpu_dropout_matched"]
    network = MARBLEEncoder(rat_params()).to(device).train()
    network.load_state_dict(
        {name: tensor(value) for name, value in fixture["initial_state"].items()},
        strict=True,
    )
    optimizer = torch.optim.SGD(network.parameters(), lr=1.0, momentum=0.9)
    for step in fixture["steps"]:
        features = tensor(step["features"]).to(device)
        mask = tensor(step["dropout_mask"]).to(device)
        actual = network(features, dropout_mask=mask)
        torch.testing.assert_close(
            actual.cpu(), tensor(step["embedding"]), rtol=2e-4, atol=2e-5
        )
        loss = contrastive_loss(actual)
        torch.testing.assert_close(
            loss.cpu(), tensor(step["loss"]), rtol=2e-4, atol=2e-5
        )
        optimizer.zero_grad()
        loss.backward()
        gradients = {
            name: value.grad
            for name, value in network.named_parameters()
            if value.grad is not None
        }
        assert gradients.keys() == step["gradients"].keys()
        for name, value in gradients.items():
            torch.testing.assert_close(
                value.cpu(), tensor(step["gradients"][name]), rtol=2e-4, atol=2e-5
            )
        optimizer.step()
        for name, value in network.state_dict().items():
            torch.testing.assert_close(
                value.cpu(), tensor(step["state_after"][name]), rtol=2e-4, atol=2e-5
            )
    network.eval()
    torch.testing.assert_close(
        network(features).cpu(), tensor(fixture["eval_embedding"]), rtol=2e-4, atol=2e-5
    )


def test_native_dropout_uses_training_rng():
    network = MARBLEEncoder(rat_params()).train()
    features = torch.randn(12, 8)
    torch.manual_seed(7)
    first = network(features)
    torch.manual_seed(7)
    torch.testing.assert_close(network(features), first, rtol=0, atol=0)
    torch.manual_seed(8)
    assert not torch.allclose(network(features), first)


def test_reference_validation_refuses_missing_mask_and_wrong_gradient():
    fixture = read_json(
        Path(__file__).parent / "fixtures/marble_dropout_reference.json"
    )
    batches = []
    for index, step in enumerate(fixture["steps"]):
        batch = {
            key: {name: tensor(value) for name, value in data.items()}
            if key in ("gradients", "state_after")
            else tensor(data)
            for key, data in step.items()
        }
        batch.update(ids=torch.tensor([index]), targets=12)
        batches.append(batch)
    reference = {
        "initial_state": {
            name: tensor(value) for name, value in fixture["initial_state"].items()
        },
        "batches": batches,
    }

    def features(ids, targets):
        assert targets == 12
        return batches[int(ids[0])]["features"]

    model = ModelMARBLE({"params": rat_params(), "device": "cpu"})
    report = model.validate_reference(features, reference)
    assert any(":gradient:" in key for key in report["max_absolute_errors"])
    missing = copy.deepcopy(reference)
    del missing["batches"][0]["dropout_mask"]
    with pytest.raises(AssertionError, match="fixed mask"):
        model.validate_reference(features, missing)
    corrupt = copy.deepcopy(reference)
    corrupt["batches"][0]["gradients"]["enc.lins.0.weight"][0, 0] += 1
    with pytest.raises(AssertionError):
        model.validate_reference(features, corrupt)


def test_failed_evaluation_restores_bn_and_checkpoint():
    network = MARBLEEncoder(rat_params())
    saved = cpu_state(network)
    reference = {
        "notebook_seed0": {
            "embedding": torch.zeros(12, 3),
            "state_after": saved,
            "dropout_mask": torch.ones(12, 8, dtype=torch.bool),
        }
    }
    with pytest.raises(AssertionError):
        evaluate_modes(network, torch.randn(12, 8), saved, reference)
    assert not network.training and network.enc.dropout == 0.5
    for name, value in network.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


def test_consistency_matches_independent_cebra_fixture():
    fixture = read_json(
        Path(__file__).parent / "fixtures/marble_dropout_reference.json"
    )
    data = fixture["consistency"]
    result, aligned = consistency_scores(
        [np.asarray(e, dtype=np.float32) for e in data["embeddings"]],
        [np.asarray(y, dtype=np.float64) for y in data["labels"]],
        data["animals"],
        n_bins=data["n_bins"],
    )
    assert result["pairs"] == data["pairs"]
    np.testing.assert_allclose(result["scores"], data["scores"], rtol=2e-5, atol=2e-6)
    for actual, expected in zip(aligned, data["aligned"], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


def test_alignment_uses_global_edges_excludes_upper_edge_and_expands_gaps():
    labels = np.array([0.0, 1.0, 3.0, 4.0])
    embeddings = np.array([[1.0, 0], [0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
    aligned = align_positions([embeddings, embeddings], [labels, labels], n_bins=5)
    # Four means for five edges. Empty bin 3 averages neighbors in bins 2 and 4.
    expected = np.array(
        [
            [1.0, 0],
            [0, 1.0],
            [1 / np.sqrt(5), 2 / np.sqrt(5)],
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
        ]
    )
    np.testing.assert_allclose(aligned[0], expected)
    shifted = align_positions(
        [embeddings[::-1], embeddings], [labels[::-1], labels], n_bins=5
    )
    np.testing.assert_allclose(shifted[0], expected)


def test_consistency_direction_and_pair_order():
    position = np.linspace(0, 1, 120)
    a = np.column_stack([np.sin(position * 10), np.cos(position * 10)])
    b = np.column_stack(
        [np.sin(position * 10), np.cos(position * 10), np.sin(position * 23)]
    )
    result, _ = consistency_scores(
        [a, b], [position, position], ["first", "second"], n_bins=20
    )
    assert result["pairs"] == [["first", "second"], ["second", "first"]]
    assert result["scores"][0] < result["scores"][1]
    with pytest.raises(AssertionError):
        consistency_scores([a, b], [position, position], ["same", "same"])


def test_alignment_refuses_missing_bins_and_undefined_normalization():
    y = np.array([0.0, 100.0])
    with pytest.raises(ValueError, match="two bins"):
        align_positions([np.ones((2, 2))] * 2, [y, y], n_bins=100)
    with pytest.raises(ValueError, match="normalize"):
        align_positions([np.zeros((2, 2))] * 2, [y, y], n_bins=3)


def test_corrupt_bundle_rejected_before_deserialization(tmp_path):
    path = tmp_path / "rat_consistency.pt"
    path.write_bytes(b"not a pickle")
    save_json(tmp_path / "receipt.json", {"bundle_sha256": sha256(path)})
    path.write_bytes(b"modified")
    with pytest.raises(AssertionError, match="hash mismatch"):
        load_bundle(tmp_path)
