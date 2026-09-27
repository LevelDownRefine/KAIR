"""Independent dense graph oracle and a small full MARBLE experiment on CPU."""

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from data.dataset_marble import DatasetMARBLE, validate_graph
from main_train_marble import run
from models.model_marble import ModelMARBLE, cpu_state
from models.network_marble import GraphFeatures, MARBLEEncoder, contrastive_loss
from utils.utils_marble import (
    audit_decoder_ties,
    position_metrics,
    predict_position,
    read_json,
    save_json,
    sha256,
    tensor_digest,
)


@pytest.fixture
def params():
    return {
        "order": 1,
        "diffusion": False,
        "inner_product_features": False,
        "include_positions": True,
        "include_self": True,
        "vec_norm": False,
        "emb_norm": True,
        "dropout": 0.0,
        "bias": True,
        "frac_sampled_nb": -1,
        "batch_norm": "batch_norm",
        "dim_emb": 2,
        "dim_signal": 2,
        "hidden_channels": [8],
        "out_channels": 4,
        "epochs": 3,
        "lr": 0.1,
        "momentum": 0.9,
    }


@pytest.fixture
def graph():
    generator = torch.Generator().manual_seed(27)
    kernel = torch.randn(8, 4, generator=generator).to_sparse().coalesce()
    return {
        "pos": torch.randn(4, 2, generator=generator),
        "x": torch.randn(4, 2, generator=generator),
        "kernel_indices": kernel.indices(),
        "kernel_values": kernel.values(),
    }


def dense_features(graph, ids=None, targets=None):
    """Slice repeated columns directly, independently of the sparse implementation."""
    n, dimensions = graph["pos"].shape
    kernel = torch.sparse_coo_tensor(
        graph["kernel_indices"], graph["kernel_values"], (n * dimensions, n)
    ).to_dense()
    if ids is None:
        ids = torch.arange(n)
        targets = n
    gradients = kernel[:, ids.long()] @ graph["x"][ids.long()]
    gradients = gradients.reshape(n, dimensions, -1).transpose(1, 2).reshape(n, -1)
    return torch.cat([graph["pos"], graph["x"], gradients], dim=1)[ids[:targets].long()]


def test_repeated_sample_columns_and_gradient_order(graph):
    ids = torch.tensor([2, 0, 2, 1, 0, 3, 1, 1], dtype=torch.int16)
    actual = GraphFeatures(graph, "cpu")
    torch.testing.assert_close(actual(ids, 6), dense_features(graph, ids, 6))
    torch.testing.assert_close(actual(), dense_features(graph))
    assert not torch.allclose(actual(ids, 6), actual()[ids[:6].long()])


@pytest.mark.parametrize(
    "field,value",
    [
        ("order", 2),
        ("diffusion", True),
        ("inner_product_features", True),
        ("frac_sampled_nb", 0.5),
        ("dropout", -0.1),
        ("dropout", 1.0),
        ("include_positions", False),
    ],
)
def test_unsupported_protocol_rejected(params, field, value):
    params[field] = value
    with pytest.raises(AssertionError, match="Unsupported MARBLE"):
        MARBLEEncoder(params)


def test_contrastive_triplets():
    embedding = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    expected = 2 * F.softplus(torch.tensor(-1.0))
    torch.testing.assert_close(contrastive_loss(embedding), expected)
    with pytest.raises(AssertionError):
        contrastive_loss(embedding[:2])


def test_bad_graph_rejected(graph):
    graph["kernel_indices"][0, 0] = 8
    with pytest.raises(AssertionError):
        validate_graph(graph)


@pytest.fixture
def bundle(tmp_path, graph, params):
    root = tmp_path / "input"
    root.mkdir()
    torch.manual_seed(17)
    network = MARBLEEncoder(params).train()
    initial = cpu_state(network)
    ids = torch.tensor([2, 0, 2, 1, 0, 3, 1, 1], dtype=torch.int16)
    optimizer = torch.optim.SGD(network.parameters(), lr=1.0, momentum=0.9)
    reference = []
    for _ in range(3):
        values = dense_features(graph, ids, 6)
        embedding = network(values)
        loss = contrastive_loss(embedding)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reference.append(
            {
                "ids": ids,
                "targets": 6,
                "features": values,
                "embedding": embedding.detach(),
                "loss": loss.detach(),
                "state_after": cpu_state(network),
            }
        )
    pack = {
        "params": params,
        "seeds": [0],
        "train_graph": graph,
        "test_graph": graph,
        "train_labels": torch.arange(4, dtype=torch.float32)[:, None],
        "test_labels": torch.arange(4, dtype=torch.float32)[:, None],
        "validation": {"initial_state": initial, "batches": reference},
    }
    torch.save(pack, root / "training_input.pt")
    save_json(root / "protocol.json", {"model_parameters": params})
    save_json(root / "provenance.json", {"test_fixture": True})
    seed_dir = root / "seed-0"
    seed_dir.mkdir()
    plan = {
        "initial_state": initial,
        "epochs": [{"train": [(6, ids)], "val": [(6, ids)]} for _ in range(3)],
        "test": [(6, ids)],
    }
    torch.save(plan, seed_dir / "sampling.pt")
    save_json(
        seed_dir / "sampling_receipt.json", {"sha256": sha256(seed_dir / "sampling.pt")}
    )
    save_json(
        seed_dir / "initialization.json",
        {
            "initial_state_sha256": tensor_digest(initial.items()),
            "author_state_sha256": "different-author-weights",
            "optimizer_state_loaded": False,
        },
    )
    return root


def test_corrupt_sampling_refused(bundle):
    path = bundle / "seed-0/sampling.pt"
    with path.open("ab") as handle:
        handle.write(b"corruption")
    dataset = DatasetMARBLE({"dataroot": bundle})
    with pytest.raises(AssertionError):
        dataset.sampling(0)


def test_full_training_decoding_and_checkpoint_selection(bundle, tmp_path):
    options = {
        "task": "test",
        "model": "marble",
        "device": "cpu",
        "threads": 1,
        "seeds": [0],
        "dataset": {"dataset_type": "marble", "dataroot": str(bundle)},
        "decoder": {"neighbors": 2, "metric": "cosine"},
    }
    untouched = copy.deepcopy(options)
    output = tmp_path / "run"
    summary = run(options, output)
    assert options == untouched
    assert "runs" in summary and len(summary["runs"]) == 1
    result = summary["runs"][0]
    assert result["initial_state_sha256"] != result["trained_state_sha256"]
    history = read_json(output / "seed-0/loss_history.json")
    best = torch.load(output / "seed-0/best_model.pth", weights_only=True)
    last = torch.load(output / "seed-0/last_model.pth", weights_only=True)
    assert (
        best["epoch"]
        == result["best_epoch_zero_based"]
        == int(np.argmin(history["val_loss"]))
    )
    assert last["epoch"] == 2 and len(history["train_loss"]) == 3
    assert (output / "seed-0/training.png").is_file()
    with pytest.raises(FileExistsError):
        run(options, output)


def test_evaluation_keeps_bn_statistics_and_weights(params, graph):
    model = ModelMARBLE({"params": params, "device": "cpu"})
    features = GraphFeatures(graph, "cpu")
    before = cpu_state(model.netG)
    model.epoch_loss(features, [(6, torch.tensor([0, 1, 2, 3, 0, 1]))])
    for key, value in before.items():
        torch.testing.assert_close(model.netG.state_dict()[key], value, rtol=0, atol=0)
    model.embeddings(features(), features() + 100)
    assert not model.netG.training
    for key, value in before.items():
        torch.testing.assert_close(model.netG.state_dict()[key], value, rtol=0, atol=0)


def test_decoder_uses_training_positions_only():
    train = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    labels = np.array([[10.0, 500.0], [20.0, -500.0], [30.0, 1000.0]])
    prediction = predict_position(train, train[:2], labels, neighbors=1)
    np.testing.assert_array_equal(prediction, [10.0, 20.0])
    metrics = position_metrics(np.array([9.0, 22.0]), prediction)
    assert metrics["mean_absolute_error_m"] == 1.5


def test_decoder_exact_boundary_tie_is_reported():
    actual, expected = np.array([2.0]), np.array([1.0])
    indices, reference = np.array([[0, 2]]), np.array([[0, 1]])
    labels = np.array([0.0, 2.0, 4.0])
    report = audit_decoder_ties(
        actual, indices, expected, reference, np.array([[0.0, 0.5, 0.5]]), labels
    )
    assert report["exact_boundary_tie_samples"] == [0]
    assert report["max_abs_error_m"] == 1.0
    with pytest.raises(AssertionError, match="not an exact boundary tie"):
        audit_decoder_ties(
            actual, indices, expected, reference, np.array([[0.0, 0.5, 0.6]]), labels
        )
    with pytest.raises(AssertionError):
        audit_decoder_ties(
            np.array([9.0]),
            indices,
            expected,
            reference,
            np.array([[0.0, 0.5, 0.5]]),
            labels,
        )
