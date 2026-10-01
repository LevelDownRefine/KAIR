"""Preserve the observed rounding failure while checking local and full64 agreement."""

import copy
from pathlib import Path

import pytest
import torch
from models.model_marble import ModelMARBLE
from scripts.marble.precision_reference import compare_references
from scripts.marble.precision_stepwise import validate_steps
from utils.utils_marble import read_json, sha256
from utils.utils_marble_storage import load_tensor_file


def fixture():
    path = Path(__file__).parent / "fixtures/marble_stepwise_reference.pt.gz"
    receipt = read_json(path.with_suffix(".json"))
    assert sha256(path) == receipt["fixture_sha256"]
    return load_tensor_file(path)


def check(data, reference, device, *, reset, dtype):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    feature_source = data[
        "original_float32" if dtype == torch.float32 else "original_float64"
    ]

    def features(ids, targets):
        for batch in feature_source["batches"]:
            if torch.equal(ids, batch["ids"]):
                assert targets == batch["targets"]
                return batch["features"].to(device=device, dtype=dtype)
        raise AssertionError("Unexpected sampled IDs")

    model = ModelMARBLE({"params": data["params"], "device": device})
    return validate_steps(
        model,
        features,
        reference,
        reset=reset,
        dtype=dtype,
        rtol=2e-4 if reset else 1e-8,
        atol=2e-5 if reset else 1e-9,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_local_float32_and_continuous_float64_original_references(device):
    data = fixture()
    # The old failed trajectory bridge must remain visible, not rewritten to pass.
    with pytest.raises(AssertionError):
        compare_references(data["original_float32"], data["original_float64"])
    check(data, data["original_float32"], device, reset=True, dtype=torch.float32)
    check(data, data["original_float64"], device, reset=False, dtype=torch.float64)


@pytest.mark.parametrize("corruption", ["gradients", "dropout_mask", "state_after"])
def test_local_check_rejects_corrupted_reference(corruption):
    data = fixture()
    broken = copy.deepcopy(data["original_float32"])
    batch = broken["batches"][1]
    if corruption == "dropout_mask":
        batch[corruption][:] = ~batch[corruption]
    else:
        batch[corruption]["enc.lins.0.weight"][0, 0] += 0.1
    with pytest.raises(AssertionError):
        check(data, broken, "cpu", reset=True, dtype=torch.float32)


def test_local_check_detects_wrong_momentum(monkeypatch):
    data = fixture()
    original = torch.optim.SGD

    def wrong_momentum(parameters, lr, momentum):
        assert momentum == 0.9
        return original(parameters, lr=lr, momentum=0.8)

    monkeypatch.setattr(torch.optim, "SGD", wrong_momentum)
    with pytest.raises(AssertionError):
        check(data, data["original_float32"], "cpu", reset=True, dtype=torch.float32)
