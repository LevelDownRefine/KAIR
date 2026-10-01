"""Original float64 oracle guards the real 3D float32 rounding regression."""

import copy
from pathlib import Path

import pytest
import torch
from models.model_marble import ModelMARBLE
from scripts.marble.precision_reference import compare_references, float32_reference
from utils.utils_marble import read_json, sha256
from utils.utils_marble_storage import load_tensor_file


def fixture():
    path = Path(__file__).parent / "fixtures/marble_float64_reference.pt.gz"
    receipt = read_json(path.with_suffix(".json"))
    assert sha256(path) == receipt["fixture_sha256"]
    return load_tensor_file(path)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_original_float64_oracle_three_sgd_steps(device):
    if device.startswith("cuda"):
        assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = fixture()
    reference = float32_reference(data["original_float64"])
    compare_references(data["original_float32"], data["original_float64"])
    for name, value in reference["initial_state"].items():
        torch.testing.assert_close(
            value, data["original_float32"]["initial_state"][name], rtol=0, atol=0
        )

    def features(ids, targets):
        for batch in data["original_float32"]["batches"]:
            if torch.equal(ids, batch["ids"]):
                assert targets == batch["targets"]
                return batch["features"].to(device)
        raise AssertionError("Unexpected sampled IDs")

    model = ModelMARBLE({"params": data["params"], "device": device})
    model.validate_reference(features, reference)
    broken = copy.deepcopy(reference)
    broken["batches"][1]["embedding"][99, 1] += 0.01
    with pytest.raises(AssertionError):
        model.validate_reference(features, broken)


def test_oracle_bridge_rejects_changed_gradients_and_masks():
    data = fixture()
    broken = copy.deepcopy(data["original_float32"])
    broken["batches"][0]["gradients"]["enc.lins.0.weight"][0, 0] += 0.01
    with pytest.raises(AssertionError):
        compare_references(broken, data["original_float64"])
    broken = copy.deepcopy(data["original_float64"])
    broken["batches"][0]["dropout_mask"][0, 0] = ~broken["batches"][0]["dropout_mask"][
        0, 0
    ]
    with pytest.raises(AssertionError):
        compare_references(data["original_float32"], broken)
