"""Original MARBLE float64 oracle on identical float32 inputs, IDs and masks."""

import argparse
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from utils.utils_marble import read_json, save_json, sha256
from utils.utils_marble_storage import load_tensor_file


def compare_references(observed, oracle):
    """Keep the original tolerance while measuring each backend against float64."""
    assert len(observed["batches"]) == len(oracle["batches"]) == 3
    errors = {}
    for step, (actual, expected) in enumerate(
        zip(observed["batches"], oracle["batches"], strict=True)
    ):
        for key in ("ids", "dropout_mask"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert actual["targets"] == expected["targets"]
        for group in ("features", "embedding", "loss", "gradients", "state_after"):
            left = (
                actual[group]
                if isinstance(actual[group], dict)
                else {group: actual[group]}
            )
            right = (
                expected[group]
                if isinstance(expected[group], dict)
                else {group: expected[group]}
            )
            assert left.keys() == right.keys()
            for name, value in left.items():
                target = right[name]
                torch.testing.assert_close(
                    value.double(), target.double(), rtol=2e-4, atol=2e-5
                )
                errors[f"{step}:{group}:{name}"] = float(
                    (value.double() - target.double()).abs().max()
                )
    return errors


def float32_reference(oracle):
    """Round oracle values once to the trainer dtype; leave indices and masks exact."""
    if isinstance(oracle, torch.Tensor):
        return oracle.float() if oracle.is_floating_point() else oracle
    if isinstance(oracle, dict):
        return {key: float32_reference(value) for key, value in oracle.items()}
    if isinstance(oracle, list):
        return [float32_reference(value) for value in oracle]
    return oracle


def prepare(repository, source):
    sys.path.insert(0, str(repository))
    import MARBLE
    from torch.nn import functional as F

    assert Path(MARBLE.__file__).resolve().parent == repository / "MARBLE"
    assert not torch.cuda.is_available()
    torch.set_num_threads(4)
    output = source / "validation_float64.pt"
    if output.exists():
        raise FileExistsError(output)
    provenance = read_json(source / "provenance.json")
    assert (
        sha256(source / "original_graphs.pt.gz") == provenance["original_graphs_sha256"]
    )
    original = load_tensor_file(source / "original_graphs.pt.gz", weights_only=False)
    graph = original["train"]
    pack = torch.load(
        source / "training_input.pt", weights_only=True, map_location="cpu"
    )
    reference = pack["validation"]
    model = MARBLE.net(graph, params=pack["params"], verbose=False).double().train()
    model.load_state_dict(reference["initial_state"], strict=True)
    graph.x, graph.pos = graph.x.double(), graph.pos.double()
    graph.kernels = [kernel.to(dtype=torch.float64) for kernel in graph.kernels]
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)

    def state():
        return {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    oracle = {"initial_state": state(), "batches": []}
    values = []
    hook = model.enc.register_forward_pre_hook(
        lambda module, args: values.append(args[0].detach().clone())
    )
    native_dropout = F.dropout
    try:
        for batch in reference["batches"]:

            def dropout(
                value, p=0.5, training=True, inplace=False, mask=batch["dropout_mask"]
            ):
                if not training or p == 0:
                    return native_dropout(value, p, training, inplace)
                assert p == 0.5 and not inplace and value.shape == mask.shape
                return value * mask / (1 - p)

            # Original forward consumes only size here; kernels and sampled IDs
            # determine the first-order feature exactly, including duplicate IDs.
            adjs = [(None, None, (len(batch["ids"]), batch["targets"]))]
            with patch("torch.nn.functional.dropout", dropout):
                embedding, mask = model(graph, batch["ids"], adjs)
            loss = model.loss(embedding, mask)
            optimizer.zero_grad()
            loss.backward()
            gradients = {
                name: value.grad.detach().clone()
                for name, value in model.named_parameters()
                if value.grad is not None
            }
            optimizer.step()
            oracle["batches"].append(
                {
                    "ids": batch["ids"],
                    "targets": batch["targets"],
                    "dropout_mask": batch["dropout_mask"],
                    "features": values[-1],
                    "embedding": embedding.detach(),
                    "loss": loss.detach(),
                    "gradients": gradients,
                    "state_after": state(),
                }
            )
    finally:
        hook.remove()
    errors = compare_references(reference, oracle)
    torch.save(oracle, output)
    save_json(
        source / "precision_reference_receipt.json",
        {
            "source_sha256": sha256(Path(__file__)),
            "oracle_sha256": sha256(output),
            "training_input_sha256": sha256(source / "training_input.pt"),
            "original_float32_vs_float64_errors": errors,
            "rtol": 2e-4,
            "atol": 2e-5,
            "scope": "Original float64 arithmetic; fixed original float32 initial weights, inputs, kernels, IDs and masks",
            "training_dtype_changed": False,
        },
    )
    logging.info(
        "Original float32 passes unchanged tolerance against original float64 oracle"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marble-repo", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.marble_repo.resolve(), args.input.resolve())
