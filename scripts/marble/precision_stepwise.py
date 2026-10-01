"""Audit identical-state float32 steps and an independent full float64 rollout.

Float32 trajectories can amplify rounding over several SGD steps. Local checks
reset weights, BN buffers and momentum to the original pre-step state; they do
not claim that continuous float32 trajectories remain within that tolerance.
The independent original float64 trajectory is checked without state resets.
"""

import argparse
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from utils.utils_marble import read_json, save_json, sha256
from utils.utils_marble_storage import load_tensor_file

ROOT = Path(__file__).resolve().parents[2]


def original_oracle(repository, source, output):
    """Export original code in double precision, recording float32 drift honestly."""
    assert not output.exists()
    sys.path.insert(0, str(repository))
    import MARBLE
    from torch.nn import functional as F

    assert Path(MARBLE.__file__).resolve().parent == repository / "MARBLE"
    assert not torch.cuda.is_available()
    torch.set_num_threads(4)
    provenance = read_json(source / "provenance.json")
    assert (
        sha256(source / "original_graphs.pt.gz") == provenance["original_graphs_sha256"]
    )
    original = load_tensor_file(source / "original_graphs.pt.gz", weights_only=False)
    graph = original["train"]
    pack = torch.load(
        source / "training_input.pt", map_location="cpu", weights_only=True
    )
    reference = pack["validation"]
    model = MARBLE.net(graph, params=pack["params"], verbose=False).double().train()
    model.load_state_dict(reference["initial_state"], strict=True)
    graph.x, graph.pos = graph.x.double(), graph.pos.double()
    graph.kernels = [k.to(dtype=torch.float64) for k in graph.kernels]
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)

    def state():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    oracle = {"initial_state": state(), "batches": []}
    features = []
    hook = model.enc.register_forward_pre_hook(
        lambda module, args: features.append(args[0].detach().clone())
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

            adjs = [(None, None, (len(batch["ids"]), batch["targets"]))]
            with patch("torch.nn.functional.dropout", dropout):
                embedding, mask = model(graph, batch["ids"], adjs)
            loss = model.loss(embedding, mask)
            optimizer.zero_grad()
            loss.backward()
            gradients = {
                k: v.grad.detach().clone()
                for k, v in model.named_parameters()
                if v.grad is not None
            }
            optimizer.step()
            oracle["batches"].append(
                {
                    "ids": batch["ids"],
                    "targets": batch["targets"],
                    "dropout_mask": batch["dropout_mask"],
                    "features": features[-1],
                    "embedding": embedding.detach(),
                    "loss": loss.detach(),
                    "gradients": gradients,
                    "state_after": state(),
                }
            )
    finally:
        hook.remove()
    drift = []
    for step, (left, right) in enumerate(
        zip(reference["batches"], oracle["batches"], strict=True)
    ):
        for group in ("features", "embedding", "loss", "gradients", "state_after"):
            pairs = (
                [(group, left[group], right[group])]
                if not isinstance(left[group], dict)
                else [
                    (key, value, right[group][key])
                    for key, value in left[group].items()
                ]
            )
            for key, actual, expected in pairs:
                difference = (actual.double() - expected.double()).abs()
                limit = 2e-5 + 2e-4 * expected.double().abs()
                drift.append(
                    {
                        "step": step,
                        "group": group,
                        "name": key,
                        "max_absolute_error": float(difference.max()),
                        "outside_previous_tolerance": int((difference > limit).sum()),
                    }
                )
    output.mkdir(parents=True)
    torch.save(oracle, output / "original_float64.pt")
    save_json(
        output / "oracle_receipt.json",
        {
            "training_input_sha256": sha256(source / "training_input.pt"),
            "oracle_sha256": sha256(output / "original_float64.pt"),
            "source_sha256": sha256(Path(__file__)),
            "float32_trajectory_drift": drift,
            "scope": "Oracle export only; continuous original float32 drift is recorded, not silently accepted as a passed check",
        },
    )


def validate_steps(model, features, reference, *, reset, dtype, rtol, atol):
    from models.network_marble import contrastive_loss

    assert len(reference["batches"]) == 3
    model.netG.to(dtype=dtype).train()
    model.netG.load_state_dict(reference["initial_state"], strict=True)
    optimizer = torch.optim.SGD(model.netG.parameters(), lr=1.0, momentum=0.9)
    momentum, errors = {}, {}

    def compare(actual, expected, key):
        left, right = actual.detach().cpu().double(), expected.detach().cpu().double()
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
        errors[key] = float((left - right).abs().max())

    for step, batch in enumerate(reference["batches"]):
        if reset:
            before = (
                reference["initial_state"]
                if step == 0
                else reference["batches"][step - 1]["state_after"]
            )
            model.netG.load_state_dict(before, strict=True)
            optimizer = torch.optim.SGD(model.netG.parameters(), lr=1.0, momentum=0.9)
            for name, parameter in model.netG.named_parameters():
                if name in momentum:
                    optimizer.state[parameter]["momentum_buffer"] = (
                        momentum[name].to(device=model.device, dtype=dtype).clone()
                    )
        values = features(batch["ids"], batch["targets"])
        compare(values, batch["features"], f"{step}:features")
        embedding = model.netG(
            values, dropout_mask=batch["dropout_mask"].to(model.device)
        )
        compare(embedding, batch["embedding"], f"{step}:embedding")
        loss = contrastive_loss(embedding)
        compare(loss, batch["loss"], f"{step}:loss")
        optimizer.zero_grad()
        loss.backward()
        gradients = {
            k: v.grad for k, v in model.netG.named_parameters() if v.grad is not None
        }
        assert gradients.keys() == batch["gradients"].keys()
        for name, value in gradients.items():
            compare(value, batch["gradients"][name], f"{step}:gradient:{name}")
        optimizer.step()
        for name, value in model.netG.state_dict().items():
            compare(value, batch["state_after"][name], f"{step}:state:{name}")
        # Reconstruct the original CPU SGD buffers, including first-step semantics.
        for name, value in batch["gradients"].items():
            if name in momentum:
                momentum[name].mul_(0.9).add_(value)
            else:
                momentum[name] = value.clone()
    return {
        "reset_to_original_state_each_step": reset,
        "rtol": rtol,
        "atol": atol,
        "max_absolute_errors": errors,
    }


def validate(source, output):
    from models.model_marble import ModelMARBLE
    from models.network_marble import GraphFeatures

    assert not (output / "validation.json").exists()
    assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    receipt = read_json(output / "oracle_receipt.json")
    assert receipt["training_input_sha256"] == sha256(source / "training_input.pt")
    assert receipt["oracle_sha256"] == sha256(output / "original_float64.pt")
    pack = torch.load(
        source / "training_input.pt", map_location="cpu", weights_only=True
    )
    oracle = torch.load(
        output / "original_float64.pt", map_location="cpu", weights_only=True
    )
    options = {"params": pack["params"], "device": "cuda:0"}
    local = validate_steps(
        ModelMARBLE(options),
        GraphFeatures(pack["train_graph"], "cuda:0"),
        pack["validation"],
        reset=True,
        dtype=torch.float32,
        rtol=2e-4,
        atol=2e-5,
    )
    graph64 = {
        k: (v.double() if v.is_floating_point() else v)
        for k, v in pack["train_graph"].items()
        if isinstance(v, torch.Tensor)
    }
    continuous = validate_steps(
        ModelMARBLE(options),
        GraphFeatures(graph64, "cuda:0"),
        oracle,
        reset=False,
        dtype=torch.float64,
        rtol=1e-8,
        atol=1e-9,
    )
    save_json(
        output / "validation.json",
        {
            "passed": True,
            "training_input_sha256": receipt["training_input_sha256"],
            "oracle_sha256": receipt["oracle_sha256"],
            "source_sha256": sha256(Path(__file__)),
            "implementation_sha256": {
                name: sha256(ROOT / name)
                for name in ("models/network_marble.py", "models/model_marble.py")
            },
            "float32_local": local,
            "float64_continuous": continuous,
            "training_dtype_changed": False,
            "continuous_float32_trajectory_alignment_claimed": False,
        },
    )
    logging.info("Float32 local steps and complete float64 trajectory passed")


def checked_receipt(source, path):
    receipt = read_json(path)
    assert receipt["passed"] is True
    assert receipt["training_input_sha256"] == sha256(source / "training_input.pt")
    assert receipt["oracle_sha256"] == sha256(path.parent / "original_float64.pt")
    assert receipt["source_sha256"] == sha256(Path(__file__))
    for name, expected in receipt["implementation_sha256"].items():
        assert sha256(ROOT / name) == expected
    assert receipt["float32_local"]["reset_to_original_state_each_step"] is True
    assert receipt["float32_local"]["rtol"] == 2e-4
    assert receipt["float32_local"]["atol"] == 2e-5
    assert receipt["float64_continuous"]["reset_to_original_state_each_step"] is False
    assert receipt["float64_continuous"]["rtol"] == 1e-8
    assert receipt["float64_continuous"]["atol"] == 1e-9
    assert receipt["training_dtype_changed"] is False
    return receipt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marble-repo", type=Path)
    args = parser.parse_args()
    if args.marble_repo is not None:
        original_oracle(
            args.marble_repo.resolve(), args.input.resolve(), args.output.resolve()
        )
    else:
        validate(args.input.resolve(), args.output.resolve())
