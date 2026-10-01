"""Diagnose floating-point reference failures without training or relaxing checks."""

import argparse
import logging
from pathlib import Path

import torch
from models.model_marble import ModelMARBLE
from models.network_marble import GraphFeatures, contrastive_loss
from utils.utils_marble import save_json

logger = logging.getLogger(__name__)


def diagnose(source, output):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pack = torch.load(
        source / "training_input.pt", weights_only=True, map_location="cpu"
    )
    validation = pack["validation"]
    results = {}
    for dtype in (torch.float32, torch.float64):
        for device in ("cpu", "cuda:0"):
            graph = {
                name: value.to(dtype) if value.is_floating_point() else value
                for name, value in pack["train_graph"].items()
                if isinstance(value, torch.Tensor)
            }
            features = GraphFeatures(graph, device)
            for use_reference in (False, True):
                label = f"{dtype}/{device}/reference_features={use_reference}"
                model = ModelMARBLE({"params": pack["params"], "device": device})
                model.netG.to(dtype=dtype)
                model.netG.load_state_dict(validation["initial_state"], strict=True)
                model.netG.train()
                optimizer = torch.optim.SGD(
                    model.netG.parameters(), lr=1.0, momentum=0.9
                )
                record = {"failures": [], "maximum_errors": {}, "raw_norm_min": []}

                def compare(actual, expected, name, captured=record):
                    left = actual.detach().cpu().double()
                    right = expected.detach().cpu().double()
                    difference = (left - right).abs()
                    captured["maximum_errors"][name] = float(difference.max())
                    failures = difference > 2e-5 + 2e-4 * right.abs()
                    if failures.any():
                        captured["failures"].append(
                            {
                                "name": name,
                                "count": int(failures.sum()),
                                "max_absolute_error": float(difference.max()),
                                "max_tolerance_ratio": float(
                                    (difference / (2e-5 + 2e-4 * right.abs())).max()
                                ),
                            }
                        )

                raw_values = []
                hook = model.netG.enc.register_forward_hook(
                    lambda module, inputs, result, storage=raw_values: storage.append(
                        result.detach().cpu()
                    )
                )
                try:
                    for step, batch in enumerate(validation["batches"]):
                        values = (
                            batch["features"].to(device=device, dtype=dtype)
                            if use_reference
                            else features(batch["ids"], batch["targets"])
                        )
                        compare(values, batch["features"], f"{step}:features")
                        embedding = model.netG(
                            values, dropout_mask=batch["dropout_mask"].to(device)
                        )
                        record["raw_norm_min"].append(
                            float(raw_values[-1].norm(dim=1).min())
                        )
                        compare(embedding, batch["embedding"], f"{step}:embedding")
                        loss = contrastive_loss(embedding)
                        compare(loss, batch["loss"], f"{step}:loss")
                        optimizer.zero_grad()
                        loss.backward()
                        for name, parameter in model.netG.named_parameters():
                            if parameter.grad is not None:
                                assert name in batch["gradients"]
                                compare(
                                    parameter.grad,
                                    batch["gradients"][name],
                                    f"{step}:gradient:{name}",
                                )
                        optimizer.step()
                        for name, value in model.netG.state_dict().items():
                            compare(
                                value,
                                batch["state_after"][name],
                                f"{step}:state:{name}",
                            )
                finally:
                    hook.remove()
                results[label] = record
                logger.info(
                    "%s: %s; minimum raw norms %s",
                    label,
                    record["failures"],
                    record["raw_norm_min"],
                )
    save_json(output, results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.input, args.output)
