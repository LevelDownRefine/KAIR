"""Train fresh 3D MARBLE models for one whole-recording consistency condition."""

import argparse
import logging
import platform
import time
from pathlib import Path

import torch
from data.dataset_marble import DatasetMARBLE
from models.model_marble import ModelMARBLE
from models.network_marble import GraphFeatures
from scripts.marble.precision_reference import float32_reference
from utils.utils_marble import read_json, save_json, sha256

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def run(source, output):
    if output.exists():
        raise FileExistsError(output)
    assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = DatasetMARBLE({"dataroot": str(source)})
    assert (
        "protocol" in dataset.protocol and dataset.protocol["protocol"] == "consistency"
    )
    assert "animal" in dataset.protocol
    assert "out_channels" in dataset.params and dataset.params["out_channels"] == 3
    output.mkdir(parents=True)
    precision = read_json(source / "precision_reference_receipt.json")
    assert precision["oracle_sha256"] == sha256(source / "validation_float64.pt")
    assert precision["training_input_sha256"] == sha256(source / "training_input.pt")
    save_json(output / "precision_reference_receipt.json", precision)
    save_json(output / "protocol.json", dataset.protocol)
    save_json(output / "input_provenance.json", dataset.provenance)
    save_json(
        output / "provenance.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "training_input_sha256": sha256(source / "training_input.pt"),
            "validation_float64_sha256": precision["oracle_sha256"],
            "source_sha256": {
                path: sha256(ROOT / path)
                for path in (
                    "scripts/marble/train_consistency.py",
                    "scripts/marble/precision_reference.py",
                    "models/model_marble.py",
                    "models/network_marble.py",
                    "data/dataset_marble.py",
                    "utils/utils_marble_storage.py",
                    "pyproject.toml",
                    "uv.lock",
                )
            },
        },
    )
    options = {"params": dataset.params, "device": "cuda:0"}
    features = GraphFeatures(dataset.pack["train_graph"], "cuda:0")
    model = ModelMARBLE(options)
    oracle = torch.load(
        source / "validation_float64.pt", map_location="cpu", weights_only=True
    )
    reference = model.validate_reference(features, float32_reference(oracle))
    reference["comparison"] = (
        "float32 CUDA against original float64 oracle; original float32 separately checked at identical tolerance"
    )
    save_json(output / "reference_validation.json", reference)
    full = features()
    results = []
    started = time.perf_counter()
    for seed in dataset.seeds:
        destination = output / f"seed-{seed}"
        destination.mkdir()
        plan, initialization, sampling_hash = dataset.sampling(seed)
        model = ModelMARBLE(options)
        _, metrics = model.fit(features, plan, destination, seed)
        del plan
        with torch.no_grad():
            model.netG.eval()
            embedding = model.netG(full).cpu()
        assert embedding.shape == (len(full), 3) and torch.isfinite(embedding).all()
        torch.save(
            {"embedding": embedding, "labels": dataset.pack["train_labels"]},
            destination / "embeddings.pt",
        )
        metrics.update(
            scope="whole-recording 3D representation; no held-out decoding",
            sampling_sha256=sampling_hash,
        )
        save_json(destination / "initialization.json", initialization)
        save_json(destination / "metrics.json", metrics)
        results.append(metrics)
        logger.info("%s seed %d completed", dataset.protocol["animal"], seed)
    save_json(
        output / "summary.json",
        {
            "runs": results,
            "seeds": dataset.seeds,
            "seconds": time.perf_counter() - started,
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input.resolve(), args.output.resolve())
