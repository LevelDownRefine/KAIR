"""Train and decode MARBLE's exported Achilles protocol in KAIR."""

import argparse
import copy
import importlib.metadata
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data.select_dataset import define_Dataset
from models.network_marble import GraphFeatures
from models.select_model import define_Model
from utils.utils_marble import (
    plot_run,
    position_metrics,
    predict_position,
    read_json,
    save_json,
    sha256,
    summarize,
)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


def run(options, output):
    options = copy.deepcopy(options)
    required = ("task", "model", "device", "threads", "seeds", "dataset", "decoder")
    assert all(key in options for key in required)
    assert options["model"] == "marble"
    assert isinstance(options["threads"], int) and options["threads"] > 0
    seeds = options["seeds"]
    assert seeds and len(set(seeds)) == len(seeds)
    assert all(isinstance(seed, int) and seed >= 0 for seed in seeds)
    assert "neighbors" in options["decoder"] and "metric" in options["decoder"]
    assert options["decoder"]["metric"] == "cosine"
    device = torch.device(options["device"])
    assert device.type in ("cpu", "cuda")
    if device.type == "cuda":
        assert torch.cuda.is_available() and torch.version.cuda == "13.0"
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(options["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new run directory: {output}")
    dataset = define_Dataset(options["dataset"])
    assert set(seeds).issubset(dataset.seeds)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    save_json(output / "options.json", options)
    save_json(output / "protocol.json", dataset.protocol)
    save_json(output / "input_provenance.json", dataset.provenance)
    sources = [
        "main_train_marble.py",
        "data/dataset_marble.py",
        "models/network_marble.py",
        "models/model_marble.py",
        "utils/utils_marble.py",
        "pyproject.toml",
        "uv.lock",
        "scripts/marble/prepare.py",
        "scripts/marble/check_original.py",
        "data/select_dataset.py",
        "models/select_model.py",
    ]
    provenance = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "scipy", "scikit-learn")
        },
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "tf32": False,
        "sparse_cuda_bitwise_determinism_guaranteed": False,
        "training_input_sha256": sha256(dataset.root / "training_input.pt"),
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
    }
    save_json(output / "provenance.json", provenance)
    train = GraphFeatures(dataset.pack["train_graph"], device)
    test = GraphFeatures(dataset.pack["test_graph"], device)
    model_options = {"model": "marble", "params": dataset.params, "device": str(device)}
    model = define_Model(model_options)
    validation = model.validate_reference(train, dataset.pack["validation"])
    save_json(output / "reference_validation.json", validation)
    logger.info("Three original-code SGD steps matched; starting fresh training.")
    train_full, test_full = train(), test()
    truth = dataset.pack["test_labels"][:, 0].numpy()
    train_labels = dataset.pack["train_labels"].numpy()
    results = []
    for seed in seeds:
        destination = output / f"seed-{seed}"
        destination.mkdir()
        plan, initialization, sampling_hash = dataset.sampling(seed)
        save_json(destination / "initialization.json", initialization)
        model = define_Model(model_options)
        history, metrics = model.fit(train, plan, destination, seed)
        del plan
        embeddings = model.embeddings(train_full, test_full)
        torch.save(embeddings, destination / "embeddings.pt")
        metrics.update(
            scope="Achilles offline position decoding; KAIR MARBLE integration",
            primary_mode="eval",
            sampling_sha256=sampling_hash,
            results={},
        )
        arrays = {"truth_position_m": truth}
        for mode in ("eval", "notebook"):
            prediction, neighbors = predict_position(
                embeddings[mode]["train"].numpy(),
                embeddings[mode]["test"].numpy(),
                train_labels,
                options["decoder"]["neighbors"],
                return_neighbors=True,
            )
            metrics["results"][mode] = position_metrics(truth, prediction)
            arrays[f"prediction_{mode}"] = prediction
            arrays[f"neighbors_{mode}"] = neighbors
            arrays[f"embedding_{mode}"] = embeddings[mode]["test"].numpy()
        np.savez_compressed(destination / "arrays.npz", **arrays)
        plot_run(destination, history, truth, arrays["prediction_eval"])
        save_json(destination / "metrics.json", metrics)
        results.append(metrics)
        logger.info(
            "Seed %d: MAE %.4f cm; best epoch %d",
            seed,
            100 * metrics["results"]["eval"]["mean_absolute_error_m"],
            metrics["best_epoch_zero_based"] + 1,
        )
    summary = summarize(results)
    summary.update(
        seeds=seeds,
        seconds=time.perf_counter() - started,
        peak_allocated_gpu_bytes=(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    )
    save_json(output / "summary.json", summary)
    logger.info("Complete: %s", output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opt", type=Path, default=ROOT / "options/marble/train_achilles.json"
    )
    parser.add_argument(
        "--input", type=Path, help="Override the exported bundle directory"
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    configuration = read_json(arguments.opt)
    if arguments.input is not None:
        assert "dataset" in configuration
        configuration["dataset"]["dataroot"] = str(arguments.input.resolve())
    destination = arguments.output
    if destination is None:
        destination = (
            ROOT / "results" / datetime.now().strftime("marble-%Y%m%d-%H%M%S-%f")
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(configuration, destination)
