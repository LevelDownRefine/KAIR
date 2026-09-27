"""Label-free encoder fitting and separately sealed evaluation of geometry priors."""

import argparse
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from models.marble_geometry_prior import make_graph, sample_pairs
from models.marble_prior_solver import METHODS, optimize
from models.network_marble import GraphFeatures, MARBLEEncoder
from utils.utils_marble import (
    position_metrics,
    predict_position,
    read_json,
    save_json,
    sha256,
)
from utils.utils_marble_consistency import consistency_scores

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
SEEDS = [0, 1, 2]
RATS = ["achilles", "buddy", "cicero", "gatsby"]


def fit_task(params, state, graph_data, reference, test_data, output, seed):
    device = "cuda:0"
    features = GraphFeatures(graph_data, device)()
    test_features = (
        GraphFeatures(test_data, device)() if test_data is not None else None
    )
    graph, neighbors, graph_info = make_graph(
        graph_data["pos"].to(device), graph_data["x"].to(device)
    )
    pairs = sample_pairs(neighbors, seed)
    torch.save(
        {
            "source": graph.source.cpu(),
            "target": graph.target.cpu(),
            "weights": graph.weights.cpu(),
            "positive": pairs[0].cpu(),
            "negative": pairs[1].cpu(),
        },
        output / "graph_and_pairs.pt",
    )
    for method in METHODS:
        target = output / method
        target.mkdir()
        model = MARBLEEncoder(params).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                model(features).cpu(), reference, rtol=2e-4, atol=2e-5
            )
        started = time.perf_counter()
        diagnostic = optimize(model, features, graph, pairs, method)
        torch.cuda.synchronize()
        diagnostic.update(seconds=time.perf_counter() - started, graph=graph_info)
        model.eval()
        with torch.no_grad():
            embeddings = {"train": model(features).cpu()}
            buffers = {name: value.clone() for name, value in model.named_buffers()}
            if test_features is not None:
                embeddings["test"] = model(test_features).cpu()
            for name, value in model.named_buffers():
                torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)
        torch.save(embeddings, target / "embeddings.pt")
        torch.save(
            {
                "params": params,
                "state": {
                    key: value.cpu() for key, value in model.state_dict().items()
                },
            },
            target / "model.pt",
        )
        save_json(target / "diagnostics.json", diagnostic)
        last = diagnostic["history"][-1]
        logger.info(
            "%s %s %.1fs loss=%.5f rank=%d",
            output.name,
            method,
            diagnostic["seconds"],
            last["contrastive"],
            diagnostic["final"]["effective_rank"],
        )


def fit(input_path, rat_input, baseline, output):
    assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    assert not output.exists(), "Choose a fresh output directory"
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    provenance = read_json(baseline / "provenance.json")
    assert (
        sha256(input_path / "training_input.pt") == provenance["training_input_sha256"]
    )
    receipt = read_json(rat_input / "receipt.json")
    assert sha256(rat_input / "rat_consistency.pt") == receipt["bundle_sha256"]
    pack = torch.load(input_path / "training_input.pt", weights_only=True)
    rats = torch.load(rat_input / "rat_consistency.pt", weights_only=True)
    assert rats["rats"] == RATS
    sources = [
        "main_explore_marble_priors.py",
        "models/marble_geometry_prior.py",
        "models/marble_prior_solver.py",
        "models/network_marble.py",
        "utils/utils_marble.py",
        "utils/utils_marble_consistency.py",
        "docs/MARBLE_PRIOR_PROTOCOL.md",
        "uv.lock",
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "methods": METHODS,
        "seeds": SEEDS,
        "rats": RATS,
        "input": str(input_path.resolve()),
        "rat_input": str(rat_input.resolve()),
        "baseline": str(baseline.resolve()),
        "input_sha256": provenance["training_input_sha256"],
        "rat_input_sha256": receipt["bundle_sha256"],
        "baseline_sha256": {
            f"seed-{seed}/{name}": sha256(baseline / f"seed-{seed}/{name}")
            for seed in SEEDS
            for name in ("embeddings.pt", "best_model.pth")
        },
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "behavioral_labels_used_in_fit": False,
        "selection": "fixed final iterate; no behavioral metric selection",
        "scope": "Exploratory warm-start continuation, not from-scratch reproduction",
    }
    output.mkdir(parents=True)
    save_json(output / "protocol.json", protocol)
    started = time.perf_counter()
    for seed in SEEDS:
        root = output / f"seed-{seed}"
        root.mkdir()
        old = torch.load(baseline / f"seed-{seed}/embeddings.pt", weights_only=True)[
            "eval"
        ]
        checkpoint = torch.load(
            baseline / f"seed-{seed}/best_model.pth", weights_only=True
        )
        target = root / "decoding"
        target.mkdir()
        fit_task(
            pack["params"],
            checkpoint["model_state_dict"],
            pack["train_graph"],
            old["train"],
            pack["test_graph"],
            target,
            seed,
        )
        for rat in RATS:
            animal = rats["animals"][rat]
            target = root / rat
            target.mkdir()
            fit_task(
                animal["params"],
                animal["state"],
                animal["graph"],
                animal["modes"]["eval"]["embedding"],
                None,
                target,
                seed,
            )
        save_json(
            output / "progress.json",
            {"seed_completed": seed, "elapsed_seconds": time.perf_counter() - started},
        )
    outputs = sorted(output.glob("seed-*/*/*/embeddings.pt"))
    assert len(outputs) == len(SEEDS) * (1 + len(RATS)) * len(METHODS)
    save_json(
        output / "fit_complete.json",
        {
            "seconds": time.perf_counter() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "protocol_sha256": sha256(output / "protocol.json"),
            "outputs_sha256": {str(p.relative_to(output)): sha256(p) for p in outputs},
        },
    )


def evaluate(output):
    torch.set_num_threads(4)
    assert not (output / "summary.json").exists(), "Evaluation already recorded"
    protocol, seal = (
        read_json(output / "protocol.json"),
        read_json(output / "fit_complete.json"),
    )
    assert sha256(output / "protocol.json") == seal["protocol_sha256"]
    for name, digest in seal["outputs_sha256"].items():
        assert sha256(output / name) == digest
    for name, digest in protocol["source_sha256"].items():
        assert sha256(ROOT / name) == digest
    input_path, rat_input, baseline = map(
        Path, (protocol["input"], protocol["rat_input"], protocol["baseline"])
    )
    assert sha256(input_path / "training_input.pt") == protocol["input_sha256"]
    assert sha256(rat_input / "rat_consistency.pt") == protocol["rat_input_sha256"]
    for name, digest in protocol["baseline_sha256"].items():
        assert sha256(baseline / name) == digest
    pack = torch.load(input_path / "training_input.pt", weights_only=True)
    rats = torch.load(rat_input / "rat_consistency.pt", weights_only=True)
    labels = [rats["animals"][r]["labels"][:, 0].numpy() for r in RATS]
    base_consistency, _ = consistency_scores(
        [rats["animals"][r]["modes"]["eval"]["embedding"].numpy() for r in RATS],
        labels,
        RATS,
    )
    results = {}
    for method in ("baseline", *METHODS):
        runs = []
        for seed in SEEDS:
            if method == "baseline":
                values = torch.load(
                    baseline / f"seed-{seed}/embeddings.pt", weights_only=True
                )["eval"]
                consistency = base_consistency
            else:
                values = torch.load(
                    output / f"seed-{seed}/decoding/{method}/embeddings.pt",
                    weights_only=True,
                )
                embeddings = [
                    torch.load(
                        output / f"seed-{seed}/{rat}/{method}/embeddings.pt",
                        weights_only=True,
                    )["train"].numpy()
                    for rat in RATS
                ]
                consistency, _ = consistency_scores(embeddings, labels, RATS)
            prediction = predict_position(
                values["train"].numpy(),
                values["test"].numpy(),
                pack["train_labels"].numpy(),
                neighbors=36,
            )
            decoding = position_metrics(pack["test_labels"][:, 0].numpy(), prediction)
            runs.append(
                {"seed": seed, "decoding": decoding, "consistency": consistency}
            )
        mae = np.array([run["decoding"]["mean_absolute_error_m"] * 100 for run in runs])
        r2 = np.array([run["consistency"]["mean_r2"] for run in runs])
        results[method] = {
            "runs": runs,
            "mae_cm_mean": float(mae.mean()),
            "mae_cm_sd": float(mae.std(ddof=1)),
            "consistency_mean": float(r2.mean()),
            "consistency_sd": float(r2.std(ddof=1)),
        }
        logger.info("%s MAE=%.5f cm consistency=%.6f", method, mae.mean(), r2.mean())
    save_json(output / "summary.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "evaluate"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--rat-input", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.phase == "fit":
        assert args.input and args.rat_input and args.baseline
        fit(args.input, args.rat_input, args.baseline, args.output)
    else:
        evaluate(args.output)
