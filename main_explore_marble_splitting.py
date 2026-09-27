"""Fit two provable proximal splits, then evaluate all sealed outputs once."""

import argparse
import copy
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from models.marble_geometry_prior import (
    GeometryGraph,
    make_graph,
    representation_diagnostics,
    sample_pairs,
)
from models.marble_provable_splitting import METHODS, SmoothResidual, train_latent
from models.network_marble import GraphFeatures, MARBLEEncoder
from utils.utils_marble import (
    position_metrics,
    predict_position,
    read_json,
    save_json,
    sha256,
)
from utils.utils_marble_consistency import consistency_scores

ROOT = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
RATS = ("achilles", "buddy", "cicero", "gatsby")
UPDATES = 300
logger = logging.getLogger(__name__)


def fit_task(
    params, state, graph_data, reference, test_data, test_reference, output, seed
):
    base = MARBLEEncoder(params).cuda().eval()
    base.load_state_dict(state, strict=True)
    with torch.no_grad():
        anchor = base(GraphFeatures(graph_data, "cuda")())
        torch.testing.assert_close(anchor.cpu(), reference, rtol=2e-4, atol=2e-5)
        test_anchor = None
        if test_data is not None:
            test_anchor = base(GraphFeatures(test_data, "cuda")())
            torch.testing.assert_close(
                test_anchor.cpu(), test_reference, rtol=2e-4, atol=2e-5
            )
    graph, neighbors, graph_info = make_graph(
        graph_data["pos"].cuda(), graph_data["x"].cuda()
    )
    graph = GeometryGraph(
        graph.source, graph.target, graph.weights.double(), graph.nodes
    )
    anchor = anchor.double()
    if test_anchor is not None:
        test_anchor = test_anchor.double()
    pairs = sample_pairs(neighbors, seed)
    torch.save(
        {"positive": pairs[0].cpu(), "negative": pairs[1].cpu()}, output / "pairs.pt"
    )
    initial = SmoothResidual(anchor.shape[1]).cuda()
    for method in ("initial", *METHODS):
        target = output / method
        target.mkdir()
        model = copy.deepcopy(initial)
        started = time.perf_counter()
        auxiliary = None
        diagnostic = {"history": []}
        if method != "initial":
            auxiliary, diagnostic = train_latent(
                model, anchor, graph, pairs, method, updates=UPDATES
            )
        torch.cuda.synchronize()
        diagnostic["fit_seconds"] = time.perf_counter() - started
        diagnostic["graph"] = graph_info
        with torch.no_grad():
            values = model(anchor)
            diagnostic["representation"] = representation_diagnostics(values)
            embeddings = {"train": values.cpu()}
            if test_anchor is not None:
                embeddings["test"] = model(test_anchor).cpu()
        torch.save(embeddings, target / "embeddings.pt")
        torch.save(
            {
                "params": params,
                "base_state": {key: value.cpu() for key, value in state.items()},
                "latent_state": {
                    key: value.cpu() for key, value in model.state_dict().items()
                },
                "channels": anchor.shape[1],
                "auxiliary": auxiliary.cpu() if auxiliary is not None else None,
                "epsilon": model.epsilon,
            },
            target / "model.pt",
        )
        save_json(target / "diagnostics.json", diagnostic)
        logger.info(
            "%s %s %.1fs rank=%d",
            output,
            method,
            diagnostic["fit_seconds"],
            diagnostic["representation"]["effective_rank"],
        )


def fit(input_path, rat_input, baseline, output):
    assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    assert not output.exists(), "Choose a fresh output directory"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    provenance = read_json(baseline / "provenance.json")
    receipt = read_json(rat_input / "receipt.json")
    assert (
        sha256(input_path / "training_input.pt") == provenance["training_input_sha256"]
    )
    assert sha256(rat_input / "rat_consistency.pt") == receipt["bundle_sha256"]
    pack = torch.load(input_path / "training_input.pt", weights_only=True)
    rats = torch.load(rat_input / "rat_consistency.pt", weights_only=True)
    assert tuple(rats["rats"]) == RATS
    sources = (
        "main_explore_marble_splitting.py",
        "models/marble_provable_splitting.py",
        "models/marble_geometry_prior.py",
        "models/network_marble.py",
        "utils/utils_marble.py",
        "utils/utils_marble_consistency.py",
        "docs/MARBLE_SPLITTING_PROTOCOL.md",
        "docs/MARBLE_SPLITTING_THEORY.md",
        "uv.lock",
    )
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "methods": METHODS,
        "seeds": SEEDS,
        "rats": RATS,
        "input": str(input_path.resolve()),
        "rat_input": str(rat_input.resolve()),
        "baseline": str(baseline.resolve()),
        "input_sha256": sha256(input_path / "training_input.pt"),
        "rat_input_sha256": sha256(rat_input / "rat_consistency.pt"),
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
        "selection": "final weights after 300 sweeps; deploy Z(theta), not V",
        "theory_gate_commit": "39a3cb1",
        "updates": UPDATES,
        "maximum_parameter_step": 16.0,
        "majorization_margin_parameter": 0.1,
        "eta": 1.0,
        "beta": 1e-4,
        "epsilon": 0.1,
        "precision": "float64 latent model; float32 encoder and graph construction",
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
            old["test"],
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
                None,
                target,
                seed,
            )
        save_json(
            output / "progress.json",
            {
                "seed_completed": seed,
                "seconds": time.perf_counter() - started,
            },
        )
    outputs = sorted(output.glob("seed-*/*/*/embeddings.pt"))
    assert len(outputs) == 60
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
    protocol = read_json(output / "protocol.json")
    seal = read_json(output / "fit_complete.json")
    assert sha256(output / "protocol.json") == seal["protocol_sha256"]
    for name, digest in seal["outputs_sha256"].items():
        assert sha256(output / name) == digest
    for name, digest in protocol["source_sha256"].items():
        assert sha256(ROOT / name) == digest
    input_path, rat_input, baseline = map(
        Path,
        (
            protocol["input"],
            protocol["rat_input"],
            protocol["baseline"],
        ),
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
    for method in ("baseline", "initial", *METHODS):
        runs = []
        for seed in SEEDS:
            if method == "baseline":
                values = torch.load(
                    baseline / f"seed-{seed}/embeddings.pt",
                    weights_only=True,
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
        logger.info("%s MAE=%.6f cm consistency=%.7f", method, mae.mean(), r2.mean())
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
