"""Audit four-rat author checkpoints on CUDA 13 and recompute directed R2."""

import argparse
import importlib.metadata
import logging
import time
from pathlib import Path

import numpy as np
import torch

from data.dataset_marble import validate_graph
from models.model_marble import ModelMARBLE, cpu_state
from models.network_marble import GraphFeatures
from utils.utils_marble import read_json, save_json, sha256, tensor_digest
from utils.utils_marble_consistency import consistency_scores

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
RATS = ["achilles", "buddy", "cicero", "gatsby"]
RTOL, ATOL = 2e-4, 2e-5


def load_bundle(source):
    source = Path(source)
    receipt = read_json(source / "receipt.json")
    assert "bundle_sha256" in receipt
    assert sha256(source / "rat_consistency.pt") == receipt["bundle_sha256"], (
        "Reference bundle hash mismatch"
    )
    pack = torch.load(
        source / "rat_consistency.pt", map_location="cpu", weights_only=True
    )
    assert all(key in pack for key in ("rats", "animals", "reference_consistency"))
    assert pack["rats"] == RATS and list(pack["animals"]) == RATS
    for rat in RATS:
        animal = pack["animals"][rat]
        assert all(
            key in animal
            for key in (
                "graph",
                "params",
                "labels",
                "state",
                "modes",
                "validation",
                "features",
                "baselines",
                "baseline_labels",
            )
        )
        validate_graph(animal["graph"])
        assert len(animal["labels"]) == len(animal["graph"]["pos"])
        assert animal["labels"].ndim == 2 and torch.isfinite(animal["labels"]).all()
        assert list(animal["modes"]) == [
            "eval",
            "batch_stats",
            "notebook_seed0",
            "notebook_seed1",
            "notebook_seed2",
        ]
        assert set(animal["baselines"]) == {"CEBRA-time", "CEBRA-behaviour"}
        for mode, reference in animal["modes"].items():
            assert "embedding" in reference and "state_after" in reference
            assert reference["embedding"].shape == (len(animal["labels"]), 3)
            assert torch.isfinite(reference["embedding"]).all()
            assert ("dropout_mask" in reference) == mode.startswith("notebook")
        assert "state_sha256" in receipt and rat in receipt["state_sha256"]
        assert tensor_digest(animal["state"].items()) == receipt["state_sha256"][rat]
    return pack, receipt


def compare(actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    return float((actual - expected).abs().max())


@torch.no_grad()
def evaluate_modes(network, features, state, modes):
    outputs, errors = {}, {}
    probability = network.enc.dropout
    try:
        for name, reference in modes.items():
            network.load_state_dict(state, strict=True)
            network.train(name != "eval")
            network.enc.dropout = 0.0 if name == "batch_stats" else probability
            mask = (
                reference["dropout_mask"].to(features.device)
                if "dropout_mask" in reference
                else None
            )
            embedding = network(features, dropout_mask=mask).cpu()
            error = compare(embedding, reference["embedding"])
            actual_state = network.state_dict()
            assert actual_state.keys() == reference["state_after"].keys()
            state_error = max(
                compare(value, reference["state_after"][key])
                for key, value in actual_state.items()
            )
            outputs[name] = embedding
            errors[name] = {
                "embedding_max_abs_error": error,
                "state_max_abs_error": state_error,
            }
    finally:
        network.enc.dropout = probability
        network.load_state_dict(state, strict=True)
        network.eval()
    return outputs, errors


def plot_comparison(output, results, animals):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].barh(names, [results[name]["mean_r2"] for name in names], color="#24788c")
    axes[0].set(xlabel="Mean directed in-sample R²", xlim=(0, 1))
    axes[0].invert_yaxis()
    matrix = np.full((4, 4), np.nan)
    for pair, score in zip(
        results["MARBLE-eval"]["pairs"], results["MARBLE-eval"]["scores"], strict=True
    ):
        matrix[animals.index(pair[0]), animals.index(pair[1])] = score
    plot = axes[1].imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axes[1].set(
        xticks=range(4),
        xticklabels=animals,
        yticks=range(4),
        yticklabels=animals,
        xlabel="Target animal",
        ylabel="Source animal",
        title="MARBLE eval: 12 directed pairs",
    )
    fig.colorbar(plot, ax=axes[1], label="R²")
    fig.savefig(output / "consistency.png", dpi=160)
    plt.close(fig)


def plot_embeddings(output, groups, pack):
    import matplotlib.pyplot as plt

    methods = ["MARBLE-eval", "CEBRA-time", "CEBRA-behaviour"]
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    for row, name in enumerate(methods):
        for col, rat in enumerate(RATS):
            axis = fig.add_subplot(3, 4, row * 4 + col + 1, projection="3d")
            embedding = groups[name][col]
            key = "labels" if name == "MARBLE-eval" else "baseline_labels"
            labels = pack["animals"][rat][key].numpy()
            for direction, cmap in ((1, "cool"), (2, "viridis")):
                selected = labels[:, direction] == 1
                axis.scatter(
                    *embedding[selected].T,
                    c=labels[selected, 0],
                    cmap=cmap,
                    vmin=0,
                    vmax=1.6,
                    s=0.6,
                    alpha=0.4,
                    rasterized=True,
                )
            axis.set_title(f"{rat} / {name}", fontsize=9)
            axis.set_axis_off()
    fig.suptitle(
        "Author checkpoints | color: position and direction | MARBLE on CUDA 13"
    )
    fig.savefig(output / "embeddings.png", dpi=160)
    plt.close(fig)


def run(source, output, device="cuda:0"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    device = torch.device(device)
    assert device.type in ("cpu", "cuda")
    if device.type == "cuda":
        assert torch.cuda.is_available() and torch.version.cuda == "13.0"
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pack, receipt = load_bundle(source)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "input_receipt.json", receipt)
    started = time.perf_counter()
    groups, labels, audits, arrays = {}, {}, {}, {}
    for rat in RATS:
        animal = pack["animals"][rat]
        model = ModelMARBLE({"params": animal["params"], "device": str(device)})
        features = GraphFeatures(animal["graph"], device)
        full = features()
        feature_error = compare(full, animal["features"])
        updates = model.validate_reference(features, animal["validation"])
        embeddings, mode_audit = evaluate_modes(
            model.netG, full, animal["state"], animal["modes"]
        )
        assert tensor_digest(cpu_state(model.netG).items()) == tensor_digest(
            animal["state"].items()
        )
        audits[rat] = {
            "feature_max_abs_error": feature_error,
            "updates": updates,
            "modes": mode_audit,
        }
        for name, embedding in embeddings.items():
            key = f"MARBLE-{name}"
            if key not in groups:
                groups[key], labels[key] = [], []
            groups[key].append(embedding.numpy())
            labels[key].append(animal["labels"][:, 0].numpy())
            arrays[f"{key}_{rat}"] = embedding.numpy()
        for name, embedding in animal["baselines"].items():
            if name not in groups:
                groups[name], labels[name] = [], []
            groups[name].append(embedding.numpy())
            labels[name].append(animal["baseline_labels"][:, 0].numpy())
            arrays[f"{name}_{rat}"] = embedding.numpy()
        arrays[f"MARBLE_labels_{rat}"] = animal["labels"].numpy()
        arrays[f"CEBRA_labels_{rat}"] = animal["baseline_labels"].numpy()
        save_json(output / "reference_validation.json", audits)
        logger.info(
            "%s %s/reference features, gradients, updates and embeddings matched",
            rat,
            device,
        )
        del features, full, model
    results = {}
    for name, embeddings in groups.items():
        result, aligned = consistency_scores(embeddings, labels[name], RATS)
        reference = pack["reference_consistency"][name]
        assert result["pairs"] == reference["pairs"]
        alignment_error = max(
            compare(torch.from_numpy(a), b)
            for a, b in zip(aligned, reference["aligned"], strict=True)
        )
        score_error = compare(
            torch.tensor(result["scores"], dtype=torch.float64), reference["scores"]
        )
        result.update(
            alignment_max_abs_error=alignment_error,
            original_score_max_abs_error=score_error,
        )
        results[name] = result
        logger.info(
            "%s mean R2 %.6f, reference score error %.3g",
            name,
            result["mean_r2"],
            score_error,
        )
    summary = {
        "scope": "four animals; public author checkpoints; no representation training",
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "rats": RATS,
        "pairs_per_method": 12,
        "rtol": RTOL,
        "atol": ATOL,
        "results": results,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else None,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "scipy", "scikit-learn")
        },
        "source_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "main_test_marble_consistency.py",
                "models/network_marble.py",
                "models/model_marble.py",
                "utils/utils_marble_consistency.py",
                "data/dataset_marble.py",
                "utils/utils_marble.py",
                "uv.lock",
            )
        },
        "limitations": [
            "In-sample, position-only binned consistency; not held-out generalization.",
            "Twelve directed pairs share four animals; pairs are not independent.",
            "CEBRA embeddings use original CPU code; KAIR recomputes the metric.",
            "Notebook dropout shares CPU-generated masks for CUDA numerical parity.",
            "batch_stats disables dropout as a diagnostic, outside notebook protocol.",
        ],
    }
    np.savez_compressed(output / "embeddings.npz", **arrays)
    save_json(output / "summary.json", summary)
    plot_comparison(output, results, RATS)
    plot_embeddings(output, groups, pack)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.input, args.output, args.device)
