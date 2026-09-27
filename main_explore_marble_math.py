"""Fit without behavioral labels, seal outputs, then evaluate both MARBLE tasks."""

import argparse
import logging
import platform
import time
from pathlib import Path

import numpy as np
import torch

from models.marble_math_common import embedding_diagnostics
from models.network_marble_experiment import INPUT_KIND, NAME, MathematicalModel
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
RATS = ["achilles", "buddy", "cicero", "gatsby"]
SEEDS = [0, 1, 2]


def tensor_input(graph, embedding, device):
    value = (
        torch.cat([graph["pos"], graph["x"]], dim=1)
        if INPUT_KIND == "phase_space"
        else embedding
    )
    return value.to(device=device, dtype=torch.float64)


def fit_one(train, test, dimensions, seed, reference, path):
    model = MathematicalModel().fit(train, dimensions, seed, reference=reference)
    training = model.transform(train)
    outputs = {"train": training.cpu()}
    if test is not None:
        outputs["test"] = model.transform(test).cpu()
    assert training.shape == (len(train), dimensions)
    diagnostics = {**model.diagnostics, **embedding_diagnostics(training)}
    torch.save(model.state(), path / "fitted_operator.pt")
    torch.save(outputs, path / "embeddings.pt")
    save_json(path / "diagnostics.json", diagnostics)
    return diagnostics


def fit(input_path, rat_input, baseline, output, device):
    if output.exists():
        raise FileExistsError(output)
    assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pack = torch.load(
        input_path / "training_input.pt", weights_only=True, map_location="cpu"
    )
    provenance = read_json(baseline / "provenance.json")
    assert (
        sha256(input_path / "training_input.pt") == provenance["training_input_sha256"]
    )
    receipt = read_json(rat_input / "receipt.json")
    assert sha256(rat_input / "rat_consistency.pt") == receipt["bundle_sha256"]
    rats = torch.load(
        rat_input / "rat_consistency.pt", weights_only=True, map_location="cpu"
    )
    assert rats["rats"] == RATS
    output.mkdir(parents=True)
    sources = [
        Path(__file__),
        ROOT / "models/network_marble_experiment.py",
        ROOT / "models/marble_math_common.py",
        ROOT / "utils/utils_marble.py",
        ROOT / "utils/utils_marble_consistency.py",
        ROOT / "docs/MARBLE_MATH_PROTOCOL.md",
        ROOT / "uv.lock",
    ]
    protocol = {
        "method": NAME,
        "input_kind": INPUT_KIND,
        "seeds": SEEDS,
        "decoding_dimensions": 32,
        "consistency_dimensions": 3,
        "decoder": "fixed cosine kNN, k=36; no decoder tuning",
        "fit_uses_behavioral_labels": False,
        "hyperparameter_search": False,
        "baseline": str(baseline.resolve()),
        "baseline_sha256": {
            f"seed-{seed}/embeddings.pt": sha256(
                baseline / f"seed-{seed}/embeddings.pt"
            )
            for seed in SEEDS
        },
        "input": str(input_path.resolve()),
        "rat_input": str(rat_input.resolve()),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "input_sha256": sha256(input_path / "training_input.pt"),
        "rat_input_sha256": receipt["bundle_sha256"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(device),
        "interpretation": (
            "Exploratory on an already observed test split; "
            "not an independent confirmation cohort."
        ),
    }
    save_json(output / "protocol.json", protocol)
    reference32 = torch.load(baseline / "seed-0/embeddings.pt", weights_only=True)[
        "eval"
    ]["train"].to(device, torch.float64)
    reference3 = rats["animals"]["achilles"]["modes"]["eval"]["embedding"].to(
        device, torch.float64
    )
    diagnostics = {}
    started = time.perf_counter()
    for seed in SEEDS:
        path = output / f"seed-{seed}"
        path.mkdir()
        old = torch.load(baseline / f"seed-{seed}/embeddings.pt", weights_only=True)[
            "eval"
        ]
        train = tensor_input(pack["train_graph"], old["train"], device)
        test = tensor_input(pack["test_graph"], old["test"], device)
        target = path / "decoding"
        target.mkdir()
        diagnostics[f"{seed}/decoding"] = fit_one(
            train, test, 32, seed, reference32, target
        )
        for rat in RATS:
            animal = rats["animals"][rat]
            data = tensor_input(
                animal["graph"], animal["modes"]["eval"]["embedding"], device
            )
            target = path / rat
            target.mkdir()
            diagnostics[f"{seed}/{rat}"] = fit_one(
                data, None, 3, seed, reference3, target
            )
        logger.info(
            "%s seed %d: operators fitted; behavioral labels not evaluated", NAME, seed
        )
    torch.cuda.synchronize()
    outputs = list(output.glob("seed-*/*/embeddings.pt"))
    save_json(
        output / "fit_complete.json",
        {
            "method": NAME,
            "seconds": time.perf_counter() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "protocol_sha256": sha256(output / "protocol.json"),
            "outputs_sha256": {str(p.relative_to(output)): sha256(p) for p in outputs},
            "diagnostics": diagnostics,
        },
    )


def evaluate(output):
    torch.set_num_threads(4)
    assert not (output / "summary.json").exists(), "Evaluation already recorded"
    protocol, sealed = (
        read_json(output / "protocol.json"),
        read_json(output / "fit_complete.json"),
    )
    assert sha256(output / "protocol.json") == sealed["protocol_sha256"]
    for name, digest in sealed["outputs_sha256"].items():
        assert sha256(output / name) == digest
    source, rat_input, baseline = map(
        Path, (protocol["input"], protocol["rat_input"], protocol["baseline"])
    )
    assert sha256(source / "training_input.pt") == protocol["input_sha256"]
    assert sha256(rat_input / "rat_consistency.pt") == protocol["rat_input_sha256"]
    for name, digest in protocol["baseline_sha256"].items():
        assert sha256(baseline / name) == digest
    for name, digest in protocol["source_sha256"].items():
        assert sha256(ROOT / name) == digest
    pack = torch.load(source / "training_input.pt", weights_only=True)
    rats = torch.load(rat_input / "rat_consistency.pt", weights_only=True)
    labels = [rats["animals"][r]["labels"][:, 0].numpy() for r in RATS]
    base_embeddings = [
        rats["animals"][r]["modes"]["eval"]["embedding"].numpy() for r in RATS
    ]
    base_consistency, _ = consistency_scores(base_embeddings, labels, RATS)
    results = []
    for seed in SEEDS:
        current = torch.load(
            output / f"seed-{seed}/decoding/embeddings.pt", weights_only=True
        )
        old = torch.load(baseline / f"seed-{seed}/embeddings.pt", weights_only=True)[
            "eval"
        ]
        record = {"seed": seed}
        for name, values in (("baseline", old), ("candidate", current)):
            prediction = predict_position(
                values["train"].numpy(),
                values["test"].numpy(),
                pack["train_labels"].numpy(),
                neighbors=36,
            )
            record[name] = position_metrics(
                pack["test_labels"][:, 0].numpy(), prediction
            )
        embeddings = [
            torch.load(output / f"seed-{seed}/{r}/embeddings.pt", weights_only=True)[
                "train"
            ].numpy()
            for r in RATS
        ]
        record["consistency"], _ = consistency_scores(embeddings, labels, RATS)
        record["mae_change_cm"] = 100 * (
            record["candidate"]["mean_absolute_error_m"]
            - record["baseline"]["mean_absolute_error_m"]
        )
        record["consistency_change"] = (
            record["consistency"]["mean_r2"] - base_consistency["mean_r2"]
        )
        results.append(record)
    maes = np.array([r["candidate"]["mean_absolute_error_m"] * 100 for r in results])
    consistencies = np.array([r["consistency"]["mean_r2"] for r in results])
    summary = {
        "method": protocol["method"],
        "runs": results,
        "baseline_consistency": base_consistency,
        "candidate_mae_cm_mean": float(maes.mean()),
        "candidate_mae_cm_seed_sd": float(maes.std(ddof=1)),
        "baseline_mae_cm_mean": float(
            np.mean([r["baseline"]["mean_absolute_error_m"] * 100 for r in results])
        ),
        "candidate_consistency_mean": float(consistencies.mean()),
        "candidate_consistency_seed_sd": float(consistencies.std(ddof=1)),
        "all_seed_mae_improved": all(r["mae_change_cm"] < 0 for r in results),
        "all_seed_consistency_improved": all(
            r["consistency_change"] > 0 for r in results
        ),
        "interpretation": (
            "Three landmark/solver seeds; four shared animals; exploratory, "
            "no independent-subject significance claim."
        ),
    }
    save_json(output / "summary.json", summary)
    logger.info(
        "%s: MAE %.4f cm; cross-animal R2 %.6f",
        protocol["method"],
        maes.mean(),
        consistencies.mean(),
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "evaluate"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--rat-input", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.phase == "fit":
        assert args.input and args.rat_input and args.baseline
        fit(args.input, args.rat_input, args.baseline, args.output, args.device)
    else:
        evaluate(args.output)
