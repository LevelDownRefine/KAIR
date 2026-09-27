"""Search finite RRR hyperparameters with fixed stability and validation rules."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from models.marble_math_common import PhaseDictionary
from models.marble_rrr_theory import choose_ridge, rrr, spectral_certificate
from utils.utils_marble import read_json, save_json, sha256

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
LAGS = (2, 4, 8, 16, 32)
BUDGETS = (3, 10, 30, 100, 300)
RATS = ("achilles", "buddy", "cicero", "gatsby")
PURGE = 64


def split_ranges(length):
    stop = int(0.6 * length)
    validation_start = stop + PURGE
    assert stop > 2 * max(LAGS) and length - validation_start > max(LAGS)
    return stop, validation_start


def lagged(values, lag):
    assert 0 < lag < len(values)
    return values[:-lag], values[lag:]


def moments(values, lag):
    x, y = lagged(values, lag)
    mx, my = x.mean(0), y.mean(0)
    x, y = x - mx, y - my
    return x.T @ x / len(x), x.T @ y / len(x), mx, my


def evaluate_candidate(training, validation, lag, budget, rank, cached=None):
    if cached is None:
        half = len(training) // 2
        cached = [
            moments(training, lag),
            moments(training[:half], lag),
            moments(training[half:], lag),
        ]
    c, cross, mx, my = cached[0]
    ridge = choose_ridge(c, condition_budget=budget)
    fitted = rrr(c, cross, rank, ridge)
    assert all(k in fitted for k in ("h", "coefficient", "condition"))
    certificates = []
    for other_c, other_cross, _, _ in cached[1:]:
        changed = rrr(other_c, other_cross, rank, ridge)
        assert "h" in changed
        certificates.append(spectral_certificate(fitted["h"], changed["h"], rank))
    x, y = lagged(validation, lag)
    prediction = (x - mx) @ fitted["coefficient"] + my
    error = float((prediction - y).square().mean())
    persistence = float((x - y).square().mean())
    assert persistence > 0
    return {
        "lag": lag,
        "condition_budget": budget,
        "ridge": float(ridge),
        "condition_number": float(fitted["condition"]),
        "certificates": certificates,
        "validation_neural_mse": error,
        "validation_persistence_mse": persistence,
        "validation_error_ratio": error / persistence,
        "training_pairs": len(training) - lag,
        "validation_pairs": len(validation) - lag,
    }


def summarize(records):
    assert records
    summaries = []
    for lag in LAGS:
        for budget in BUDGETS:
            group = []
            for record in records:
                assert "lag" in record and "condition_budget" in record
                if record["lag"] == lag and record["condition_budget"] == budget:
                    group.append(record)
            assert len(group) == 15
            certificates, ratios = [], []
            for row in group:
                assert "certificates" in row and "validation_error_ratio" in row
                for certificate in row["certificates"]:
                    assert "nontrivial_certificate" in certificate
                    certificates.append(certificate["nontrivial_certificate"])
                ratios.append(row["validation_error_ratio"])
            assert len(certificates) == 30
            summaries.append(
                {
                    "lag": lag,
                    "condition_budget": budget,
                    "certificates_passed": sum(certificates),
                    "certificates_total": len(certificates),
                    "eligible": all(certificates),
                    "validation_mean_error_ratio": float(np.mean(ratios)),
                }
            )
    eligible = []
    for row in summaries:
        assert "eligible" in row
        if row["eligible"]:
            eligible.append(row)
    selected = (
        min(
            eligible,
            key=lambda r: (
                r["validation_mean_error_ratio"],
                r["lag"],
                r["condition_budget"],
            ),
        )
        if eligible
        else None
    )
    return {
        "candidates": summaries,
        "selected": selected,
        "behavioral_benchmark_run": False,
        "records": records,
    }


@torch.no_grad()
def search(input_path, rat_input, original_run, output, device):
    assert torch.cuda.is_available() and torch.version.cuda == "13.0"
    assert not output.exists(), "Preserve prior search outputs"
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    provenance = read_json(original_run / "protocol.json")
    assert "input_sha256" in provenance and "rat_input_sha256" in provenance
    assert sha256(input_path / "training_input.pt") == provenance["input_sha256"]
    assert sha256(rat_input / "rat_consistency.pt") == provenance["rat_input_sha256"]
    pack = torch.load(input_path / "training_input.pt", weights_only=True)
    rats = torch.load(rat_input / "rat_consistency.pt", weights_only=True)
    assert "train_graph" in pack and "animals" in rats
    output.mkdir(parents=True)
    source_files = [
        Path(__file__),
        ROOT / "models/marble_rrr_theory.py",
        ROOT / "models/marble_math_common.py",
        ROOT / "docs/MARBLE_SEARCH_PROTOCOL.md",
        ROOT / "uv.lock",
    ]
    save_json(
        output / "protocol.json",
        {
            "lags": LAGS,
            "condition_budgets": BUDGETS,
            "seeds": [0, 1, 2],
            "behavioral_labels_used": False,
            "purge_samples": PURGE,
            "training_fraction": 0.6,
            "candidate_count": len(LAGS) * len(BUDGETS),
            "inputs": {
                "achilles": provenance["input_sha256"],
                "rats": provenance["rat_input_sha256"],
            },
            "source_sha256": {
                str(p.relative_to(ROOT)): sha256(p) for p in source_files
            },
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    )
    started = time.perf_counter()
    records = []
    for seed in (0, 1, 2):
        for task in ("decoding", *RATS):
            if task == "decoding":
                graph, rank = pack["train_graph"], 32
            else:
                assert task in rats["animals"] and "graph" in rats["animals"][task]
                graph, rank = rats["animals"][task]["graph"], 3
            assert "pos" in graph and "x" in graph
            phase = torch.cat([graph["pos"], graph["x"]], dim=1).to(
                device, torch.float64
            )
            stop, validation_start = split_ranges(len(phase))
            dictionary = PhaseDictionary().fit(phase[:stop], seed)
            training, validation = [
                dictionary.kernel(v) for v in (phase[:stop], phase[validation_start:])
            ]
            training /= training.sum(1, keepdim=True)
            validation /= validation.sum(1, keepdim=True)
            for lag in LAGS:
                half = len(training) // 2
                cached = [
                    moments(v, lag)
                    for v in (training, training[:half], training[half:])
                ]
                for budget in BUDGETS:
                    result = evaluate_candidate(
                        training, validation, lag, budget, rank, cached=cached
                    )
                    records.append({"seed": seed, "task": task, "rank": rank, **result})
                logger.info(
                    "seed=%d task=%s lag=%d finished five budgets", seed, task, lag
                )
            save_json(output / f"seed-{seed}-{task}.json", records[-25:])
    result = summarize(records)
    result["seconds"] = time.perf_counter() - started
    save_json(output / "search.json", result)
    assert "selected" in result
    logger.info("Search complete: selected=%s", result["selected"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rat-input", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    search(args.input, args.rat_input, args.original_run, args.output, args.device)
