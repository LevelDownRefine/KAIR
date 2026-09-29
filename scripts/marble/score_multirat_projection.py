"""Aggregate every fixed-alpha run and independently audit consistency in CEBRA."""

import argparse
import importlib.metadata
import logging
from pathlib import Path

import numpy as np
import torch
from utils.utils_marble import read_json, save_json, sha256
from utils.utils_marble_consistency import consistency_scores

RATS = ("achilles", "buddy", "cicero", "gatsby")
CONDITIONS = ("pca", "joint-01")
SEEDS = (0, 1, 2)
ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def decoding_rows(source, animal, condition):
    assert (source / "original_code_audit.json").is_file()
    summary = read_json(source / "summary.json")
    assert "runs" in summary and "seeds" in summary
    assert summary["seeds"] == list(SEEDS)
    rows = []
    for run in summary["runs"]:
        assert "results" in run and "seed" in run
        assert "training_epochs" in run and run["training_epochs"] == 100
        for mode in ("eval", "notebook"):
            assert mode in run["results"]
            result = run["results"][mode]
            assert "mean_absolute_error_m" in result and "position_r2" in result
            rows.append(
                {
                    "animal": animal,
                    "condition": condition,
                    "seed": run["seed"],
                    "mode": mode,
                    "mae_cm": 100 * result["mean_absolute_error_m"],
                    "position_r2": result["position_r2"],
                    "best_epoch": run["best_epoch_zero_based"] + 1,
                }
            )
    assert len(rows) == 6
    return rows


def summarize_decoding(rows):
    summary = []
    for animal in RATS:
        for mode in ("eval", "notebook"):
            groups = {}
            for condition in CONDITIONS:
                values = sorted(
                    (
                        r
                        for r in rows
                        if r["animal"] == animal
                        and r["mode"] == mode
                        and r["condition"] == condition
                    ),
                    key=lambda r: r["seed"],
                )
                assert [r["seed"] for r in values] == list(SEEDS)
                groups[condition] = values
            baseline = np.array([r["mae_cm"] for r in groups["pca"]])
            candidate = np.array([r["mae_cm"] for r in groups["joint-01"]])
            summary.append(
                {
                    "animal": animal,
                    "mode": mode,
                    "discovery": animal == "achilles",
                    "pca_mean_cm": float(baseline.mean()),
                    "pca_sd_cm": float(baseline.std(ddof=1)),
                    "joint_mean_cm": float(candidate.mean()),
                    "joint_sd_cm": float(candidate.std(ddof=1)),
                    "delta_cm": float((candidate - baseline).mean()),
                    "relative_change_percent": float(
                        100 * (candidate.mean() / baseline.mean() - 1)
                    ),
                    "paired_seed_deltas_cm": (candidate - baseline).tolist(),
                    "seeds_improved": int((candidate < baseline).sum()),
                    "pca_mean_r2": float(
                        np.mean([r["position_r2"] for r in groups["pca"]])
                    ),
                    "joint_mean_r2": float(
                        np.mean([r["position_r2"] for r in groups["joint-01"]])
                    ),
                }
            )
    return summary


def load_embeddings(root, condition, seed):
    embeddings, labels, hashes = [], [], {}
    for animal in RATS:
        run = root / "consistency" / animal / condition / "run"
        assert (run / "original_code_audit.json").is_file()
        path = run / f"seed-{seed}/embeddings.pt"
        pack = torch.load(path, map_location="cpu", weights_only=True)
        assert "embedding" in pack and "labels" in pack
        expected_rows = 6576 if animal == "buddy" else 9999
        assert pack["embedding"].shape == (expected_rows, 3)
        assert len(pack["labels"]) == expected_rows
        embeddings.append(pack["embedding"].numpy())
        labels.append(pack["labels"][:, 0].numpy())
        hashes[animal] = sha256(path)
    return embeddings, labels, hashes


def novel_pair_mean(entry):
    assert "pairs" in entry and "scores" in entry
    values = [
        score
        for pair, score in zip(entry["pairs"], entry["scores"], strict=True)
        if "achilles" not in pair
    ]
    assert len(values) == 6
    return float(np.mean(values))


def aggregate(root, discovery):
    assert (root / "completed.json").is_file()
    rows = []
    for animal in RATS:
        for condition in CONDITIONS:
            source = (
                discovery / condition / "run"
                if animal == "achilles"
                else root / "decoding" / animal / condition / "run"
            )
            rows.extend(decoding_rows(source, animal, condition))
    summary = summarize_decoding(rows)
    new_animals = [r for r in summary if not r["discovery"] and r["mode"] == "eval"]
    consistency = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            embeddings, labels, hashes = load_embeddings(root, condition, seed)
            values, _ = consistency_scores(embeddings, labels, RATS)
            values["new_animal_pair_mean_r2"] = novel_pair_mean(values)
            consistency.append(
                dict(condition=condition, seed=seed, hashes=hashes, **values)
            )
    paired = [consistency[i + 3]["mean_r2"] - consistency[i]["mean_r2"] for i in SEEDS]
    result = {
        "decoding_rows": rows,
        "decoding_summary": summary,
        "new_animals_eval": {
            "n_animals": len(new_animals),
            "animals_improved": sum(r["delta_cm"] < 0 for r in new_animals),
            "pca_macro_mean_cm": float(
                np.mean([r["pca_mean_cm"] for r in new_animals])
            ),
            "joint_macro_mean_cm": float(
                np.mean([r["joint_mean_cm"] for r in new_animals])
            ),
            "mean_relative_change_percent": float(
                np.mean([r["relative_change_percent"] for r in new_animals])
            ),
            "paired_seeds_improved": sum(r["seeds_improved"] for r in new_animals),
        },
        "consistency": consistency,
        "consistency_summary": {
            condition: {
                "mean_r2": float(np.mean(values)),
                "sd_r2": float(np.std(values, ddof=1)),
                "seed_mean_r2": values,
            }
            for condition in CONDITIONS
            for values in [
                [r["mean_r2"] for r in consistency if r["condition"] == condition]
            ]
        },
        "paired_consistency_deltas": paired,
        "new_animal_pair_consistency_summary": {
            condition: {
                "mean_r2": float(np.mean(values)),
                "sd_r2": float(np.std(values, ddof=1)),
                "seed_mean_r2": values,
            }
            for condition in CONDITIONS
            for values in [
                [
                    r["new_animal_pair_mean_r2"]
                    for r in consistency
                    if r["condition"] == condition
                ]
            ]
        },
        "score_source_sha256": sha256(Path(__file__)),
        "caveat": "Seeds and animal pairs are not independent biological samples; descriptive statistics only",
    }
    save_json(root / "aggregate.json", result)
    logger.info("New animals: %s", result["new_animals_eval"])
    logger.info("Consistency: %s", result["consistency_summary"])
    return result


def audit_cebra(root):
    import cebra
    from cebra.integrations.sklearn.helpers import align_embeddings

    assert importlib.metadata.version("cebra") == "0.4.0"
    assert importlib.metadata.version("scikit-learn") == "1.3.2"
    recorded = read_json(root / "aggregate.json")
    assert "consistency" in recorded
    errors = []
    for entry in recorded["consistency"]:
        condition, seed = entry["condition"], entry["seed"]
        embeddings, labels, hashes = load_embeddings(root, condition, seed)
        assert hashes == entry["hashes"]
        values, pairs, _ = cebra.sklearn.metrics.consistency_score(
            embeddings=embeddings,
            labels=labels,
            dataset_ids=list(RATS),
            between="datasets",
        )
        assert pairs.tolist() == entry["pairs"]
        np.testing.assert_allclose(values, entry["scores"], atol=1e-6, rtol=1e-6)
        expected = align_embeddings(embeddings, labels)
        _, observed = consistency_scores(embeddings, labels, RATS)
        for actual, reference in zip(observed, expected, strict=True):
            np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-6)
        errors.append(
            {
                "condition": condition,
                "seed": seed,
                "max_r2_error": float(np.abs(values - entry["scores"]).max()),
            }
        )
    save_json(
        root / "cebra_metric_audit.json",
        {
            "cebra": "0.4.0",
            "sklearn": "1.3.2",
            "results": errors,
            "aggregate_sha256": sha256(root / "aggregate.json"),
            "source_sha256": sha256(Path(__file__)),
        },
    )
    logger.info("Independent CEBRA metric audit passed: %s", errors)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--discovery", type=Path, default=ROOT / "results/dynamics-projection-20260929"
    )
    parser.add_argument("--audit-cebra", action="store_true")
    args = parser.parse_args()
    if args.audit_cebra:
        audit_cebra(args.root.resolve())
    else:
        aggregate(args.root.resolve(), args.discovery.resolve())
