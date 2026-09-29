"""Summarize every grid point, common-alpha selection and retrospective LOAO."""

import argparse
import importlib.metadata
import logging
from pathlib import Path

import numpy as np
import torch
from scripts.marble.precision_stepwise import checked_receipt
from scripts.marble.run_alpha_grid import ALPHAS, RATS, SEEDS, bundle_receipt, find_job
from scripts.marble.score_multirat_projection import decoding_rows
from utils.utils_marble import read_json, save_json, sha256
from utils.utils_marble_consistency import consistency_scores

logger = logging.getLogger(__name__)


def summarize_grid(rows):
    """Never select a seed; give each animal equal weight and retain both modes."""
    cells = []
    for mode in ("eval", "notebook"):
        for alpha in ALPHAS:
            for animal in RATS:
                matches = sorted(
                    [
                        r
                        for r in rows
                        if r["mode"] == mode
                        and r["alpha"] == alpha
                        and r["animal"] == animal
                    ],
                    key=lambda r: r["seed"],
                )
                assert [r["seed"] for r in matches] == list(SEEDS)
                values = [r["mae_cm"] for r in matches]
                assert np.isfinite(values).all()
                cells.append(
                    {
                        "mode": mode,
                        "alpha": alpha,
                        "animal": animal,
                        "seed_mae_cm": values,
                        "mean_cm": float(np.mean(values)),
                        "sd_cm": float(np.std(values, ddof=1)),
                        "mean_r2": float(np.mean([r["position_r2"] for r in matches])),
                    }
                )
    assert len(rows) == len(cells) * len(SEEDS), "Unexpected extra or duplicate rows"

    def cell(animal, alpha, mode="eval"):
        matches = [
            c
            for c in cells
            if c["animal"] == animal and c["alpha"] == alpha and c["mode"] == mode
        ]
        assert len(matches) == 1
        return matches[0]

    common = []
    for alpha in ALPHAS:
        animals = [cell(animal, alpha) for animal in RATS]
        seed_macro = np.mean([c["seed_mae_cm"] for c in animals], axis=0)
        deltas = np.array(
            [c["mean_cm"] - cell(c["animal"], 0.0)["mean_cm"] for c in animals]
        )
        paired_deltas = np.array([c["seed_mae_cm"] for c in animals]) - np.array(
            [cell(c["animal"], 0.0)["seed_mae_cm"] for c in animals]
        )
        common.append(
            {
                "alpha": alpha,
                "macro_mean_cm": float(seed_macro.mean()),
                "seed_macro_cm": seed_macro.tolist(),
                "seed_macro_sd_cm": float(seed_macro.std(ddof=1)),
                "animals_improved": int((deltas < 0).sum()),
                "animal_deltas_cm": dict(zip(RATS, deltas.tolist(), strict=True)),
                "paired_seeds_improved": int((paired_deltas < 0).sum()),
                "paired_seed_deltas_cm": dict(
                    zip(RATS, paired_deltas.tolist(), strict=True)
                ),
            }
        )
    best = min(common, key=lambda c: (c["macro_mean_cm"], c["alpha"]))
    per_animal, folds = [], []
    for animal in RATS:
        winner = min(
            [cell(animal, a) for a in ALPHAS], key=lambda c: (c["mean_cm"], c["alpha"])
        )
        per_animal.append(winner)
        other = [r for r in RATS if r != animal]
        scores = {
            a: float(np.mean([cell(r, a)["mean_cm"] for r in other])) for a in ALPHAS
        }
        chosen = min(ALPHAS, key=lambda a: (scores[a], a))
        folds.append(
            {
                "excluded_animal": animal,
                "selected_alpha": chosen,
                "selection_animals": other,
                "selection_mean_cm": scores[chosen],
                "excluded_mean_cm": cell(animal, chosen)["mean_cm"],
                "pca_mean_cm": cell(animal, 0.0)["mean_cm"],
            }
        )
    return {
        "cells": cells,
        "common_alpha": common,
        "best_common": best,
        "best_per_animal": per_animal,
        "loao": {
            "folds": folds,
            "macro_mean_cm": float(np.mean([f["excluded_mean_cm"] for f in folds])),
            "pca_macro_mean_cm": float(np.mean([f["pca_mean_cm"] for f in folds])),
            "scope": "Retrospective selection sensitivity; previously studied animals",
        },
    }


def load_embeddings(work, alpha, seed):
    embeddings, labels, hashes = [], [], {}
    for animal in RATS:
        job = find_job(work, "consistency", animal, alpha)
        run = Path(job["folder"]) / "run"
        assert (run / "original_code_audit.json").is_file()
        path = run / f"seed-{seed}/embeddings.pt"
        pack = torch.load(path, map_location="cpu", weights_only=True)
        assert "embedding" in pack and "labels" in pack
        assert pack["embedding"].shape == ((6576 if animal == "buddy" else 9999), 3)
        assert len(pack["labels"]) == len(pack["embedding"])
        embeddings.append(pack["embedding"].numpy())
        labels.append(pack["labels"][:, 0].numpy())
        hashes[animal] = sha256(path)
    return embeddings, labels, hashes


def check_parameters(candidate, control):
    left = read_json(candidate / "run/protocol.json")
    right = read_json(control / "run/protocol.json")
    assert "model_parameters" in left and "model_parameters" in right
    assert left["model_parameters"] == right["model_parameters"], (
        "Grid comparison changed model hyperparameters"
    )


def aggregate(root):
    frozen = read_json(root / "frozen_protocol.json")
    completed = read_json(root / "completed.json")
    work = frozen["jobs"]
    assert len(work) == completed["jobs"] == 48
    rows, residuals, numerics = [], [], []
    for job in work:
        assert bundle_receipt(job) == completed["receipts"][job["folder"]]
        folder = Path(job["folder"])
        control = find_job(work, job["protocol"], job["animal"], 0.0)
        check_parameters(folder, Path(control["folder"]))
        if job["protocol"] == "consistency":
            path = (
                root / "numerics" / job["animal"] / job["condition"] / "validation.json"
            )
            validation = checked_receipt(folder / "input", path)
            numerics.append(
                {
                    "animal": job["animal"],
                    "alpha": job["alpha"],
                    "validation": validation,
                    "oracle_receipt": read_json(path.parent / "oracle_receipt.json"),
                }
            )
        residuals.append(
            {
                **job,
                "projection": read_json(folder / "input/projection_diagnostics.json"),
            }
        )
        if job["protocol"] == "decoding":
            entries = decoding_rows(folder / "run", job["animal"], job["condition"])
            rows.extend([{**entry, "alpha": job["alpha"]} for entry in entries])
    consistency, summary = [], []
    for alpha in ALPHAS:
        entries = []
        for seed in SEEDS:
            embeddings, labels, hashes = load_embeddings(work, alpha, seed)
            value, _ = consistency_scores(embeddings, labels, RATS)
            entry = {"alpha": alpha, "seed": seed, "hashes": hashes, **value}
            entries.append(entry)
            consistency.append(entry)
        values = [v["mean_r2"] for v in entries]
        summary.append(
            {
                "alpha": alpha,
                "mean_r2": float(np.mean(values)),
                "sd_r2": float(np.std(values, ddof=1)),
                "seed_mean_r2": values,
                "pairs": entries[0]["pairs"],
                "pair_seed_mean_r2": np.mean(
                    [v["scores"] for v in entries], axis=0
                ).tolist(),
            }
        )
    result = {
        "decoding_rows": rows,
        "decoding": summarize_grid(rows),
        "consistency": consistency,
        "consistency_summary": summary,
        "best_consistency": min(summary, key=lambda v: (-v["mean_r2"], v["alpha"])),
        "projection_residuals": residuals,
        "numerics": numerics,
        "caveat": "Exploratory grid on previously examined data; seeds/pairs are not biological replicates",
        "score_source_sha256": sha256(Path(__file__)),
    }
    save_json(root / "aggregate.json", result)
    logger.info("Best common decoding alpha: %s", result["decoding"]["best_common"])
    logger.info("Best consistency alpha: %s", result["best_consistency"])
    return result


def audit_cebra(root):
    import cebra
    from cebra.integrations.sklearn.helpers import align_embeddings

    assert importlib.metadata.version("cebra") == "0.4.0"
    assert importlib.metadata.version("scikit-learn") == "1.3.2"
    result = read_json(root / "aggregate.json")
    work = read_json(root / "frozen_protocol.json")["jobs"]
    errors = []
    for entry in result["consistency"]:
        embeddings, labels, hashes = load_embeddings(
            work, entry["alpha"], entry["seed"]
        )
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
        _, actual = consistency_scores(embeddings, labels, RATS)
        for observed, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(observed, reference, atol=1e-6, rtol=1e-6)
        errors.append(
            {
                "alpha": entry["alpha"],
                "seed": entry["seed"],
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
    logger.info("All %d consistency metric audits passed", len(errors))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-cebra", action="store_true")
    args = parser.parse_args()
    if args.audit_cebra:
        audit_cebra(args.root.resolve())
    else:
        aggregate(args.root.resolve())
