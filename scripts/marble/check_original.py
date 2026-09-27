"""Run with the pinned MARBLE CPU interpreter to audit KAIR-trained checkpoints.

original_graphs.pt contains PyG objects created by the original exporter; only
use a trusted locally generated export. Model weights use weights_only=True.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances

from utils.utils_marble import (
    audit_decoder_ties,
    read_json,
    save_json,
    sha256,
    tensor_digest,
)

logger = logging.getLogger(__name__)


def check(repository, source, run):
    repository, source, run = map(Path.resolve, (repository, source, run))
    sys.path.insert(0, str(repository))
    import MARBLE
    from cebra import KNNDecoder

    assert Path(MARBLE.__file__).resolve().parent == repository / "MARBLE"
    torch.set_num_threads(4)
    provenance = read_json(run / "provenance.json")
    assert "training_input_sha256" in provenance
    assert sha256(source / "training_input.pt") == provenance["training_input_sha256"]
    pack = torch.load(
        source / "training_input.pt", map_location="cpu", weights_only=True
    )
    original = torch.load(
        source / "original_graphs.pt", map_location="cpu", weights_only=False
    )
    summary = read_json(run / "summary.json")
    options = read_json(run / "options.json")
    assert "seeds" in summary and "decoder" in options
    assert "neighbors" in options["decoder"]
    assert all(key in pack for key in ("train_labels", "test_labels"))
    assert "train" in original and "test" in original
    results = {}
    for seed in summary["seeds"]:
        directory = run / f"seed-{seed}"
        model = MARBLE.net(
            original["train"], loadpath=str(directory / "best_model.pth"), verbose=False
        ).eval()
        best = torch.load(
            directory / "best_model.pth", map_location="cpu", weights_only=True
        )
        assert "model_state_dict" in best
        assert tensor_digest(model.state_dict().items()) == tensor_digest(
            best["model_state_dict"].items()
        )
        embeddings = torch.load(
            directory / "embeddings.pt", map_location="cpu", weights_only=True
        )
        expected = model.transform(original["test"].clone()).emb.numpy()
        assert "eval" in embeddings and "test" in embeddings["eval"]
        np.testing.assert_allclose(
            embeddings["eval"]["test"].numpy(), expected, rtol=2e-4, atol=2e-5
        )
        errors = {
            "original_embedding_max_abs_error": float(
                np.max(np.abs(embeddings["eval"]["test"].numpy() - expected))
            )
        }
        with np.load(directory / "arrays.npz") as arrays:
            assert "truth_position_m" in arrays
            np.testing.assert_array_equal(
                arrays["truth_position_m"], pack["test_labels"][:, 0].numpy()
            )
            for mode in ("eval", "notebook"):
                assert mode in embeddings
                assert "train" in embeddings[mode] and "test" in embeddings[mode]
                decoder = KNNDecoder(
                    n_neighbors=options["decoder"]["neighbors"], metric="cosine"
                )
                decoder.fit(
                    embeddings[mode]["train"].numpy(),
                    pack["train_labels"][:, 0].numpy(),
                )
                prediction = decoder.predict(embeddings[mode]["test"].numpy())
                key = f"prediction_{mode}"
                assert key in arrays
                neighbor_key = f"neighbors_{mode}"
                assert neighbor_key in arrays
                reference_indices = decoder.knn.kneighbors(
                    embeddings[mode]["test"].numpy(), return_distance=False
                )
                distances = cosine_distances(
                    embeddings[mode]["test"].numpy(), embeddings[mode]["train"].numpy()
                )
                errors[f"{mode}_decoder"] = audit_decoder_ties(
                    arrays[key],
                    arrays[neighbor_key],
                    prediction,
                    reference_indices,
                    distances,
                    pack["train_labels"][:, 0].numpy(),
                )
        history = read_json(directory / "loss_history.json")
        metrics = read_json(directory / "metrics.json")
        assert "val_loss" in history and "best_epoch_zero_based" in metrics
        assert metrics["best_epoch_zero_based"] == int(np.argmin(history["val_loss"]))
        results[str(seed)] = errors
        logger.info("Seed %d original-code audit passed: %s", seed, errors)
    save_json(
        run / "original_code_audit.json",
        {
            "rtol": 2e-4,
            "atol": 2e-5,
            "seeds": results,
            "decoder_reference": "CEBRA KNNDecoder in original pinned CPU environment",
            "decoder_rule": "Strict prediction tolerance or proven exact kNN boundary ties",
            "audit_source_sha256": sha256(Path(__file__)),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marble-repo", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    check(arguments.marble_repo, arguments.input, arguments.run)
