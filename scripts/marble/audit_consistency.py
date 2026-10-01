"""Read new 3D weights through original MARBLE in its pinned CPU environment."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from utils.utils_marble import read_json, save_json, sha256, tensor_digest
from utils.utils_marble_storage import load_tensor_file


def audit(repository, source, run):
    sys.path.insert(0, str(repository))
    import MARBLE

    assert Path(MARBLE.__file__).resolve().parent == repository / "MARBLE"
    torch.set_num_threads(4)
    receipt = read_json(source / "provenance.json")
    assert "original_graphs_sha256" in receipt
    assert sha256(source / "original_graphs.pt.gz") == receipt["original_graphs_sha256"]
    trained = read_json(run / "provenance.json")
    assert "training_input_sha256" in trained
    assert sha256(source / "training_input.pt") == trained["training_input_sha256"]
    graphs = load_tensor_file(source / "original_graphs.pt.gz", weights_only=False)
    assert "train" in graphs
    summary = read_json(run / "summary.json")
    assert "seeds" in summary
    errors, embeddings = {}, {}
    for seed in summary["seeds"]:
        destination = run / f"seed-{seed}"
        model = MARBLE.net(
            graphs["train"], loadpath=str(destination / "best_model.pth"), verbose=False
        ).eval()
        expected = model.transform(graphs["train"].clone()).emb
        observed = torch.load(
            destination / "embeddings.pt", map_location="cpu", weights_only=True
        )
        assert "embedding" in observed and "labels" in observed
        torch.testing.assert_close(
            observed["embedding"], expected, rtol=2e-4, atol=2e-5
        )
        best = torch.load(
            destination / "best_model.pth", map_location="cpu", weights_only=True
        )
        assert "model_state_dict" in best
        assert tensor_digest(model.state_dict().items()) == tensor_digest(
            best["model_state_dict"].items()
        )
        history = read_json(destination / "loss_history.json")
        metrics = read_json(destination / "metrics.json")
        assert "val_loss" in history and "best_epoch_zero_based" in metrics
        assert metrics["best_epoch_zero_based"] == int(np.argmin(history["val_loss"]))
        errors[str(seed)] = float((observed["embedding"] - expected).abs().max())
        embeddings[str(seed)] = {"embedding": expected, "labels": observed["labels"]}
    torch.save(embeddings, run / "original_code_embeddings.pt")
    save_json(
        run / "original_code_audit.json",
        {
            "rtol": 2e-4,
            "atol": 2e-5,
            "embedding_max_errors": errors,
            "audit_source_sha256": sha256(Path(__file__)),
        },
    )
    logging.info("Original-code checkpoint audit passed: %s", errors)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marble-repo", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    audit(args.marble_repo.resolve(), args.input.resolve(), args.run.resolve())
