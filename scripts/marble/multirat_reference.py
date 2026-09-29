"""Prepare fresh multi-rat experiments with original CPU graphs and samplers."""

import argparse
import copy
import functools
import logging
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scripts.marble.prepare import SOURCE_COMMIT
from utils.utils_dynamics_projection import DynamicsProjection
from utils.utils_marble import read_json, save_json, sha256, tensor_digest
from utils.utils_marble_storage import save_compressed

ROOT = Path(__file__).resolve().parents[2]
RATS = ("achilles", "buddy", "cicero", "gatsby")
logger = logging.getLogger(__name__)


def projection_diagnostics(projection, observed):
    assert len(observed) in (1, 2)
    result = {
        "alpha": projection.alpha,
        "dimensions": projection.n_components,
        "effective_lambda": projection.effective_lambda_,
        "mean_fit_rows": len(observed[0]),
        "raw_dimensions": observed[0].shape[1],
        "basis_convention": "Procrustes to exact PCA; no whitening",
    }
    for index, values in enumerate(observed):
        split = "train" if index == 0 else "test"
        result[split] = projection.errors(values)
        result[f"pca_{split}"] = projection.errors(values, reference=True)
    assert "train" in result and "pca_train" in result
    for key in ("increment_squared_error", "state_squared_error"):
        assert key in result["train"] and key in result["pca_train"]
    assert result["train"]["increment_squared_error"] <= (
        result["pca_train"]["increment_squared_error"] * (1 + 1e-10)
    )
    assert result["train"]["state_squared_error"] >= (
        result["pca_train"]["state_squared_error"] * (1 - 1e-10)
    )
    return result


def prepare(repository, data_root, output, animal, protocol, alpha):
    assert animal in RATS and protocol in ("decoding", "consistency")
    assert np.isfinite(alpha) and alpha >= 0
    if output.exists():
        raise FileExistsError(output)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    assert revision == SOURCE_COMMIT
    changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository,
        text=True,
    ).strip()
    assert not changes
    sys.path[:0] = [
        str(repository),
        str(repository / "reproduction/src"),
        str(repository / "examples/rat_hippocampus"),
    ]
    import MARBLE
    import rat_utils
    from MARBLE import dataloader
    from modern_gpu import cpu_state
    from prepare_gpu_training import export_batches, export_graph, fixture
    from rat_decoding import provenance, split_recording
    from repro_io import load_author_data, verify_file
    from scripts.marble.rat_reference import reference_steps
    from train_rat import fresh_model, graph_digest

    assert Path(MARBLE.__file__).resolve().parent == repository / "MARBLE"
    assert not torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    inventory = read_json(
        ROOT / "docs/reports/marble-20260928/full_reproduction_inventory.json"
    )
    assert "files" in inventory
    entries = {}
    for item in inventory["files"]:
        assert "path" in item
        entries[Path(item["path"]).name] = item
    checkpoint = (
        "marble_achilles_32D.pth"
        if protocol == "decoding"
        else f"marble_{animal}_3D.pth"
    )
    for name in ("rat_data.pkl", checkpoint):
        assert name in entries
        verify_file(data_root / name, entries[name])
    author = torch.load(data_root / checkpoint, map_location="cpu", weights_only=False)
    assert "params" in author and "model_state_dict" in author
    params = copy.deepcopy(author["params"])
    assert "epochs" in params and params["epochs"] == 100
    data = load_author_data(data_root / "rat_data.pkl")
    assert animal in data
    rat = data[animal]
    assert "neural" in rat and "continuous_index" in rat
    neural, labels = rat["neural"].numpy(), rat["continuous_index"].numpy()
    expected_rows = 6577 if animal == "buddy" else 10000
    assert len(neural) == len(labels) == expected_rows
    dimensions = 20 if protocol == "decoding" else 10
    observed = []

    class RecordedProjection(DynamicsProjection):
        def transform(self, values):
            observed.append(np.array(values, copy=True))
            return super().transform(values)

    projection = RecordedProjection(dimensions, alpha)

    def constructor(n_components):
        assert n_components == dimensions
        return projection

    np.random.seed(0)
    torch.manual_seed(0)
    graph_constructor = functools.partial(
        MARBLE.construct_dataset, number_of_eigenvectors=1
    )
    with (
        patch.object(rat_utils, "PCA", constructor),
        patch("MARBLE.construct_dataset", graph_constructor),
    ):
        if protocol == "decoding":
            train, test, y_train, y_test = split_recording(neural, labels)
            train_graph, train_labels, _ = rat_utils.convert_spikes_to_rates(
                train.T, y_train, pca_n=dimensions
            )
            test_graph, test_labels, _ = rat_utils.convert_spikes_to_rates(
                test.T, y_test, pca=projection, pca_n=dimensions
            )
            assert len(observed) == 2
        else:
            train_graph, train_labels, _ = rat_utils.convert_spikes_to_rates(
                neural.T, labels, pca_n=dimensions
            )
            test_graph, test_labels = train_graph, train_labels
            assert len(observed) == 1
    output.mkdir(parents=True)
    fingerprints = {
        "train_graph": graph_digest(train_graph),
        "test_graph": graph_digest(test_graph),
    }
    details = projection_diagnostics(projection, observed)
    details["graphs"] = {
        split: {
            "edges": graph.edge_index.shape[1],
            "split_masks_sha256": tensor_digest(
                (key, getattr(graph, key))
                for key in ("train_mask", "val_mask", "test_mask")
            ),
        }
        for split, graph in (("train", train_graph), ("test", test_graph))
    }
    np.savez_compressed(
        output / "projection.npz",
        mean=projection.mean_,
        basis=projection.basis_,
        pca_basis=projection.reference_basis_,
    )
    save_json(output / "projection_diagnostics.json", details)
    metadata = {
        "animal": animal,
        "protocol": protocol,
        "alpha": alpha,
        "seeds": [0, 1, 2],
        "model_parameters": params,
        "pca_dimensions": dimensions,
        "raw_shape": list(neural.shape),
        "graph_nodes": {"train": len(train_labels), "test": len(test_labels)},
        "input_fingerprints": fingerprints,
        "projection_intervention": details,
        "split": "chronological 80% train / 20% test; projection fit on train only"
        if protocol == "decoding"
        else "whole public recording; in-sample consistency, no behavioral holdout",
        "selection": "minimum contrastive validation loss inside training graph",
        "source_checkpoint_hyperparameters_only": checkpoint,
        "author_weights_loaded_for_training": False,
        "preprocessing": "unchanged author smoothing and bin handling",
    }
    save_json(output / "protocol.json", metadata)
    validation = (
        fixture(train_graph, params)
        if protocol == "decoding"
        else reference_steps(fresh_model(train_graph, params, 999), train_graph)
    )
    train_export = export_graph(train_graph)
    test_export = export_graph(test_graph) if protocol == "decoding" else train_export
    pack = {
        "params": params,
        "seeds": [0, 1, 2],
        "train_graph": train_export,
        "test_graph": test_export,
        "train_labels": torch.from_numpy(train_labels),
        "test_labels": torch.from_numpy(test_labels),
        "validation": validation,
    }
    torch.save(pack, output / "training_input.pt")
    save_compressed(
        {"train": train_graph, "test": test_graph}, output / "original_graphs.pt.gz"
    )
    receipt = provenance({name: entries[name] for name in ("rat_data.pkl", checkpoint)})
    receipt["projection_sources"] = {
        path: sha256(ROOT / path)
        for path in (
            "scripts/marble/multirat_reference.py",
            "scripts/marble/rat_reference.py",
            "utils/utils_dynamics_projection.py",
            "utils/utils_marble_storage.py",
        )
    }
    receipt["animal"] = animal
    receipt["protocol"] = protocol
    receipt["original_graphs_sha256"] = sha256(output / "original_graphs.pt.gz")
    save_json(output / "provenance.json", receipt)
    author_hash = tensor_digest(author["model_state_dict"].items())
    for seed in (0, 1, 2):
        started = time.perf_counter()
        destination = output / f"seed-{seed}"
        destination.mkdir()
        model = fresh_model(train_graph, params, seed)
        initial = cpu_state(model)
        digest = tensor_digest(initial.items())
        assert digest != author_hash
        save_json(
            destination / "initialization.json",
            {
                "seed": seed,
                "initial_state_sha256": digest,
                "author_state_sha256": author_hash,
                "optimizer_state_loaded": False,
                "input_fingerprints": fingerprints,
            },
        )
        loaders = dataloader.loaders(train_graph, model.params)
        epochs = []
        for epoch in range(params["epochs"]):
            epochs.append(
                {"train": export_batches(loaders[0]), "val": export_batches(loaders[1])}
            )
            if (epoch + 1) % 25 == 0:
                logger.info(
                    "%s %s alpha=%s seed=%s: sampled %s/100",
                    protocol,
                    animal,
                    alpha,
                    seed,
                    epoch + 1,
                )
        plan = {
            "initial_state": initial,
            "epochs": epochs,
            "test": export_batches(loaders[2]),
        }
        filename = "sampling.pt.gz"
        save_compressed(plan, destination / filename)
        save_json(
            destination / "sampling_receipt.json",
            {
                "file": filename,
                "sha256": sha256(destination / filename),
                "seconds": time.perf_counter() - started,
            },
        )
    save_json(
        output / "prepared.json",
        {"complete": True, "protocol": protocol, "animal": animal, "alpha": alpha},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marble-repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--animal", choices=RATS, required=True)
    parser.add_argument(
        "--protocol", choices=("decoding", "consistency"), required=True
    )
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    prepare(
        args.marble_repo.resolve(),
        args.data.resolve(),
        args.output.resolve(),
        args.animal,
        args.protocol,
        args.alpha,
    )
