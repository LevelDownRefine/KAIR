"""Export four-rat author models and independent references in pinned CPU Python.

Invoked through prepare.py after checking the original repository revision.
Only hash-verified public artifacts are deserialized. No representation training.
"""

import argparse
import copy
import functools
import importlib.metadata
import logging
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from scripts.marble.prepare import SOURCE_COMMIT
from utils.utils_marble import read_json, save_json, sha256, tensor_digest

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
RATS = ("achilles", "buddy", "cicero", "gatsby")


@contextmanager
def capture_dropout():
    """Record Bernoulli masks without assuming CUDA shares the CPU RNG stream."""
    native = F.dropout
    masks = []

    def dropout(value, p=0.5, training=True, inplace=False):
        if not training or p == 0:
            return native(value, p, training, inplace)
        assert p == 0.5 and not inplace
        mask = native(torch.ones_like(value), p, True, False) != 0
        masks.append(mask.detach().cpu())
        return value * mask / (1 - p)

    with patch("torch.nn.functional.dropout", dropout):
        yield masks


def reference_steps(model, graph):
    from MARBLE import dataloader, utils
    from modern_gpu import cpu_state

    torch.manual_seed(999)
    model.train()
    initial = cpu_state(model)
    loader, _, _ = dataloader.loaders(graph, model.params)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
    features = []
    hook = model.enc.register_forward_pre_hook(
        lambda module, args: features.append(args[0].detach().clone())
    )
    batches = []
    try:
        for targets, ids, adjs in loader:
            with capture_dropout() as masks:
                embedding, mask = model(graph, ids, utils.to_list(adjs))
            assert len(masks) == 1
            loss = model.loss(embedding, mask)
            optimizer.zero_grad()
            loss.backward()
            gradients = {
                name: value.grad.detach().cpu().clone()
                for name, value in model.named_parameters()
                if value.grad is not None
            }
            optimizer.step()
            batches.append(
                {
                    "targets": targets,
                    "ids": ids,
                    "features": features[-1],
                    "dropout_mask": masks[0],
                    "embedding": embedding.detach(),
                    "loss": loss.detach(),
                    "gradients": gradients,
                    "state_after": cpu_state(model),
                }
            )
            if len(batches) == 3:
                break
    finally:
        hook.remove()
    assert len(batches) == 3
    return {"initial_state": initial, "batches": batches}


def export(repository, data_root, output):
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(repository / "examples/rat_hippocampus"))
    import cebra
    import MARBLE
    from cebra.integrations.sklearn.helpers import align_embeddings
    from modern_gpu import cpu_state
    from prepare_gpu_training import export_graph
    from rat_decoding import load_cebra_cpu
    from rat_utils import convert_spikes_to_rates
    from repro_io import load_author_data, verify_file

    assert Path(MARBLE.__file__).resolve().parent == repository / "MARBLE"
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    assert revision == SOURCE_COMMIT
    assert importlib.metadata.version("scikit-learn") == "1.3.2"
    assert importlib.metadata.version("cebra") == "0.4.0"
    assert not torch.cuda.is_available(), "Reference export uses pinned CPU Python"
    torch.set_num_threads(4)
    np.random.seed(0)
    torch.manual_seed(0)
    inventory_path = (
        ROOT / "docs/reports/marble-20260928/full_reproduction_inventory.json"
    )
    inventory = read_json(inventory_path)
    entries = {Path(item["path"]).name: item for item in inventory["files"]}
    names = ["rat_data.pkl"] + [
        f"{method}_{rat}_3D.{suffix}"
        for rat in RATS
        for method, suffix in (
            ("marble", "pth"),
            ("cebra_time", "pt"),
            ("cebra_behaviour", "pt"),
        )
    ]
    for name in names:
        assert name in entries
        verify_file(data_root / name, entries[name])
    dataset = load_author_data(data_root / "rat_data.pkl")
    assert tuple(dataset) == RATS
    output.mkdir(parents=True, exist_ok=False)
    pack = {"rats": list(RATS), "animals": {}, "reference_consistency": {}}
    groups, labels = {}, {}
    for rat in RATS:
        logger.info("Preparing %s: public CEBRA weights and PCA=10 graph", rat)
        neural = dataset[rat]["neural"].numpy()
        y = dataset[rat]["continuous_index"].numpy()
        baselines = {}
        for method in ("time", "behaviour"):
            model = load_cebra_cpu(data_root / f"cebra_{method}_{rat}_3D.pt")
            baselines[f"CEBRA-{method}"] = torch.from_numpy(model.transform(neural))
        constructor = functools.partial(
            MARBLE.construct_dataset, number_of_eigenvectors=1
        )
        with patch("MARBLE.construct_dataset", constructor):
            graph, y_marble, pca = convert_spikes_to_rates(neural.T, y, pca_n=10)
        model = MARBLE.net(
            graph, loadpath=str(data_root / f"marble_{rat}_3D.pth"), verbose=False
        )
        params = copy.deepcopy(model.params)
        assert params["dropout"] == 0.5 and params["out_channels"] == 3
        assert params["diffusion"] is False and params["order"] == 1
        state = cpu_state(model)
        animal = {
            "params": params,
            "state": state,
            "graph": export_graph(graph),
            "labels": torch.from_numpy(y_marble.copy()),
            "baseline_labels": torch.from_numpy(y.copy()),
            "baselines": baselines,
            "pca": {
                "components": torch.from_numpy(pca.components_),
                "mean": torch.from_numpy(pca.mean_),
                "solver": pca._fit_svd_solver,
            },
            "modes": {},
        }
        captured = []
        hook = model.enc.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone())
        )
        try:
            for mode in (
                "eval",
                "batch_stats",
                "notebook_seed0",
                "notebook_seed1",
                "notebook_seed2",
            ):
                model.load_state_dict(state, strict=True)
                model.train(mode != "eval")
                model.enc.dropout = [0.0 if mode == "batch_stats" else 0.5, 0.0]
                torch.manual_seed(int(mode[-1]) if mode.startswith("notebook") else 0)
                with capture_dropout() as masks:
                    embedding = model.transform(graph.clone()).emb
                result = {"embedding": embedding, "state_after": cpu_state(model)}
                if mode.startswith("notebook"):
                    assert len(masks) == 1
                    result["dropout_mask"] = masks[0]
                else:
                    assert not masks
                animal["modes"][mode] = result
                if mode == "eval":
                    animal["features"] = captured[-1]
        finally:
            hook.remove()
        model.load_state_dict(state, strict=True)
        model.enc.dropout = [0.5, 0.0]
        animal["validation"] = reference_steps(model, graph)
        pack["animals"][rat] = animal
        for name, values in baselines.items():
            if name not in groups:
                groups[name], labels[name] = [], []
            groups[name].append(values.numpy())
            labels[name].append(y[:, 0])
        for mode, result in animal["modes"].items():
            key = f"MARBLE-{mode}"
            if key not in groups:
                groups[key], labels[key] = [], []
            groups[key].append(result["embedding"].numpy())
            labels[key].append(y_marble[:, 0])
        logger.info(
            "Exported %s, %d anchors, PCA %s", rat, len(graph.x), pca._fit_svd_solver
        )
    for name, embeddings in groups.items():
        scores, pairs, subjects = cebra.sklearn.metrics.consistency_score(
            embeddings=embeddings,
            labels=labels[name],
            dataset_ids=list(RATS),
            between="datasets",
        )
        pack["reference_consistency"][name] = {
            "scores": torch.from_numpy(scores),
            "pairs": pairs.tolist(),
            "subjects": subjects.tolist(),
            "aligned": [
                torch.from_numpy(np.asarray(a))
                for a in align_embeddings(embeddings, labels[name])
            ],
        }
        logger.info("Original CEBRA metric %s: mean R2 %.6f", name, scores.mean())
    torch.save(pack, output / "rat_consistency.pt")
    # Also catch accidental NumPy scalars before the modern weights-only reader.
    torch.load(output / "rat_consistency.pt", weights_only=True, map_location="cpu")
    sources = [
        Path(__file__),
        ROOT / "scripts/marble/prepare.py",
        repository / "examples/rat_hippocampus/rat_utils.py",
    ]
    save_json(
        output / "receipt.json",
        {
            "bundle_sha256": sha256(output / "rat_consistency.pt"),
            "marble_commit": revision,
            "reference_lock_sha256": sha256(repository / "reproduction/uv.lock"),
            "inventory_sha256": sha256(inventory_path),
            "source_sha256": {
                str(
                    p.relative_to(ROOT if p.is_relative_to(ROOT) else repository)
                ): sha256(p)
                for p in sources
            },
            "marble_source_sha256": {
                str(path.relative_to(repository)): sha256(path)
                for path in sorted((repository / "MARBLE").glob("*.py"))
            },
            "author_files": {name: entries[name]["sha256"] for name in names},
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "cebra",
                    "torch-geometric",
                )
            },
            "state_sha256": {
                rat: tensor_digest(pack["animals"][rat]["state"].items())
                for rat in RATS
            },
            "protocol": {
                "representation_training_performed": False,
                "profile": "released-checkpoint / consistency notebook PCA=10",
                "preprocessing_seed": 0,
                "preprocessing_rng": "one sequential NumPy stream in animal order",
                "graph_delta": 1.5,
                "spike_bin_units_and_counts": (
                    "original millisecond-index and nonzero-event behavior"
                ),
                "sampling": "whole recordings; no chronological holdout",
                "metric": (
                    "CEBRA 0.4.0 position-only alignment, 100 edges, "
                    "99 normalized bin means, directed in-sample OLS R2"
                ),
                "notebook_masks": (
                    "CPU Bernoulli masks shared for CPU/CUDA numerical comparison; "
                    "seeds 0/1/2"
                ),
                "batch_stats": (
                    "diagnostic BN train mode with dropout disabled; "
                    "not a paper protocol"
                ),
            },
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marble-repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.marble_repo.resolve(), args.data.resolve(), args.output.resolve())
