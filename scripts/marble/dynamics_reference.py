"""Export a projection intervention through the unchanged MARBLE CPU pipeline."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scripts.marble.prepare import SOURCE_COMMIT
from utils.utils_dynamics_projection import DynamicsProjection
from utils.utils_marble import read_json, save_json, sha256


def prepare(repository, output, alpha):
    repository, output = Path(repository).resolve(), Path(output).resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    assert revision == SOURCE_COMMIT
    changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository,
        text=True,
    ).strip()
    assert not changes, "Original MARBLE tracked sources must be clean"
    if output.exists():
        raise FileExistsError(f"Choose a new export directory: {output}")
    sys.path[:0] = [str(repository), str(repository / "reproduction/src")]
    import prepare_gpu_training
    import rat_utils

    assert Path(prepare_gpu_training.__file__).resolve().is_relative_to(repository)
    observed = []

    class RecordedProjection(DynamicsProjection):
        def transform(self, values):
            observed.append(np.array(values, copy=True))
            return super().transform(values)

    projection = RecordedProjection(n_components=20, alpha=alpha)

    def constructor(n_components):
        assert n_components == projection.n_components
        return projection

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with patch.object(rat_utils, "PCA", constructor):
        prepare_gpu_training.prepare(output)
    assert len(observed) == 2
    assert observed[0].shape == (8000, 120)
    assert observed[1].shape == (2000, 120)
    diagnostics = {
        "alpha": alpha,
        "effective_lambda": projection.effective_lambda_,
        "dimensions": projection.n_components,
        "fit_scope": "8000 training bins only; no behavior labels",
        "basis_convention": "orthogonal Procrustes to exact PCA; no whitening",
        "train": projection.errors(observed[0]),
        "test": projection.errors(observed[1]),
        "pca_train": projection.errors(observed[0], reference=True),
        "pca_test": projection.errors(observed[1], reference=True),
    }
    graphs = torch.load(output / "original_graphs.pt", weights_only=False)
    from utils.utils_marble import tensor_digest

    diagnostics["graphs"] = {
        split: {
            "edges": graphs[split].edge_index.shape[1],
            "split_masks_sha256": tensor_digest(
                (name, getattr(graphs[split], name))
                for name in ("train_mask", "val_mask", "test_mask")
            ),
        }
        for split in ("train", "test")
    }
    for values in observed:
        reduced = (values - projection.mean_) @ projection.basis_
        np.testing.assert_allclose(
            np.diff(reduced, axis=0),
            np.diff(values, axis=0) @ projection.basis_,
            rtol=1e-9,
            atol=1e-10,
        )
    assert diagnostics["train"]["increment_squared_error"] <= diagnostics["pca_train"][
        "increment_squared_error"
    ] * (1 + 1e-10)
    assert diagnostics["train"]["state_squared_error"] >= diagnostics["pca_train"][
        "state_squared_error"
    ] * (1 - 1e-10)
    np.savez_compressed(
        output / "projection.npz",
        basis=projection.basis_,
        mean=projection.mean_,
        pca_basis=projection.reference_basis_,
        eigenvalues=projection.eigenvalues_,
    )
    save_json(output / "projection_diagnostics.json", diagnostics)
    protocol = read_json(output / "protocol.json")
    protocol["projection_intervention"] = diagnostics
    save_json(output / "protocol.json", protocol)
    provenance = read_json(output / "provenance.json")
    provenance["projection_intervention"] = {
        "alpha": alpha,
        "unchanged_upstream_files": True,
        "in_memory_patch": "rat_utils.PCA constructor only",
        "projection_sha256": sha256(output / "projection.npz"),
        "sources": {
            str(path): sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[2]
                / "utils/utils_dynamics_projection.py",
            )
        },
    }
    save_json(output / "provenance.json", provenance)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marble-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    arguments = parser.parse_args()
    prepare(arguments.marble_repo, arguments.output, arguments.alpha)
