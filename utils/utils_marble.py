"""MARBLE experiment receipts, position decoding and result plots."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(items):
    """Use the reference export's fingerprint format, including dtype and shape."""
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def predict_position(
    train_embedding, test_embedding, train_labels, neighbors=36, return_neighbors=False
):
    assert train_embedding.ndim == test_embedding.ndim == 2
    assert train_embedding.shape[1] == test_embedding.shape[1]
    assert train_labels.ndim == 2 and len(train_labels) == len(train_embedding)
    assert 1 <= neighbors <= len(train_embedding)
    # CEBRA's continuous-target decoder delegates to this sklearn estimator.
    decoder = KNeighborsRegressor(n_neighbors=neighbors, metric="cosine")
    decoder.fit(train_embedding, train_labels[:, 0])
    prediction = decoder.predict(test_embedding)
    if return_neighbors:
        return prediction, decoder.kneighbors(test_embedding, return_distance=False)
    return prediction


def audit_decoder_ties(actual, indices, expected, reference_indices, distances, labels):
    """Accept a different kNN boundary choice only at exactly equal distances.

    NumPy versions can choose different entries in argpartition ties. Do not widen
    the prediction tolerance: prove that every changed neighbor is on the boundary.
    """
    assert indices.shape == reference_indices.shape
    assert indices.ndim == 2 and len(indices) == len(actual) == len(expected)
    assert indices.dtype.kind in "iu" and indices.min() >= 0
    assert indices.max() < len(labels)
    assert all(len(np.unique(row)) == len(row) for row in indices)
    np.testing.assert_allclose(
        actual, labels[indices].mean(axis=1), rtol=1e-6, atol=1e-6
    )
    changed = np.flatnonzero(~np.isclose(actual, expected, rtol=2e-4, atol=2e-5))
    for index in changed:
        exchanged = np.setxor1d(indices[index], reference_indices[index])
        assert len(exchanged) > 0, "Prediction differs without a neighbor change"
        boundary = distances[index, reference_indices[index]].max()
        assert np.all(distances[index, exchanged] == boundary), (
            f"Decoder mismatch at sample {index} is not an exact boundary tie"
        )
    return {
        "max_abs_error_m": float(np.max(np.abs(actual - expected))),
        "exact_boundary_tie_samples": changed.tolist(),
    }


def position_metrics(truth, prediction):
    assert truth.ndim == 1 and truth.shape == prediction.shape and len(truth) > 1
    assert np.isfinite(truth).all() and np.isfinite(prediction).all()
    error = np.abs(truth - prediction)
    return {
        "samples": len(truth),
        "mean_absolute_error_m": float(error.mean()),
        "median_absolute_error_m": float(np.median(error)),
        "std_absolute_error_m": float(error.std(ddof=1)),
        "position_r2": float(r2_score(truth, prediction)),
    }


def summarize(runs):
    assert runs
    result = {}
    for mode in ("eval", "notebook"):
        values = []
        for run in runs:
            assert "results" in run and mode in run["results"]
            assert "mean_absolute_error_m" in run["results"][mode]
            values.append(run["results"][mode]["mean_absolute_error_m"])
        result[mode] = {
            "mean_mae_m": float(np.mean(values)),
            "sample_std_across_seeds_m": (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            ),
            "min_mae_m": min(values),
            "max_mae_m": max(values),
        }
    return {"summary": result, "runs": runs}


def plot_run(directory, history, truth, prediction):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    assert "train_loss" in history and "val_loss" in history
    for key in ("train_loss", "val_loss"):
        axes[0].plot(np.arange(1, len(history[key]) + 1), history[key], label=key)
    axes[0].set(xlabel="Epoch", ylabel="Contrastive loss")
    axes[0].legend()
    times = np.arange(len(truth)) / 40
    axes[1].plot(times, truth, label="Measured", color="#444444", lw=1)
    axes[1].plot(times, prediction, label="Decoded (eval)", color="#0072B2", lw=1)
    axes[1].set(xlabel="Held-out time (s)", ylabel="Position (m)")
    axes[1].legend()
    fig.savefig(Path(directory) / "training.png", dpi=160)
    plt.close(fig)
