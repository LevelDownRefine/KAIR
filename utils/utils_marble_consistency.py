"""Position-binned, directed in-sample consistency for the four-rat protocol.

Alignment semantics adapted from CEBRA 0.4.0 sklearn/helpers.py and metrics.py.
Copyright Mackenzie W. Mathis & Steffen Schneider (v0.4.0+), Apache-2.0.
See licenses/CEBRA.txt. Restricted to unique animal IDs and finite observations;
missing bins expand by at most two neighbors, exactly as in the reference.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def align_positions(embeddings, labels, n_bins=100):
    assert len(embeddings) == len(labels) >= 2
    assert isinstance(n_bins, int) and n_bins >= 3
    for embedding, position in zip(embeddings, labels, strict=True):
        assert embedding.ndim == 2 and position.ndim == 1
        assert len(embedding) == len(position) > 1
        assert np.isfinite(embedding).all() and np.isfinite(position).all()
    low, high = min(y.min() for y in labels), max(y.max() for y in labels)
    assert low < high
    edges = np.linspace(low, high, n_bins)
    aligned = []
    for embedding, position in zip(embeddings, labels, strict=True):
        bins = np.digitize(position, edges)
        means = []
        for index in range(1, n_bins):
            distances = np.abs(bins - index)
            nearest = int(distances.min())
            if nearest > 2:
                raise ValueError(f"No observations within two bins of bin {index}")
            value = embedding[distances <= nearest].mean(axis=0)
            norm = np.linalg.norm(value)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError(f"Cannot normalize mean embedding in bin {index}")
            means.append(value / norm)
        aligned.append(np.asarray(means))
    return aligned


def consistency_scores(embeddings, labels, animals, n_bins=100):
    assert len(animals) == len(embeddings) == len(set(animals))
    aligned = align_positions(embeddings, labels, n_bins)
    scores, pairs = [], []
    for source, source_name in enumerate(animals):
        for target, target_name in enumerate(animals):
            if source == target:
                continue
            regression = LinearRegression().fit(aligned[source], aligned[target])
            scores.append(float(regression.score(aligned[source], aligned[target])))
            pairs.append([source_name, target_name])
    assert np.isfinite(scores).all()
    return {
        "scores": scores,
        "pairs": pairs,
        "mean_r2": float(np.mean(scores)),
    }, aligned
