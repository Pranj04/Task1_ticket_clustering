"""Cluster quality metrics."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score

LOGGER = logging.getLogger(__name__)


def evaluate_clusters(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute clustering quality metrics."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("At least two clusters are required for evaluation.")
    if len(unique_labels) >= len(labels):
        raise ValueError("Each sample cannot be its own cluster for evaluation.")

    metrics = {
        "silhouette_score": float(silhouette_score(embeddings, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(embeddings, labels)),
    }
    LOGGER.info("Cluster metrics: %s", metrics)
    return metrics

