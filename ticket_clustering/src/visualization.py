"""Cluster visualization utilities."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)


def timestamped_path(path: Path) -> Path:
    """Return a timestamped path to avoid overwriting existing outputs."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def plot_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    random_state: int = 42,
) -> Path:
    """Project embeddings with UMAP and save a cluster scatter plot."""
    if len(embeddings) < 3:
        raise ValueError("At least three tickets are required for UMAP visualization.")

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    neighbors = min(15, max(2, len(embeddings) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    LOGGER.info("Reducing embeddings to 2D with UMAP.")
    coordinates = reducer.fit_transform(embeddings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        cmap="tab10",
        s=36,
        alpha=0.85,
        edgecolors="none",
    )
    plt.title("Ticket Clusters (UMAP Projection)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=180)
    except PermissionError as exc:
        fallback_path = timestamped_path(output_path)
        LOGGER.warning(
            "Unable to write to %s (%s). Writing to %s instead.",
            output_path,
            exc,
            fallback_path,
        )
        plt.savefig(fallback_path, dpi=180)
        plt.close()
        LOGGER.info("Saved cluster plot to %s", fallback_path)
        return fallback_path
    plt.close()
    LOGGER.info("Saved cluster plot to %s", output_path)
    return output_path
