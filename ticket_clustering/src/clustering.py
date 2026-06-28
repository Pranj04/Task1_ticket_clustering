"""Clustering and cluster interpretation helpers."""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

LOGGER = logging.getLogger(__name__)


def run_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster ticket embeddings with KMeans."""
    if n_clusters < 2:
        raise ValueError("KMeans requires at least 2 clusters.")
    if len(embeddings) <= n_clusters:
        raise ValueError("Number of clusters must be smaller than number of tickets.")

    LOGGER.info("Running KMeans with %d clusters.", n_clusters)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return model.fit_predict(embeddings)


def generate_cluster_labels(
    descriptions: pd.Series,
    labels: np.ndarray,
    top_n: int = 3,
) -> dict[int, str]:
    """Create human-readable cluster labels from frequent TF-IDF keywords."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
    )
    matrix = vectorizer.fit_transform(descriptions)
    terms = np.array(vectorizer.get_feature_names_out())
    cluster_labels: dict[int, str] = {}

    for cluster_id in sorted(np.unique(labels)):
        rows = matrix[labels == cluster_id]
        mean_scores = np.asarray(rows.mean(axis=0)).ravel()
        best_indices = mean_scores.argsort()[::-1][:top_n]
        keywords = [term for term in terms[best_indices] if term]
        cluster_labels[int(cluster_id)] = " / ".join(keywords) or f"cluster {cluster_id}"

    return cluster_labels


def representative_tickets(
    frame: pd.DataFrame,
    labels: np.ndarray,
    embeddings: np.ndarray,
    description_column: str = "Ticket Description",
    examples_per_cluster: int = 3,
) -> dict[int, list[str]]:
    """Select representative tickets nearest to each cluster centroid."""
    representatives: dict[int, list[str]] = {}

    for cluster_id in sorted(np.unique(labels)):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_vectors = embeddings[cluster_indices]
        centroid = cluster_vectors.mean(axis=0)
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        nearest_local_indices = distances.argsort()[:examples_per_cluster]
        nearest_indices = cluster_indices[nearest_local_indices]
        representatives[int(cluster_id)] = (
            frame.iloc[nearest_indices][description_column].astype(str).tolist()
        )

    return representatives


def frequent_keyword_label(texts: list[str], top_n: int = 3) -> str:
    """Fallback frequency-based label for very small clusters."""
    tokens: list[str] = []
    for text in texts:
        tokens.extend(
            token
            for token in text.split()
            if len(token) > 2 and token not in ENGLISH_STOP_WORDS
        )
    keywords = [word for word, _ in Counter(tokens).most_common(top_n)]
    return " / ".join(keywords)

