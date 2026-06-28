"""Sentence Transformer embedding generation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def generate_embeddings(
    descriptions: pd.Series,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """Generate dense semantic embeddings for cleaned ticket descriptions."""
    if descriptions.empty:
        raise ValueError("Cannot generate embeddings for an empty description series.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    LOGGER.info("Loading Sentence Transformer model: %s", model_name)
    model = SentenceTransformer(model_name)

    LOGGER.info("Generating embeddings for %d tickets.", len(descriptions))
    embeddings = model.encode(
        descriptions.tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)
