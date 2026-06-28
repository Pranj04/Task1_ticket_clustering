"""Text cleaning and dataset preparation utilities."""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPECIAL_CHARS_RE = re.compile(r"[^a-z0-9\s]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    """Normalize one ticket description for embedding and keyword extraction."""
    text = "" if pd.isna(value) else str(value)
    text = html.unescape(text)
    text = text.lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = SPECIAL_CHARS_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def preprocess_descriptions(descriptions: pd.Series) -> pd.Series:
    """Clean a pandas Series containing ticket descriptions."""
    return descriptions.fillna("").map(clean_text)


def combine_text_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Combine multiple text columns into one description field."""
    selected_columns = [column for column in columns if column in frame.columns]
    if not selected_columns:
        raise ValueError("None of the configured text columns exist in the dataset.")

    LOGGER.info("Using text columns: %s", ", ".join(selected_columns))
    return frame[selected_columns].fillna("").astype(str).agg(" ".join, axis=1)


def infer_text_columns(frame: pd.DataFrame) -> list[str]:
    """Infer likely ticket text columns from common helpdesk dataset schemas."""
    preferred_groups = [
        ["ticket_description"],
        ["description"],
        ["body"],
        ["subject", "body"],
        ["title", "body"],
        ["summary", "description"],
        ["issue_title", "issue_body"],
        ["text"],
    ]
    lower_to_original = {column.lower(): column for column in frame.columns}

    for group in preferred_groups:
        if all(column in lower_to_original for column in group):
            return [lower_to_original[column] for column in group]

    object_columns = frame.select_dtypes(include=["object", "string"]).columns.tolist()
    if not object_columns:
        raise ValueError("No text-like columns were found in the input dataset.")

    LOGGER.warning(
        "Could not infer a canonical ticket description column; using %s.",
        object_columns[0],
    )
    return [object_columns[0]]


def prepare_ticket_frame(
    frame: pd.DataFrame,
    text_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of the dataset with raw and cleaned ticket descriptions."""
    prepared = frame.copy()
    columns = text_columns or infer_text_columns(prepared)
    prepared["Ticket Description"] = combine_text_columns(prepared, columns)
    prepared["clean_description"] = preprocess_descriptions(
        prepared["Ticket Description"]
    )
    prepared = prepared[prepared["clean_description"].str.len() > 0].reset_index(
        drop=True
    )
    if prepared.empty:
        raise ValueError("No non-empty ticket descriptions remain after cleaning.")
    return prepared

