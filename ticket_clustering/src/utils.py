"""General I/O, EDA, configuration, and logging utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: Path) -> dict[str, Any]:
    """Load JSON configuration from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV ticket dataset with pandas."""
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    return pd.read_csv(path)


def run_eda(frame: pd.DataFrame, description_column: str) -> dict[str, Any]:
    """Return basic EDA results for the ticket dataset."""
    descriptions = frame[description_column].fillna("").astype(str)
    length_stats = descriptions.str.len().describe().to_dict()
    return {
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "missing_values": {
            column: int(value) for column, value in frame.isna().sum().items()
        },
        "duplicate_records": int(frame.duplicated().sum()),
        "description_length_distribution": {
            key: float(value) for key, value in length_stats.items()
        },
    }


def timestamped_path(path: Path) -> Path:
    """Return a timestamped path to avoid overwriting existing outputs."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def save_json(payload: dict[str, Any], path: Path) -> Path:
    """Persist a dictionary as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        return path
    except PermissionError as exc:
        fallback_path = timestamped_path(path)
        LOGGER.warning(
            "Unable to write to %s (%s). Writing to %s instead.",
            path,
            exc,
            fallback_path,
        )
        with fallback_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        return fallback_path


def save_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    """Persist a dataframe as CSV, falling back to a timestamped path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(path, index=False)
        return path
    except PermissionError as exc:
        fallback_path = timestamped_path(path)
        LOGGER.warning(
            "Unable to write to %s (%s). Writing to %s instead.",
            path,
            exc,
            fallback_path,
        )
        frame.to_csv(fallback_path, index=False)
        return fallback_path
    except OSError as exc:
        fallback_path = timestamped_path(path)
        LOGGER.warning(
            "Unable to write to %s (%s). Writing to %s instead.",
            path,
            exc,
            fallback_path,
        )
        frame.to_csv(fallback_path, index=False)
        return fallback_path


def ensure_output_dir(path: Path) -> Path:
    """Create the output directory if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path

