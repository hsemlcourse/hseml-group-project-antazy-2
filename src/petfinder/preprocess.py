"""Data preprocessing for inference."""

from typing import Any, Dict, List, Optional

import pandas as pd

from petfinder.constants import FEATURE_COLUMNS, TARGET_COLUMN


def clean_dataframe(df: pd.DataFrame, drop_duplicates: bool = False) -> pd.DataFrame:
    """Add NameLength; optionally drop duplicate rows (training EDA only)."""
    clean_df = df.copy()
    if drop_duplicates:
        clean_df = clean_df.drop_duplicates()
    if "Name" in clean_df.columns:
        clean_df["NameLength"] = clean_df["Name"].fillna("").astype(str).str.len()
    return clean_df


def validate_features(record: Dict[str, Any], required: Optional[List[str]] = None) -> None:
    """Raise ValueError if required feature keys are missing."""
    required = required or FEATURE_COLUMNS
    missing = [col for col in required if col not in record]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    if TARGET_COLUMN in record:
        raise ValueError(f"Target column '{TARGET_COLUMN}' must not be in inference input.")


def record_to_dataframe(record: Dict[str, Any]) -> pd.DataFrame:
    """Convert a single feature dict to a one-row DataFrame."""
    validate_features(record)
    row = {col: record.get(col) for col in FEATURE_COLUMNS if col in record}
    for col in FEATURE_COLUMNS:
        if col not in row:
            row[col] = None
    return pd.DataFrame([row])


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert multiple records to a DataFrame."""
    if not records:
        raise ValueError("At least one record is required.")
    for record in records:
        validate_features(record)
    rows = []
    for record in records:
        row = {col: record.get(col) for col in FEATURE_COLUMNS}
        rows.append(row)
    return pd.DataFrame(rows)
