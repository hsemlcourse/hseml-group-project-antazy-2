"""Default feature values for partial inputs (Telegram wizard)."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from petfinder.constants import DEFAULT_DEFAULTS_PATH, FEATURE_COLUMNS, TARGET_COLUMN


def build_defaults_from_train(train_path: Path, output_path: Path) -> Dict[str, Any]:
    """Compute median/mode defaults from training CSV."""
    df = pd.read_csv(train_path)
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    defaults: Dict[str, Any] = {}
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(series.median()) if series.notna().any() else 0
            if col not in ("Fee", "PhotoAmt", "VideoAmt", "Age", "State"):
                defaults[col] = int(defaults[col])
        else:
            mode = series.mode(dropna=True)
            defaults[col] = mode.iloc[0] if len(mode) else ""
    return defaults


def save_defaults(defaults: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(defaults, f, ensure_ascii=False, indent=2)


def load_defaults(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or DEFAULT_DEFAULTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Defaults file not found: {path}. Run: python scripts/build_defaults.py"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def merge_with_defaults(
    partial: Dict[str, Any],
    defaults_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fill missing feature columns from saved defaults."""
    defaults = load_defaults(defaults_path)
    merged = dict(defaults)
    merged.update({k: v for k, v in partial.items() if v is not None and v != ""})
    return merged
