#!/usr/bin/env python3
"""Build feature_defaults.json from training data."""

from petfinder.constants import DEFAULT_DEFAULTS_PATH, PROJECT_ROOT
from petfinder.defaults import build_defaults_from_train, save_defaults

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train" / "train.csv"


def main() -> None:
    if not TRAIN_PATH.exists():
        raise SystemExit(f"Train file not found: {TRAIN_PATH}")
    defaults = build_defaults_from_train(TRAIN_PATH, DEFAULT_DEFAULTS_PATH)
    save_defaults(defaults, DEFAULT_DEFAULTS_PATH)
    print(f"Saved defaults to {DEFAULT_DEFAULTS_PATH}")


if __name__ == "__main__":
    main()
