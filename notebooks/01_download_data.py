#!/usr/bin/env python3
"""Скачать данные соревнования PetFinder с Kaggle и распаковать в data/raw/.

Ожидается `kaggle.json` в корне репозитория (см. README) или стандартный
`~/.kaggle/kaggle.json`. Запуск из любой директории:

    python notebooks/01_download_data.py
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

COMPETITION = "petfinder-adoption-prediction"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def configure_kaggle_credentials(root: Path) -> None:
    """Если в корне проекта есть kaggle.json — использовать его для API."""
    if (root / "kaggle.json").is_file():
        os.environ["KAGGLE_CONFIG_DIR"] = str(root)


def unzip_all_zips(raw_dir: Path) -> None:
    for zpath in sorted(raw_dir.glob("*.zip")):
        print(f"Распаковка: {zpath.name} -> {raw_dir}/")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(raw_dir)


def download(raw_dir: Path, *, force: bool) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    marker = raw_dir / "train" / "train.csv"
    if marker.is_file() and not force:
        print(f"Уже есть {marker.relative_to(raw_dir.parent.parent)} — пропуск загрузки.")
        print("Повторить принудительно: python notebooks/01_download_data.py --force")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    print(f"Загрузка файлов соревнования {COMPETITION} в {raw_dir} …")
    api.competition_download_files(
        COMPETITION,
        path=str(raw_dir),
        force=force,
        quiet=False,
    )
    unzip_all_zips(raw_dir)
    print("Готово: сырые данные и zip-архивы лежат в data/raw/.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Скачать PetFinder Adoption Prediction с Kaggle.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Скачать заново и перезаписать, даже если train/train.csv уже есть.",
    )
    args = parser.parse_args(argv)

    root = project_root()
    configure_kaggle_credentials(root)
    raw_dir = root / "data" / "raw"

    try:
        download(raw_dir, force=args.force)
    except OSError as e:
        print(f"Ошибка ввода-вывода: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        # KaggleApiError и прочие — показать причину без огромного traceback
        print(f"Ошибка Kaggle: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
