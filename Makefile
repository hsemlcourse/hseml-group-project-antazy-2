.PHONY: lint test
lint:
	python3 -m flake8 notebooks/01_download_data.py app src scripts tests --max-line-length=120 --exclude=__pycache__

test:
	PYTHONPATH=src:. python3 -m pytest tests/ -v

defaults:
	PYTHONPATH=src python3 scripts/build_defaults.py

importance:
	PYTHONPATH=src python3 scripts/plot_feature_importance.py
