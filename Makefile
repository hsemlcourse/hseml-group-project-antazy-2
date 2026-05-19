.PHONY: lint
lint:
	python3 -m flake8 notebooks/01_download_data.py --max-line-length=120
