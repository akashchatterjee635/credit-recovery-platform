.PHONY: setup test train-smoke run

setup:
	python -m pip install --upgrade pip
	pip install -e ".[static,dev]"

test:
	pytest

train-smoke:
	python -m scripts.train_static --config configs/static.yaml --smoke

run:
	uvicorn backend.main:app --host 0.0.0.0 --port 8000
