# One-command entry points. On Windows without make, run the python commands directly.
PY ?= python

.PHONY: all install models test lint format sim clean

all: models test sim

install:
	$(PY) -m pip install -r requirements-lock.txt
	$(PY) -m pip install -e . --no-deps

models:
	$(PY) -m dpt.fetch_models

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m black --check src tests scripts
	$(PY) -m mypy

format:
	$(PY) -m ruff check --fix src tests scripts
	$(PY) -m black src tests scripts

sim:
	$(PY) -m dpt.run_all

clean:
	rm -rf build
