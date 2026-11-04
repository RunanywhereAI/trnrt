# Developer shortcuts. Every target runs what CI runs (.github/workflows/ci.yml).
PYTHON ?= python3.13
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help dev lint test check docs image clean

help:  ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

dev:  ## create .venv: CPU torch, editable install with test and lint tools, the pre-commit hook
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
	$(BIN)/pip install -q -e '.[dev]' 'transformers==5.17.0' 'tokenizers==0.23.3'
	$(BIN)/pre-commit install

lint:  ## pre-commit on every file (ruff, file checks, docs links)
	$(BIN)/pre-commit run --all-files

test:  ## the CPU test suite
	$(BIN)/pytest -q

check: lint test  ## lint, then the tests

docs:  ## check that every relative Markdown link resolves
	$(BIN)/python tools/check_docs.py

image:  ## build the serving and compile image (docker/Dockerfile)
	docker build -f docker/Dockerfile \
	  --build-arg TRNRT_VERSION=$$($(PYTHON) -c 'import trnrt; print(trnrt.__version__)') \
	  --build-arg TRNRT_REVISION=$$(git rev-parse HEAD) -t trnrt .

clean:  ## remove build output and caches
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
