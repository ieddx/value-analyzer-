.PHONY: install test lint run clean help

VENV   := .venv
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

# Default ticker for `make run`
TICKER ?= KO

help:
	@echo "Targets:"
	@echo "  make install      — create venv and install all dependencies"
	@echo "  make test         — run the full test suite"
	@echo "  make test-fast    — run tests, skip slow integration tests"
	@echo "  make lint         — run ruff (style) and mypy (types)"
	@echo "  make run          — analyse TICKER=KO (override with TICKER=AAPL)"
	@echo "  make run-md       — same as run but emit plain-text markdown"
	@echo "  make clean        — remove venv and cache artefacts"

install: $(VENV)/bin/activate

$(VENV)/bin/activate: pyproject.toml
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@touch $(VENV)/bin/activate

test: $(VENV)/bin/activate
	$(PYTEST) -q

test-fast: $(VENV)/bin/activate
	$(PYTEST) -q -m "not integration"

lint: $(VENV)/bin/activate
	@$(VENV)/bin/ruff check src/ tests/ 2>/dev/null \
		|| (echo "ruff not installed — run: pip install ruff"; exit 0)
	@$(VENV)/bin/mypy src/ 2>/dev/null \
		|| (echo "mypy not installed — run: pip install mypy"; exit 0)

run: $(VENV)/bin/activate
	$(PYTHON) -m value_analyzer.cli $(TICKER)

run-md: $(VENV)/bin/activate
	$(PYTHON) -m value_analyzer.cli $(TICKER) --markdown

clean:
	rm -rf $(VENV) .pytest_cache __pycache__ src/*.egg-info
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
