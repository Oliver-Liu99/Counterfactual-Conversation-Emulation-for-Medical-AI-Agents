.PHONY: install install-locked test lint synthetic clean

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

# Install the project in editable mode using version *ranges* from requirements.txt.
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

# Install pinned exact versions from the lockfile for byte-reproducible runs.
install-locked:
	$(PIP) install -r requirements-lock.txt
	$(PIP) install -e . --no-deps

# Run the full test suite. Real-data tests skip automatically when
# data/raw/aci_bench/ is absent.
test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -q

# Style check (non-blocking; just reports).
lint:
	-ruff check src tests

# Step-8 synthetic OPE oracle benchmark (no data / API needed).
synthetic:
	PYTHONPATH=src $(PYTHON) scripts/08_synthetic_benchmark.py

# Remove caches and build artefacts.
clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
