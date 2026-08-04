# Makefile for the callme project.
# Automates common tasks: install, run, debug, clean, lint.

.PHONY: install run debug clean lint lint-strict

# Install project dependencies using uv.
install:
	uv sync

# Execute the main script of the project.
run:
	uv run python -m src

# Run the main script in debug mode using Python's built-in debugger.
debug:
	uv run python -m pdb -m src

# Remove temporary files or caches to keep the project environment clean.
clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Run the linters (flake8 and mypy) with the mandatory flags.
lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

# Run the linters with strict mypy checking (optional).
lint-strict:
	uv run flake8 .
	uv run mypy . --strict
