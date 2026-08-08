# Makefile for the callme project.
# Automates common tasks: install, run, debug, clean, lint.

.PHONY: install run debug clean lint lint-strict

# uv cache/tool locations (writable goinfre paths, independent of shell env).
UV_CACHE_DIR := $(HOME)/goinfre/uv/cache
UV_TOOL_DIR := $(HOME)/goinfre/uv/tools
HF_HOME := $(HOME)/goinfre/hf_cache
HUGGINGFACE_HUB_CACHE := $(HOME)/goinfre/hf_cache/hub

# Install project dependencies using uv.
install:
	UV_CACHE_DIR=$(UV_CACHE_DIR) UV_TOOL_DIR=$(UV_TOOL_DIR) \
		HF_HOME=$(HF_HOME) HUGGINGFACE_HUB_CACHE=$(HUGGINGFACE_HUB_CACHE) \
		uv sync

# Execute the main script of the project.
run:
	UV_CACHE_DIR=$(UV_CACHE_DIR) UV_TOOL_DIR=$(UV_TOOL_DIR) \
		HF_HOME=$(HF_HOME) HUGGINGFACE_HUB_CACHE=$(HUGGINGFACE_HUB_CACHE) \
		uv run python -m src

# Run the main script in debug mode using Python's built-in debugger.
debug:
	UV_CACHE_DIR=$(UV_CACHE_DIR) UV_TOOL_DIR=$(UV_TOOL_DIR) \
		HF_HOME=$(HF_HOME) HUGGINGFACE_HUB_CACHE=$(HUGGINGFACE_HUB_CACHE) \
		uv run python -m pdb -m src

# Remove temporary files or caches to keep the project environment clean.
clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache data/output
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Run the linters (flake8 and mypy) with the mandatory flags.
lint:
	UV_CACHE_DIR=$(UV_CACHE_DIR) UV_TOOL_DIR=$(UV_TOOL_DIR) \
		uv run flake8 .
	UV_CACHE_DIR=$(UV_CACHE_DIR) UV_TOOL_DIR=$(UV_TOOL_DIR) \
		uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

# Run the linters with strict mypy checking (optional).
lint-strict:
	UV_CACHE_DIR=$(UV_CACHE_DIR) UV_TOOL_DIR=$(UV_TOOL_DIR) \
		uv run flake8 .
	UV_CACHE_DIR=$(UV_CACHE_DIR) UV_TOOL_DIR=$(UV_TOOL_DIR) \
		uv run mypy . --strict
