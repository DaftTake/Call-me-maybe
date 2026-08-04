"""Entry point for the callme program.

Usage:
    uv run python -m src [--functions_definition <file>]
        [--input <file>] [--output <file>] [--model <name>] [--debug]
"""

from __future__ import annotations

from src.parser import parse


def main() -> None:
    """Parse arguments, load inputs, and run the generation pipeline."""
    function_registry, prompts, output_path, model, debug = parse()

    print(f"Loaded {len(function_registry.functions)} function(s).")
    print(f"Loaded {len(prompts)} prompt(s).")
    print(f"Output will be written to: {output_path}")
    print(f"Model: {model}")
    print(f"Debug: {debug}")

    # TODO: implement the constrained-decoding generation pipeline here.


if __name__ == "__main__":
    main()
