"""Command-line parsing and input file validation for the callme project.

This module is responsible for:
- Parsing CLI arguments (--functions_definition, --input, --output, ...).
- Loading and validating the function definitions and prompt files.
- Performing extra semantic checks (identifiers, duplicates, emptiness).
- Ensuring the output path is writable.
- Returning a validated tuple consumed by the rest of the program.

Any validation failure raises ``ValueError`` with a helpful message.
File/OS errors bubble up to :func:`parse`, which catches them and
delegates to :func:`src.ui.print_error` (which prints and exits).
"""

from __future__ import annotations

import json
import keyword
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import ValidationError

from src.models import FunctionRegistry, Prompt
from src.ui import print_error

DEFAULT_FUNCTIONS_DEFINITION = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def _get_args() -> Dict[str, Any]:
    """Parse command-line arguments and return them as a dictionary.

    Returns:
        A dict with keys: functions_definition, input, output, model, debug.
    """
    parser = ArgumentParser(
        prog="python -m src",
        description=(
            "Translate natural-language prompts into structured function "
            "calls using constrained decoding."
        ),
    )
    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help=(
            "Path to the JSON file describing the available functions "
            f"(default: {DEFAULT_FUNCTIONS_DEFINITION})."
        ),
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=(
            "Path to the JSON file containing the prompts to process "
            f"(default: {DEFAULT_INPUT})."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Path where the results JSON file will be written "
            f"(default: {DEFAULT_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Name of the LLM model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output during generation.",
    )
    args: Namespace = parser.parse_args()
    return vars(args)


def _load_json(path: str) -> Any:
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON content.

    Raises:
        ValueError: If the file contains invalid JSON, with the file path,
            line, and column of the error.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid json file: '{path}:{error.lineno}:{error.colno}'"
            f" - {error.msg}"
        ) from error


def _load_function_registry(path: str) -> FunctionRegistry:
    """Load and validate the function definitions file.

    Args:
        path: Path to the functions_definition JSON file.

    Returns:
        A validated :class:`FunctionRegistry`.

    Raises:
        ValueError: If the file is invalid JSON or fails pydantic validation.
    """
    json_data = _load_json(path)
    try:
        function_registry: FunctionRegistry = FunctionRegistry.model_validate(
            {"functions": json_data}
        )
    except ValidationError as error:
        first_error = error.errors(include_url=False)[0]
        message = first_error.get("msg", "malformed input")
        location = first_error.get("loc", ())
        if location:
            raise ValueError(
                f"Invalid config file: '{path}' - '{location[-1]}' - "
                f"{message}"
            ) from error
        raise ValueError(
            f"Invalid config file: '{path}' - {message}"
        ) from error
    return function_registry


def _load_prompts(path: str) -> List[Prompt]:
    """Load and validate the prompts file.

    Args:
        path: Path to the function_calling_tests JSON file.

    Returns:
        A list of validated :class:`Prompt` objects.

    Raises:
        ValueError: If the file is invalid or fails pydantic validation.
    """
    json_file = _load_json(path)
    try:
        prompts: List[Prompt] = [Prompt(**item) for item in json_file]
    except ValidationError as error:
        first_error = error.errors(include_url=False)[0]
        message = first_error.get("msg", "malformed input")
        location = first_error.get("loc", ())
        if location:
            raise ValueError(
                f"Invalid input file: '{path}' - '{location[-1]}' - "
                f"{message}"
            ) from error
        raise ValueError(
            f"Invalid input file: '{path}' - {message}"
        ) from error
    return prompts


def _validate_registry(
    function_registry: FunctionRegistry, path: str
) -> None:
    """Perform semantic checks on the function registry.

    Args:
        function_registry: The registry to validate.
        path: The source file path, used in error messages.

    Raises:
        ValueError: If any semantic check fails.
    """
    if not function_registry.functions:
        raise ValueError(
            f"Invalid config file: '{path}' - No functions provided"
        )
    for function in function_registry.functions:
        if not function.name:
            raise ValueError(
                f"Invalid config file: '{path}' - Empty function name"
            )
        if not function.name.isidentifier():
            raise ValueError(
                f"Invalid config file: '{path}' - Function name "
                f"'{function.name}' is not a valid identifier"
            )
        if keyword.iskeyword(function.name):
            raise ValueError(
                f"Invalid config file: '{path}' - Function name "
                f"'{function.name}' is a Python keyword"
            )
    names = function_registry.names
    if len(names) != len(set(names)):
        raise ValueError(
            f"Invalid config file: '{path}' - Duplicated function names"
        )


def _validate_prompts(prompts: List[Prompt], path: str) -> None:
    """Perform semantic checks on the prompts list.

    Args:
        prompts: The list of prompts to validate.
        path: The source file path, used in error messages.

    Raises:
        ValueError: If any semantic check fails.
    """
    if not prompts:
        raise ValueError(
            f"Invalid input file: '{path}' - No prompts provided"
        )
    for prompt in prompts:
        if not prompt.prompt:
            raise ValueError(
                f"Invalid input file: '{path}' - Empty prompt"
            )


def parse() -> Tuple[FunctionRegistry, List[Prompt], Path, str, bool]:
    """Parse arguments, load and validate all input files.

    Returns:
        A tuple ``(function_registry, prompts, output_path, model, debug)``.

    Raises:
        SystemExit: Via :func:`src.ui.print_error` when any error occurs.
    """
    args = _get_args()
    try:
        function_registry = _load_function_registry(
            args["functions_definition"]
        )
        _validate_registry(
            function_registry, args["functions_definition"]
        )

        prompts = _load_prompts(args["input"])
        _validate_prompts(prompts, args["input"])

        output_path = Path(args["output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write("")

        model = str(args["model"])
        debug = bool(args["debug"])
        return function_registry, prompts, output_path, model, debug
    except FileNotFoundError as error:
        print_error(f"Missing input file: '{error.filename}'")
    except NotADirectoryError as error:
        print_error(f"Not a directory: '{error.filename}'")
    except IsADirectoryError as error:
        print_error(f"Is a directory: '{error.filename}'")
    except PermissionError as error:
        print_error(f"Permission denied: '{error.filename}'")
    except FileExistsError as error:
        print_error(f"File already exists: '{error.filename}'")
    except OSError as error:
        print_error(f"OS error: '{error}'")
    except ValueError as error:
        print_error(str(error))
    # Unreachable in practice: print_error always exits.
    raise RuntimeError("Unreachable code reached in parse()")
