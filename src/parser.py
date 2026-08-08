"""CLI parsing and input validation."""



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
    """Parse command-line arguments into a dict."""
    parser = ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into "
        "structured function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help=f"Functions file (default: {DEFAULT_FUNCTIONS_DEFINITION}).",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Prompts file (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output file (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output.",
    )
    args: Namespace = parser.parse_args()
    return vars(args)


def _load_json(path: str) -> Any:
    """Load a JSON file, raising ValueError with location on bad JSON."""
    with open(path, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid json file: '{path}:{error.lineno}:{error.colno}'"
                f" - {error.msg}"
            ) from error


def _validation_message(path: str, error: ValidationError,
                        prefix: str) -> str:
    """Build a short validation error message."""
    first_error = error.errors(include_url=False)[0]
    message = first_error.get("msg", "malformed input")
    location = first_error.get("loc", ())
    field = location[-1] if location else None
    if field:
        return f"Invalid {prefix} file: '{path}' - '{field}' - {message}"
    return f"Invalid {prefix} file: '{path}' - {message}"


def _load_function_registry(path: str) -> FunctionRegistry:
    """Load and validate the function definitions file."""
    json_data = _load_json(path)
    try:
        return FunctionRegistry.model_validate({"functions": json_data})
    except ValidationError as error:
        raise ValueError(
            _validation_message(path, error, "config")
        ) from error


def _load_prompts(path: str) -> List[Prompt]:
    """Load and validate the prompts file."""
    json_file = _load_json(path)
    try:
        return [Prompt(**item) for item in json_file]
    except ValidationError as error:
        raise ValueError(
            _validation_message(path, error, "input")
        ) from error


def _validate_registry(function_registry: FunctionRegistry,
                       path: str) -> None:
    """Reject empty, invalid, keyword, or duplicate function names."""
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
    """Reject an empty prompt list or empty prompt strings."""
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
    """Parse args, validate inputs, ensure writable output, return config."""
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
        open(output_path, "w", encoding="utf-8").close()

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
    raise RuntimeError("Unreachable code reached in parse()")
