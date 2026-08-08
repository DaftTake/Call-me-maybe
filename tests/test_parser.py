"""Unit tests for the parser module."""



import json
from pathlib import Path

import pytest

from src.models import FunctionRegistry, Prompt
from src.parser import (
    _load_function_registry,
    _load_json,
    _load_prompts,
    _validate_prompts,
    _validate_registry,
)

VALID_FUNCTIONS = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers.",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "returns": {"type": "number"},
    },
    {
        "name": "fn_greet",
        "description": "Greet a person.",
        "parameters": {"name": {"type": "string"}},
        "returns": {"type": "string"},
    },
]

VALID_PROMPTS = [
    {"prompt": "What is the sum of 2 and 3?"},
    {"prompt": "Greet shrek"},
]


def _write_json(path: Path, data: object) -> Path:
    """Write a JSON file and return its path."""
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_json_valid(tmp_path: Path) -> None:
    """A valid JSON file loads correctly."""
    path = _write_json(tmp_path / "data.json", {"key": "value"})
    assert _load_json(str(path)) == {"key": "value"}


def test_load_json_invalid(tmp_path: Path) -> None:
    """Invalid JSON raises ValueError with line and column."""
    path = tmp_path / "bad.json"
    path.write_text("{invalid", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        _load_json(str(path))
    assert "Invalid json file" in str(excinfo.value)
    assert "bad.json" in str(excinfo.value)


def test_load_function_registry_valid(tmp_path: Path) -> None:
    """A valid function registry loads."""
    path = _write_json(tmp_path / "functions.json", VALID_FUNCTIONS)
    registry = _load_function_registry(str(path))
    assert isinstance(registry, FunctionRegistry)
    assert registry.names == ["fn_add_numbers", "fn_greet"]


def test_load_function_registry_invalid(tmp_path: Path) -> None:
    """An invalid function registry raises ValueError."""
    path = _write_json(tmp_path / "functions.json", [{"name": "fn_x"}])
    with pytest.raises(ValueError) as excinfo:
        _load_function_registry(str(path))
    assert "Invalid config file" in str(excinfo.value)


def test_load_prompts_valid(tmp_path: Path) -> None:
    """A valid prompts file loads."""
    path = _write_json(tmp_path / "prompts.json", VALID_PROMPTS)
    prompts = _load_prompts(str(path))
    assert len(prompts) == 2
    assert all(isinstance(p, Prompt) for p in prompts)


def test_load_prompts_invalid(tmp_path: Path) -> None:
    """An invalid prompts file raises ValueError."""
    path = _write_json(tmp_path / "prompts.json", [{"not_prompt": 1}])
    with pytest.raises(ValueError) as excinfo:
        _load_prompts(str(path))
    assert "Invalid input file" in str(excinfo.value)


def test_validate_registry_empty() -> None:
    """An empty registry is rejected."""
    registry = FunctionRegistry(functions=[])
    with pytest.raises(ValueError) as excinfo:
        _validate_registry(registry, "functions.json")
    assert "No functions provided" in str(excinfo.value)


def test_validate_registry_duplicates() -> None:
    """Duplicate function names are rejected."""
    registry = FunctionRegistry.model_validate(
        {"functions": [VALID_FUNCTIONS[0], VALID_FUNCTIONS[0]]}
    )
    with pytest.raises(ValueError) as excinfo:
        _validate_registry(registry, "functions.json")
    assert "Duplicated function names" in str(excinfo.value)


def test_validate_registry_keyword_name() -> None:
    """A function named with a Python keyword is rejected."""
    registry = FunctionRegistry.model_validate(
        {
            "functions": [
                {
                    "name": "class",
                    "description": "Bad keyword name.",
                    "parameters": {},
                    "returns": {"type": "number"},
                }
            ]
        }
    )
    with pytest.raises(ValueError) as excinfo:
        _validate_registry(registry, "functions.json")
    assert "Python keyword" in str(excinfo.value)


def test_validate_prompts_empty() -> None:
    """An empty prompts list is rejected."""
    with pytest.raises(ValueError) as excinfo:
        _validate_prompts([], "prompts.json")
    assert "No prompts provided" in str(excinfo.value)


def test_validate_prompts_empty_string() -> None:
    """A prompt with an empty string is rejected."""
    with pytest.raises(ValueError) as excinfo:
        _validate_prompts([Prompt(prompt="")], "prompts.json")
    assert "Empty prompt" in str(excinfo.value)
