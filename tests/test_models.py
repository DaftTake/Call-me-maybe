"""Unit tests for the pydantic models."""



import pytest
from pydantic import ValidationError

from src.models import (
    Function,
    FunctionCallResult,
    FunctionRegistry,
    ParameterType,
    Prompt,
    ReturnType,
)

NUMBER = ReturnType(type="number")
STRING = ReturnType(type="string")


def _function(
    name: str,
    *,
    description: str = "A function.",
    parameters: dict[str, ParameterType] | None = None,
    returns: ReturnType | None = None,
) -> Function:
    """Build a Function model with sensible defaults."""
    return Function(
        name=name,
        description=description,
        parameters=parameters or {},
        returns=returns or NUMBER,
    )


def test_function_valid() -> None:
    """A well-formed function definition validates."""
    function = _function(
        name="fn_add_numbers",
        description="Add two numbers.",
        parameters={
            "a": ParameterType(type="number"),
            "b": ParameterType(type="number"),
        },
        returns=NUMBER,
    )
    assert function.name == "fn_add_numbers"
    assert function.parameters["a"].type == "number"


def test_function_invalid_name() -> None:
    """A function name that is not a valid identifier is rejected."""
    with pytest.raises(ValidationError):
        _function(name="not valid!")


def test_function_missing_returns() -> None:
    """A function without a returns field is rejected."""
    with pytest.raises(ValidationError):
        Function(
            name="fn_ok",
            description="Missing returns.",
            parameters={},
        )  # type: ignore[call-arg]


def test_registry_names_property() -> None:
    """The names property returns all function names."""
    registry = FunctionRegistry(
        functions=[
            _function(name="fn_a", returns=NUMBER),
            _function(name="fn_b", returns=STRING),
        ]
    )
    assert registry.names == ["fn_a", "fn_b"]


def test_registry_get() -> None:
    """get() returns the matching function or None."""
    registry = FunctionRegistry(
        functions=[_function(name="fn_a", returns=NUMBER)]
    )
    assert registry.get("fn_a") is not None
    assert registry.get("fn_missing") is None


def test_prompt_valid() -> None:
    """A well-formed prompt validates."""
    prompt = Prompt(prompt="What is the sum of 2 and 3?")
    assert prompt.prompt == "What is the sum of 2 and 3?"


def test_prompt_missing_field() -> None:
    """A prompt without the prompt field is rejected."""
    with pytest.raises(ValidationError):
        Prompt()  # type: ignore[call-arg]


def test_result_valid() -> None:
    """A well-formed result validates."""
    result = FunctionCallResult(
        prompt="Greet shrek",
        name="fn_greet",
        parameters={"name": "shrek"},
    )
    assert result.name == "fn_greet"
    assert result.parameters == {"name": "shrek"}
