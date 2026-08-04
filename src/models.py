"""Pydantic models for the callme project.

These models validate the input files (function definitions and prompts)
and the output results, ensuring type safety and schema compliance.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class ParameterType(BaseModel):
    """A single parameter definition inside a function."""

    type: str = Field(..., description="The JSON type of the parameter")


class ReturnType(BaseModel):
    """The return type of a function."""

    type: str = Field(..., description="The JSON type of the return value")


class Function(BaseModel):
    """A single function definition."""

    name: str = Field(..., description="The name of the function")
    description: str = Field(..., description="A human-readable description")
    parameters: Dict[str, ParameterType] = Field(
        default_factory=dict,
        description="Mapping of parameter name to its type",
    )
    returns: ReturnType = Field(
        ..., description="The return type of the function"
    )

    @field_validator("name")
    @classmethod
    def name_must_be_identifier(cls, value: str) -> str:
        """Ensure the function name is a valid Python identifier."""
        if not value.isidentifier():
            raise ValueError(
                f"Function name '{value}' is not a valid identifier"
            )
        return value


class FunctionRegistry(BaseModel):
    """A collection of available functions."""

    functions: List[Function] = Field(default_factory=list)

    @property
    def names(self) -> List[str]:
        """Return the list of function names."""
        return [f.name for f in self.functions]

    def get(self, name: str) -> Function | None:
        """Return the function with the given name, or None."""
        for function in self.functions:
            if function.name == name:
                return function
        return None


class Prompt(BaseModel):
    """A single natural-language prompt to process."""

    prompt: str = Field(..., description="The natural-language request")


class FunctionCallResult(BaseModel):
    """The output object for a single prompt."""

    prompt: str = Field(
        ..., description="The original natural-language request"
    )
    name: str = Field(..., description="The name of the function to call")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="The arguments for the function call",
    )


class FunctionCallResults(BaseModel):
    """The full output file: a list of results."""

    results: List[FunctionCallResult] = Field(default_factory=list)
