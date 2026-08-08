"""Tests for error handling in the main entry point."""



from pathlib import Path
from typing import Any, Tuple

import pytest

from src.models import (
    Function,
    FunctionRegistry,
    ParameterType,
    Prompt,
    ReturnType,
)


def test_main_exits_on_unknown_function(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """main() exits with code 1 when the model picks an unknown function."""
    from src import __main__ as app

    registry = FunctionRegistry(
        functions=[
            Function(
                name="fn_add_numbers",
                description="Add two numbers.",
                parameters={
                    "a": ParameterType(type="number"),
                    "b": ParameterType(type="number"),
                },
                returns=ReturnType(type="number"),
            )
        ]
    )
    prompts = [Prompt(prompt="Do something unknown")]

    class DummyLive:
        """A no-op live display context manager."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "DummyLive":
            return self

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            tb: object,
        ) -> None:
            return None

    class DummyModel:
        """A dummy model whose init always succeeds."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    def fake_parse() -> Tuple[FunctionRegistry, list[Prompt], Path, str, bool]:
        return registry, prompts, tmp_path / "out.json", "dummy-model", True

    def fake_decode_function_name(*args: Any, **kwargs: Any) -> None:
        result = args[4]
        result.name = "fn_unknown"

    monkeypatch.setattr(app, "parse", fake_parse)
    monkeypatch.setattr(app, "LLMWrapper", DummyModel)
    monkeypatch.setattr(app, "Live", DummyLive)
    monkeypatch.setattr(app, "decode_function_name", fake_decode_function_name)
    monkeypatch.setattr(
        app, "decode_function_arguments", lambda *args, **kwargs: None
    )

    with pytest.raises(SystemExit) as excinfo:
        app.main()

    assert excinfo.value.code == 1
