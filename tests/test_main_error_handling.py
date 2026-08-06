from __future__ import annotations

import pytest

from src.models import Function, FunctionRegistry, ParameterType, Prompt, ReturnType


def test_main_exits_on_unknown_function(monkeypatch, tmp_path) -> None:
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
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class DummyModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

    def fake_parse():
        return registry, prompts, tmp_path / "out.json", "dummy-model", True

    def fake_decode_function_name(*args, **kwargs) -> None:
        result = args[4]
        result.name = "fn_unknown"

    monkeypatch.setattr(app, "parse", fake_parse)
    monkeypatch.setattr(app, "LLMWrapper", DummyModel)
    monkeypatch.setattr(app, "Live", DummyLive)
    monkeypatch.setattr(app, "decode_function_name", fake_decode_function_name)
    monkeypatch.setattr(app, "decode_function_arguments", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as excinfo:
        app.main()

    assert excinfo.value.code == 1