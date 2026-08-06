from __future__ import annotations

import pytest

from src.constrained_decoder import decode_parameter_float, decode_parameter_integer
from src.errors import DecodingError
from src.models import FunctionCallResult


class DummyLive:
    def update(self, *_args, **_kwargs) -> None:
        pass


class FakeModel:
    def __init__(self) -> None:
        self._tokens = {
            0: "<pad>",
            1: "1",
            2: "2",
            3: ".",
            4: "e",
            5: "-",
            6: "+",
            7: "}",
            8: ",",
        }

    def encode(self, _text: str) -> list[int]:
        return []

    def decode(self, ids: list[int]) -> str:
        return "".join(self._tokens[i] for i in ids)

    def prefill_logits(
         self, _input_ids: list[int]) -> tuple[list[float], object]:
        logits = [0.0] * len(self._tokens)
        logits[1] = 10.0
        return logits, object()

    def advance_logits(
         self, _token_id: int, _past_key_values: object) -> tuple[
         list[float], object]:
        logits = [0.0] * len(self._tokens)
        logits[1] = 10.0
        return logits, object()

    def next_token(self, logits: list[float]) -> int:
        return max(range(len(logits)), key=lambda idx: logits[idx])


class ImpossibleModel:
    def encode(self, _text: str) -> list[int]:
        return []

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return "x"

    def prefill_logits(self, _input_ids: list[int]) -> tuple[
         list[float], object]:
        return [10.0], object()

    def advance_logits(
         self, _token_id: int, _past_key_values: object) -> tuple[list[float], object]:
        return [10.0], object()

    def next_token(self, logits: list[float]) -> int:
        return max(range(len(logits)), key=lambda idx: logits[idx])


def test_decode_parameter_float_stops_after_budget() -> None:
    model = FakeModel()
    result = FunctionCallResult(prompt="test", name="fn_get_square_root", parameters={})

    decode_parameter_float(
        model,
        context="",
        live=DummyLive(),
        result=result,
        param_name="a",
        max_generated_tokens=3,
    )

    assert result.parameters["a"] == 111.0


def test_decode_parameter_integer_stops_after_budget() -> None:
    model = FakeModel()
    result = FunctionCallResult(prompt="test", name="fn_add_numbers", parameters={})

    decode_parameter_integer(
        model,
        context="",
        live=DummyLive(),
        result=result,
        param_name="a",
        max_generated_tokens=2,
    )

    assert result.parameters["a"] == 11


def test_decode_parameter_float_raises_when_no_tokens_are_valid() -> None:
    model = ImpossibleModel()
    result = FunctionCallResult(prompt="test", name="fn_get_square_root", parameters={})

    with pytest.raises(DecodingError, match="Unable to decode float parameter"):
        decode_parameter_float(
            model,
            context="",
            live=DummyLive(),
            result=result,
            param_name="a",
        )