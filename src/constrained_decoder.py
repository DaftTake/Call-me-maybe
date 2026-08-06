"""Constrained decoding engine for function calls."""

from __future__ import annotations

import math
from typing import Any, List

from rich.live import Live

from src.errors import DecodingError
from src.llm_wrapper import LLMWrapper
from src.models import Function, FunctionCallResult
from src.ui import live_update_function_call

_TOKEN_STR_CACHE: dict[tuple[int, int], str] = {}


def _prefill(model: LLMWrapper, context: str) -> tuple[list[float], Any]:
    """Encode the context and return initial logits plus cache."""
    return model.prefill_logits(model.encode(context))


def _token_str(model: LLMWrapper, token_id: int) -> str:
    """Return the decoded string for a token id, cached per model."""
    key = (id(model), token_id)
    if key not in _TOKEN_STR_CACHE:
        _TOKEN_STR_CACHE[key] = model.decode([token_id])
    return _TOKEN_STR_CACHE[key]


def _ensure_decodable(logits: list[float], message: str) -> None:
    """Raise if masking removed every valid token."""
    if not any(math.isfinite(value) for value in logits):
        raise DecodingError(message)


def decode_function_name(
    model: LLMWrapper,
    context: str,
    valid_names: List[str],
    live: Live,
    result: FunctionCallResult,
) -> None:
    """Generate a function name with token-level constraint checking."""
    logits, past_key_values = _prefill(model, context)
    generated_ids: List[int] = []
    found_match = False

    prefix_allowed: dict[tuple, set[int]] = {}
    for seq in (model.encode(n) for n in valid_names):
        for pos in range(len(seq)):
            prefix_allowed.setdefault(tuple(seq[:pos]), set()).add(seq[pos])

    quote_ids = model.encode('"')
    quote_id = quote_ids[0] if quote_ids else None

    while True:
        allowed_next = prefix_allowed.get(tuple(generated_ids), set())
        if not allowed_next and not found_match:
            raise DecodingError(
                "Unable to decode a valid function name for the "
                "provided prompt."
            )
        for token_id in range(len(logits)):
            if token_id in allowed_next:
                continue
            if found_match and quote_id is not None and token_id == quote_id:
                continue
            logits[token_id] = float("-inf")

        _ensure_decodable(
            logits,
            "Unable to decode a valid function name for the "
            "provided prompt.",
        )

        next_token_id = model.next_token(logits)
        next_token_str = _token_str(model, next_token_id)
        if next_token_str == '"' and found_match:
            break

        generated_ids.append(next_token_id)
        result.name += next_token_str
        if len(generated_ids) % 4 == 0:
            live_update_function_call(live, result)
        if result.name in valid_names:
            found_match = True
        logits, past_key_values = model.advance_logits(
            next_token_id, past_key_values
        )

    live_update_function_call(live, result)


def decode_parameter_string(
    model: LLMWrapper,
    context: str,
    live: Live,
    result: FunctionCallResult,
    param_name: str,
) -> None:
    """Generate a string parameter; stops at a closing quote."""
    result.parameters[param_name] = ""
    logits, past_key_values = _prefill(model, context)
    generated_ids: List[int] = []
    generated_str = ""

    while True:
        _ensure_decodable(
            logits, f"Unable to decode string parameter '{param_name}'."
        )
        next_token_id = model.next_token(logits)
        next_token_str = _token_str(model, next_token_id)
        if '"' in next_token_str:
            break

        generated_ids.append(next_token_id)
        generated_str += next_token_str
        result.parameters[param_name] = generated_str
        logits, past_key_values = model.advance_logits(
            next_token_id, past_key_values
        )
        if len(generated_ids) % 4 == 0:
            live_update_function_call(live, result)

    result.parameters[param_name] = generated_str.strip()
    live_update_function_call(live, result)


def _is_valid_float_prefix(s: str) -> bool:
    """Return True if s is a valid float or a prefix of one."""
    if not s:
        return True
    if any(c.isspace() for c in s):
        return False
    if s in ("-", ".", "-."):
        return True
    if s.lower().endswith(("e", "e-", "e+")):
        try:
            float(s + "0")
            return True
        except ValueError:
            return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def _digit_token_ids(
    model: LLMWrapper, logits: list[float], charset: str
) -> tuple[set[int], set[int]]:
    """Return numeric-related and delimiter token ids."""
    numeric_ids: set[int] = set()
    delimiter_ids: set[int] = set()
    for tid in range(len(logits)):
        tstr = _token_str(model, tid)
        if tstr in (",", "}"):
            delimiter_ids.add(tid)
        elif tstr and all(c in charset for c in tstr):
            numeric_ids.add(tid)
    return numeric_ids, delimiter_ids


def decode_parameter_float(
    model: LLMWrapper,
    context: str,
    live: Live,
    result: FunctionCallResult,
    param_name: str,
    max_generated_tokens: int = 12,
) -> None:
    """Generate a float parameter with constrained decoding."""
    number_str = ""
    logits, past_key_values = _prefill(model, context)
    generated_ids: List[int] = []
    numeric_ids, delimiter_ids = _digit_token_ids(
        model, logits, "0123456789eE+-."
    )
    candidate_ids = numeric_ids | delimiter_ids

    while len(generated_ids) < max_generated_tokens:
        for tid in range(len(logits)):
            if tid not in candidate_ids:
                logits[tid] = float("-inf")
        for tid in candidate_ids:
            tstr = _token_str(model, tid)
            if tstr in (",", "}"):
                if not _is_valid_float_prefix(number_str):
                    logits[tid] = float("-inf")
            elif not _is_valid_float_prefix(number_str + tstr):
                logits[tid] = float("-inf")

        _ensure_decodable(
            logits, f"Unable to decode float parameter '{param_name}'."
        )

        next_tid = model.next_token(logits)
        tstr = _token_str(model, next_tid)
        if tstr in (",", "}"):
            break

        generated_ids.append(next_tid)
        number_str += tstr
        try:
            result.parameters[param_name] = float(number_str)
        except ValueError:
            result.parameters[param_name] = None
        logits, past_key_values = model.advance_logits(
            next_tid, past_key_values
        )
        if len(generated_ids) % 4 == 0:
            live_update_function_call(live, result)

    live_update_function_call(live, result)


def decode_parameter_integer(
    model: LLMWrapper,
    context: str,
    live: Live,
    result: FunctionCallResult,
    param_name: str,
    max_generated_tokens: int = 8,
) -> None:
    """Generate an integer parameter with constrained decoding."""
    number_str = ""
    logits, past_key_values = _prefill(model, context)
    generated_ids: List[int] = []
    int_ids, delimiter_ids = _digit_token_ids(model, logits, "0123456789-")
    candidate_ids = int_ids | delimiter_ids

    while len(generated_ids) < max_generated_tokens:
        for tid in range(len(logits)):
            if tid not in candidate_ids:
                logits[tid] = float("-inf")
        for tid in candidate_ids:
            tstr = _token_str(model, tid)
            if tstr in (",", "}"):
                if not number_str.lstrip("-").isdigit():
                    logits[tid] = float("-inf")
            else:
                candidate = number_str + tstr
                if candidate != "-" and not candidate.lstrip("-").isdigit():
                    logits[tid] = float("-inf")

        _ensure_decodable(
            logits, f"Unable to decode integer parameter '{param_name}'."
        )

        next_tid = model.next_token(logits)
        tstr = _token_str(model, next_tid)
        if tstr in (",", "}"):
            break

        generated_ids.append(next_tid)
        number_str += tstr
        try:
            result.parameters[param_name] = int(number_str)
        except ValueError:
            result.parameters[param_name] = None
        logits, past_key_values = model.advance_logits(
            next_tid, past_key_values
        )
        if len(generated_ids) % 4 == 0:
            live_update_function_call(live, result)

    live_update_function_call(live, result)


def decode_parameter_boolean(
    model: LLMWrapper,
    context: str,
    live: Live,
    result: FunctionCallResult,
    param_name: str,
) -> None:
    """Generate a boolean parameter (true or false)."""
    output_str = ""
    parsed_bool = False
    logits, past_key_values = _prefill(model, context)
    generated_ids: List[int] = []

    prefix_allowed: dict[tuple, set[int]] = {}
    for seq in (model.encode(t) for t in ("true", "false")):
        for pos in range(len(seq)):
            prefix_allowed.setdefault(tuple(seq[:pos]), set()).add(seq[pos])

    while True:
        allowed_next = prefix_allowed.get(tuple(generated_ids), set())
        if not allowed_next:
            raise DecodingError("Unable to decode a boolean parameter.")
        for token_id in range(len(logits)):
            if token_id not in allowed_next:
                logits[token_id] = float("-inf")

        _ensure_decodable(logits, "Unable to decode a boolean parameter.")

        next_token_id = model.next_token(logits)
        next_token_str = _token_str(model, next_token_id)
        full_candidate = (output_str + next_token_str).lower()
        if "true" in full_candidate:
            parsed_bool = True
            break
        if "false" in full_candidate:
            parsed_bool = False
            break

        generated_ids.append(next_token_id)
        output_str += next_token_str
        logits, past_key_values = model.advance_logits(
            next_token_id, past_key_values
        )

    result.parameters[param_name] = parsed_bool
    live_update_function_call(live, result)


def decode_function_arguments(
    model: LLMWrapper,
    context: str,
    function_def: Function,
    live: Live,
    result: FunctionCallResult,
) -> None:
    """Generate all parameters for a function call."""
    for param_name, param_type in function_def.parameters.items():
        context += f'"{param_name}": '
        result.parameters[param_name] = None
        live_update_function_call(live, result)

        match param_type.type:
            case "string":
                context += '"'
                decode_parameter_string(
                    model, context, live, result, param_name
                )
                context += f'{result.parameters[param_name]}",'
            case "number":
                decode_parameter_float(
                    model,
                    context,
                    live,
                    result,
                    param_name,
                    max_generated_tokens=8,
                )
                context += f'{result.parameters[param_name]},'
            case "integer":
                decode_parameter_integer(
                    model,
                    context,
                    live,
                    result,
                    param_name,
                    max_generated_tokens=6,
                )
                context += f'{result.parameters[param_name]},'
            case "boolean":
                decode_parameter_boolean(
                    model, context, live, result, param_name
                )
                context += f'{str(result.parameters[param_name]).lower()},'
