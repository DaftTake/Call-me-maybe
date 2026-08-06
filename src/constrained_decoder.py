"""Constrained decoding engine for function calls.

This module forces the language model to only output structurally valid
function calls that adhere to the provided JSON schema. It implements
token-id level masking using the model's tokenizer so constraints are
deterministic (not heuristic) while being much faster than decoding the
entire vocabulary to strings each generation step.
"""

import math
from typing import Any, List

from rich.live import Live

from src.models import FunctionCallResult, Function
from src.errors import DecodingError
from src.llm_wrapper import LLMWrapper
from src.ui import live_update_function_call

# Module-level cache for decoded token strings to avoid repeated
# expensive `model.decode([token_id])` calls inside tight loops.
_TOKEN_STR_CACHE: dict = {}


def _prefill(model: LLMWrapper, context: str) -> tuple[list[float], Any]:
    """Encode the context once and return initial logits plus cache."""
    context_ids = model.encode(context)
    return model.prefill_logits(context_ids)


def _token_str(model: LLMWrapper, token_id: int) -> str:
    """Return the string for a single token id, caching results.

    The cache key includes the model object's id so the same token id
    used with different model instances won't collide.
    """
    key = (id(model), token_id)
    if key not in _TOKEN_STR_CACHE:
        _TOKEN_STR_CACHE[key] = model.decode([token_id])
    return _TOKEN_STR_CACHE[key]


def _ensure_decodable(logits: list[float], message: str) -> None:
    """Raise a controlled error if masking removed every valid option."""
    if not any(math.isfinite(value) for value in logits):
        raise DecodingError(message)


def decode_function_name(
    model: LLMWrapper,
    context: str,
    valid_names: List[str],
    live: Live,
    result: FunctionCallResult,
) -> None:
    """Generate a function name token-by-token with token-level constraint checking.

    This builds exact token sequences for every `valid_name` using the model's
    tokenizer and then constructs a prefix->allowed-next-token-id map. During
    generation we only allow token ids present in that map for the current
    prefix. This is deterministic and enforces validity exactly (not a heuristic).
    """
    found_match = False

    # Token-id based generation: encode the fixed context once and then
    # append chosen token ids to a growing list. This avoids re-encoding
    # the entire context every generation step.
    logits, past_key_values = _prefill(model, context)
    generated_ids: List[int] = []
    generated_str = ""

    # Precompute token-id sequences for all valid names and build a mapping
    # from token-prefix tuples to the set of allowed next token ids. This
    # is deterministic because it uses the model's tokenizer to get the
    # exact tokenization of each valid name.
    name_token_seqs: List[List[int]] = [model.encode(n) for n in valid_names]
    prefix_allowed: dict[tuple, set[int]] = {}
    for seq in name_token_seqs:
        for pos in range(len(seq)):
            prefix = tuple(seq[:pos])
            prefix_allowed.setdefault(prefix, set()).add(seq[pos])

    # Token id for a double-quote (used to close names)
    quote_ids = model.encode('"')
    quote_id = quote_ids[0] if quote_ids else None

    while True:
        # Mask invalid tokens using the precomputed token-id prefix map.
        allowed_next = prefix_allowed.get(tuple(generated_ids), set())
        if not allowed_next and not found_match:
            raise DecodingError("Unable to decode a valid function name for the provided prompt.")
        for token_id in range(len(logits)):
            if token_id in allowed_next:
                continue
            # Allow closing quote if we've already matched a full name
            if found_match and quote_id is not None and token_id == quote_id:
                continue
            logits[token_id] = float("-inf")

        _ensure_decodable(logits, "Unable to decode a valid function name for the provided prompt.")

        next_token_id = model.next_token(logits)
        next_token_str = _token_str(model, next_token_id)

        if next_token_str == '"' and found_match:
            break

        generated_ids.append(next_token_id)
        generated_str += next_token_str
        result.name += next_token_str

        # Update UI less frequently to reduce overhead
        if len(generated_ids) % 4 == 0:
            live_update_function_call(live, result)

        if result.name in valid_names:
            found_match = True

        logits, past_key_values = model.advance_logits(next_token_id, past_key_values)

    # Final UI update after finishing name
    live_update_function_call(live, result)


def decode_parameter_string(
    model: LLMWrapper,
    context: str,
    live: Live,
    result: FunctionCallResult,
    param_name: str,
) -> None:
    """Generate a string parameter token-by-token.

    Stops when it generates a closing quote. Uses token-id generation to avoid
    repeated encoding of the full context. Strings are free-form, so we allow
    any token until a token containing a quote is produced.
    """
    result.parameters[param_name] = ""
    logits, past_key_values = _prefill(model, context)
    generated_ids: List[int] = []
    generated_str = ""

    while True:
        _ensure_decodable(logits, f"Unable to decode string parameter '{param_name}'.")
        next_token_id = model.next_token(logits)
        next_token_str = _token_str(model, next_token_id)

        if '"' in next_token_str:
            break

        generated_ids.append(next_token_id)
        generated_str += next_token_str
        result.parameters[param_name] = generated_str

        logits, past_key_values = model.advance_logits(next_token_id, past_key_values)

        if len(generated_ids) % 4 == 0:
            live_update_function_call(live, result)

    result.parameters[param_name] = generated_str.strip()
    live_update_function_call(live, result)


def _is_valid_float_prefix(s: str) -> bool:
    """Check if a string represents a valid float or a prefix of one."""
    if not s:
        return True
    # Disallow whitespace/newlines entirely
    if any(c.isspace() for c in s):
        return False
    # Valid incomplete prefixes that float() would reject
    if s in ("-", ".", "-."):
        return True
    if s.lower().endswith("e") or s.lower().endswith("e-") or s.lower().endswith("e+"):
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


def decode_parameter_float(
    model: LLMWrapper,
    context: str,
    live: Live,
    result: FunctionCallResult,
    param_name: str,
    max_generated_tokens: int = 12,
) -> None:
    """Generate a float parameter token-by-token with constraint checking.

    Stops at a comma or closing brace. We speed up masking by precomputing
    token ids whose token strings are numeric-related (digits, sign, dot,
    exponent) and delimiter tokens. We then only validate those tokens
    precisely instead of decoding the entire vocabulary every step.
    """
    number_str = ""
    logits, past_key_values = _prefill(model, context)
    generated_ids: List[int] = []

    # Precompute numeric-related and delimiter token ids once.
    numeric_charset = set("0123456789eE+-.")
    vocab_size = len(logits)
    numeric_token_ids = set()
    delimiter_token_ids = set()
    for tid in range(vocab_size):
        tstr = _token_str(model, tid)
        if tstr in (",", "}"):
            delimiter_token_ids.add(tid)
            continue
        if tstr and all((c in numeric_charset) for c in tstr):
            numeric_token_ids.add(tid)

    while True:
        if len(generated_ids) >= max_generated_tokens:
            break

        # Disallow all tokens that are neither numeric-related nor delimiters.
        for token_id in range(len(logits)):
            if token_id in numeric_token_ids or token_id in delimiter_token_ids:
                continue
            logits[token_id] = float("-inf")

        # Precisely validate numeric and delimiter tokens.
        for token_id in list(numeric_token_ids | delimiter_token_ids):
            token_str = _token_str(model, token_id)
            if token_str not in (",", "}"):
                if not _is_valid_float_prefix(number_str + token_str):
                    logits[token_id] = float("-inf")
            else:
                if not _is_valid_float_prefix(number_str):
                    logits[token_id] = float("-inf")

        _ensure_decodable(logits, f"Unable to decode float parameter '{param_name}'.")

        next_token_id = model.next_token(logits)
        next_token_str = _token_str(model, next_token_id)

        if next_token_str == "," or next_token_str == "}":
            break

        generated_ids.append(next_token_id)
        number_str += next_token_str
        try:
            result.parameters[param_name] = float(number_str)
        except ValueError:
            result.parameters[param_name] = None

        logits, past_key_values = model.advance_logits(next_token_id, past_key_values)

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
    """Generate an integer parameter token-by-token with constraint checking."""
    number_str = ""
    logits, past_key_values = _prefill(model, context)
    generated_ids: List[int] = []

    # Precompute tokens that are integer-related (digits and sign) or delimiters
    int_charset = set("0123456789-")
    vocab_size = len(logits)
    int_token_ids = set()
    delimiter_token_ids = set()
    for tid in range(vocab_size):
        tstr = _token_str(model, tid)
        if tstr in (",", "}"):
            delimiter_token_ids.add(tid)
            continue
        if tstr and all((c in int_charset) for c in tstr):
            int_token_ids.add(tid)

    while True:
        if len(generated_ids) >= max_generated_tokens:
            break

        for token_id in range(len(logits)):
            if token_id in int_token_ids or token_id in delimiter_token_ids:
                continue
            logits[token_id] = float("-inf")

        for token_id in list(int_token_ids | delimiter_token_ids):
            token_str = _token_str(model, token_id)
            if token_str not in (",", "}"):
                candidate = number_str + token_str
                is_valid_int_prefix = candidate == "-" or candidate.lstrip("-").isdigit()
                if not is_valid_int_prefix:
                    logits[token_id] = float("-inf")
            else:
                if not number_str.lstrip("-").isdigit():
                    logits[token_id] = float("-inf")

        _ensure_decodable(logits, f"Unable to decode integer parameter '{param_name}'.")

        next_token_id = model.next_token(logits)
        next_token_str = _token_str(model, next_token_id)

        if next_token_str == "," or next_token_str == "}":
            break

        generated_ids.append(next_token_id)
        number_str += next_token_str
        try:
            result.parameters[param_name] = int(number_str)
        except ValueError:
            result.parameters[param_name] = None

        logits, past_key_values = model.advance_logits(next_token_id, past_key_values)

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
    """Generate a boolean parameter token-by-token with constraint checking.

    We build exact token sequences for "true" and "false" and use a
    prefix->allowed-next-token-id map to deterministically constrain
    generation to those two words.
    """
    output_str = ""
    parsed_bool = False
    logits, past_key_values = _prefill(model, context)
    generated_ids: List[int] = []

    # Build token sequences for the two valid boolean words
    targets = ["true", "false"]
    target_seqs = [model.encode(t) for t in targets]
    prefix_allowed: dict[tuple, set[int]] = {}
    for seq in target_seqs:
        for pos in range(len(seq)):
            prefix = tuple(seq[:pos])
            prefix_allowed.setdefault(prefix, set()).add(seq[pos])

    while True:
        allowed_next = prefix_allowed.get(tuple(generated_ids), set())
        if not allowed_next:
            raise DecodingError("Unable to decode a boolean parameter.")
        for token_id in range(len(logits)):
            if token_id in allowed_next:
                continue
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

        logits, past_key_values = model.advance_logits(next_token_id, past_key_values)

    result.parameters[param_name] = parsed_bool
    live_update_function_call(live, result)


def decode_function_arguments(
    model: LLMWrapper,
    context: str,
    function_def: Function,
    live: Live,
    result: FunctionCallResult,
) -> None:
    """Generate all parameters for a function call using the schema definitions.

    Iterates through each parameter defined in the function signature and delegates
    to the specific type decoder.
    """
    for param_name, param_type in function_def.parameters.items():
        context += f'"{param_name}": '
        result.parameters[param_name] = None
        live_update_function_call(live, result)

        match param_type.type:
            case "string":
                context += '"'
                decode_parameter_string(model, context, live, result, param_name)
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
                decode_parameter_boolean(model, context, live, result, param_name)
                context += f'{str(result.parameters[param_name]).lower()},'
