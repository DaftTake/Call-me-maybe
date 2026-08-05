"""Constrained decoding engine for function calls.

This module forces the language model to only output structurally valid
function calls that adhere to the provided JSON schema.
"""

from typing import List

from rich.live import Live

from src.models import FunctionCallResult, Function
from src.llm_wrapper import LLMWrapper
from src.ui import live_update_function_call


def decode_function_name(
    model: LLMWrapper,
    context: str,
    valid_names: List[str],
    live: Live,
    result: FunctionCallResult,
) -> None:
    """Generate a function name token-by-token with constraint checking.

    Args:
        model: The LLMWrapper instance.
        context: The prompt context seen so far.
        valid_names: List of acceptable function names from the registry.
        live: The rich Live display instance for UI updates.
        result: The FunctionCallResult object being populated.
    """
    found_match = False

    while True:
        logits = model.get_logits(model.encode(context + result.name))

        # Mask invalid tokens
        for token_id, _ in enumerate(logits):
            token_str = model.decode([token_id])
            candidate_prefix = result.name + token_str

            # Check if this token could lead to a valid name
            is_valid_prefix = any(
                name.startswith(candidate_prefix) for name in valid_names
            )
            
            if not is_valid_prefix:
                # If we already generated a valid full name, the only valid next token is '"'
                if token_str == '"' and found_match:
                    continue
                logits[token_id] = float("-inf")

        next_token_id = model.next_token(logits)
        next_token_str = model.decode([next_token_id])

        if next_token_str == '"' and found_match:
            break

        result.name += next_token_str
        live_update_function_call(live, result)

        if result.name in valid_names:
            found_match = True


def decode_parameter_string(
    model: LLMWrapper,
    context: str,
    live: Live,
    result: FunctionCallResult,
    param_name: str,
) -> None:
    """Generate a string parameter token-by-token.

    Stops when it generates a closing quote.

    Args:
        model: The LLMWrapper instance.
        context: The prompt context seen so far.
        live: The rich Live display instance for UI updates.
        result: The FunctionCallResult object being populated.
        param_name: The name of the parameter being generated.
    """
    result.parameters[param_name] = ""
    while True:
        logits = model.get_logits(
            model.encode(context + result.parameters[param_name])
        )
        next_token_id = model.next_token(logits)
        next_token_str = model.decode([next_token_id])

        if '"' in next_token_str:
            break

        result.parameters[param_name] += next_token_str
        live_update_function_call(live, result)

    result.parameters[param_name] = result.parameters[param_name].strip()
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
) -> None:
    """Generate a float parameter token-by-token with constraint checking.

    Stops at a comma or closing brace.

    Args:
        model: The LLMWrapper instance.
        context: The prompt context seen so far.
        live: The rich Live display instance for UI updates.
        result: The FunctionCallResult object being populated.
        param_name: The name of the parameter being generated.
    """
    number_str = ""
    while True:
        logits = model.get_logits(model.encode(context + number_str))

        for token_id, _ in enumerate(logits):
            token_str = model.decode([token_id])

            if "," in token_str and token_str != ",":
                logits[token_id] = float("-inf")
            if "}" in token_str and token_str != "}":
                logits[token_id] = float("-inf")
            
            # If it's a delimiter, the number collected so far must be valid
            if (token_str == "," or token_str == "}") and not _is_valid_float_prefix(number_str):
                logits[token_id] = float("-inf")
                
            # If it's not a delimiter, it must be part of a valid float
            if (
                token_str != ","
                and token_str != "}"
                and not _is_valid_float_prefix(number_str + token_str)
            ):
                logits[token_id] = float("-inf")

        next_token_id = model.next_token(logits)
        next_token_str = model.decode([next_token_id])

        if next_token_str == "," or next_token_str == "}":
            break

        number_str += next_token_str
        result.parameters[param_name] = float(number_str)
        live_update_function_call(live, result)


def decode_parameter_integer(
    model: LLMWrapper,
    context: str,
    live: Live,
    result: FunctionCallResult,
    param_name: str,
) -> None:
    """Generate an integer parameter token-by-token with constraint checking.

    Stops at a comma or closing brace.

    Args:
        model: The LLMWrapper instance.
        context: The prompt context seen so far.
        live: The rich Live display instance for UI updates.
        result: The FunctionCallResult object being populated.
        param_name: The name of the parameter being generated.
    """
    number_str = ""
    while True:
        logits = model.get_logits(model.encode(context + number_str))

        for token_id, _ in enumerate(logits):
            token_str = model.decode([token_id])

            if "," in token_str and token_str != ",":
                logits[token_id] = float("-inf")
            if "}" in token_str and token_str != "}":
                logits[token_id] = float("-inf")
            
            # If it's a delimiter, the number collected so far must be a valid integer
            if (token_str == "," or token_str == "}") and not number_str.lstrip("-").isdigit():
                logits[token_id] = float("-inf")
                
            # If it's not a delimiter, it must be part of a valid integer
            candidate = number_str + token_str
            is_valid_int_prefix = candidate == "-" or candidate.lstrip("-").isdigit()
            if (
                token_str != ","
                and token_str != "}"
                and not is_valid_int_prefix
            ):
                logits[token_id] = float("-inf")

        next_token_id = model.next_token(logits)
        next_token_str = model.decode([next_token_id])

        if next_token_str == "," or next_token_str == "}":
            break

        number_str += next_token_str
        result.parameters[param_name] = int(number_str)
        live_update_function_call(live, result)


def decode_parameter_boolean(
    model: LLMWrapper,
    context: str,
    live: Live,
    result: FunctionCallResult,
    param_name: str,
) -> None:
    """Generate a boolean parameter token-by-token with constraint checking.

    Forces the model to output 'true' or 'false'.

    Args:
        model: The LLMWrapper instance.
        context: The prompt context seen so far.
        live: The rich Live display instance for UI updates.
        result: The FunctionCallResult object being populated.
        param_name: The name of the parameter being generated.
    """
    output_str = ""
    parsed_bool = False
    
    while True:
        logits = model.get_logits(model.encode(context + output_str))

        for token_id, _ in enumerate(logits):
            token_str = model.decode([token_id])
            candidate = output_str + token_str
            
            if not any(target.startswith(candidate) for target in ("true", "false")):
                logits[token_id] = float("-inf")

        next_token_id = model.next_token(logits)
        next_token_str = model.decode([next_token_id])
        
        full_candidate = (output_str + next_token_str).lower()
        if "true" in full_candidate:
            parsed_bool = True
            break
        if "false" in full_candidate:
            parsed_bool = False
            break
            
        output_str += next_token_str
        
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

    Args:
        model: The LLMWrapper instance.
        context: The prompt context built up so far.
        function_def: The schema definition of the function.
        live: The rich Live display instance for UI updates.
        result: The FunctionCallResult object being populated.
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
                decode_parameter_float(model, context, live, result, param_name)
                context += f'{result.parameters[param_name]},'
            case "integer":
                decode_parameter_integer(model, context, live, result, param_name)
                context += f'{result.parameters[param_name]},'
            case "boolean":
                decode_parameter_boolean(model, context, live, result, param_name)
                # Boolean in json context usually doesn't have a trailing comma unless more elements
                # The reference had logic here but basically we just append it
                context += f'{str(result.parameters[param_name]).lower()},'
