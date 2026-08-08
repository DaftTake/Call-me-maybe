"""Entry point for the callme program."""



import sys
import time
from typing import Callable

from pydantic import ValidationError
from rich.live import Live

from src.constrained_decoder import (
    decode_function_arguments,
    decode_function_name,
)
from src.errors import CallMeMaybeError, DecodingError, UnknownFunctionError
from src.llm_wrapper import LLMWrapper
from src.models import FunctionCallResult, FunctionCallResults
from src.parser import parse
from src.ui import (
    console,
    print_header,
    show_prompts,
    show_progress_bar,
    show_registry,
    show_summary,
)

MAX_RETRIES = 3


def _noop(*args: object, **kwargs: object) -> None:
    """No-op logger used when debug output is disabled."""


def _load_cached(output_path: str) -> FunctionCallResults:
    """Load previously written results so re-runs skip done prompts."""
    import json

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return FunctionCallResults.model_validate(data)
    except Exception:
        return FunctionCallResults()


def main() -> None:
    """Parse arguments, load inputs, and run the generation pipeline."""
    function_registry, prompts, output_path, model_name, debug = parse()

    print("\033[2J\033[H\033[3J", end="")
    print_header()
    logger: Callable[..., None] = console.log if debug else _noop

    show_registry(function_registry)
    show_prompts([p.prompt for p in prompts])

    try:
        model = LLMWrapper(model_name=model_name)
        function_names = function_registry.names
        registry_json = function_registry.model_dump_json()

        results = _load_cached(str(output_path))
        done_prompts = {r.prompt for r in results.results}
        pending = [p for p in prompts if p.prompt not in done_prompts]

        start_time = time.perf_counter()
        total = len(pending)

        for idx, p in enumerate(pending, start=1):
            result = FunctionCallResult(
                prompt=p.prompt, name="", parameters={}
            )

            context = (
                "You are a natural language to function call system.\n"
                "Given this function registry:\n"
                f"{registry_json}\n"
                "Chose the appropriate function and its parameters based "
                "on the user input.\n"
                "{\n"
                f'    "prompt": "{p.prompt}",\n'
                f'    "name": "'
            )

            attempt = 0
            while attempt < MAX_RETRIES:
                try:
                    with Live(
                        console=console, refresh_per_second=10
                    ) as live:
                        logger("Generating function name...")
                        decode_function_name(
                            model, context, function_names, live, result
                        )
                        logger("Done generating function name.")

                        context += (
                            f'{result.name}",\n    "parameters": {{\n        '
                        )

                        function_def = function_registry.get(result.name)
                        if function_def is None:
                            raise UnknownFunctionError(
                                "The model selected an unknown function "
                                f"'{result.name}'."
                            )

                        logger("Generating function parameters...")
                        decode_function_arguments(
                            model, context, function_def, live, result
                        )
                        logger("Done generating function parameters.")
                    break
                except DecodingError as error:
                    attempt += 1
                    logger(f"Retry {attempt}/{MAX_RETRIES}: {error}")
                    result = FunctionCallResult(
                        prompt=p.prompt, name="", parameters={}
                    )
                    if attempt >= MAX_RETRIES:
                        raise

            results.results.append(result)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(results.dump())
            show_progress_bar(idx, total)

        total_time = time.perf_counter() - start_time
        show_summary(total_time, len(prompts))
    except OSError:
        console.print(
            "[bold red]Error:[/] Invalid Hugging Face model "
            f"identifier: '{model_name}'"
        )
        sys.exit(1)
    except ValidationError:
        console.print(
            "[bold red]Error:[/] Model unsupported by llm_sdk: "
            f"'{model_name}'"
        )
        sys.exit(1)
    except RuntimeError:
        console.print(
            f"[bold red]Error:[/] Unable to run model: '{model_name}'"
        )
        sys.exit(1)
    except CallMeMaybeError as error:
        console.print(f"[bold red]Error:[/] {error}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user. Exiting...[/bold red]")
        sys.exit(0)
