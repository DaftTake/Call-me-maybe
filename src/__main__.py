"""Entry point for the callme program.

Usage:
    uv run python -m src [--functions_definition <file>]
        [--input <file>] [--output <file>] [--model <name>] [--debug]
"""

from __future__ import annotations

import time
import sys
from pydantic import ValidationError
from rich.live import Live

from src.parser import parse
from src.llm_wrapper import LLMWrapper
from src.models import FunctionCallResult, FunctionCallResults
from src.constrained_decoder import decode_function_name, decode_function_arguments
from src.ui import (
    console,
    log,
    print_header,
    show_registry,
    show_prompts,
    show_summary,
    live_update_function_call,
)


def main() -> None:
    """Parse arguments, load inputs, and run the generation pipeline."""
    function_registry, prompts, output_path, model_name, debug = parse()

    # Clear terminal
    print("\033[2J\033[H\033[3J", end="")
    print_header()

    if not debug:
        global log
        log = lambda *args, **kwargs: None

    show_registry(function_registry)
    show_prompts([p.prompt for p in prompts])

    try:
        model = LLMWrapper(model_name=model_name)
    except OSError:
        console.print(f"[bold red]Error:[/] Invalid Hugging Face model identifier: '{model_name}'")
        sys.exit(1)
    except ValidationError:
        console.print(f"[bold red]Error:[/] Model unsupported by llm_sdk: '{model_name}'")
        sys.exit(1)
    except RuntimeError:
        console.print(f"[bold red]Error:[/] Unable to run model: '{model_name}'")
        sys.exit(1)

    function_names = function_registry.names
    registry_json = function_registry.model_dump_json()

    results = FunctionCallResults()
    start_time = time.perf_counter()

    for i, p in enumerate(prompts, start=1):
        prompt_start_time = time.perf_counter()
        
        result = FunctionCallResult(prompt=p.prompt, name="", parameters={})
        
        # Build the system context
        context = (
            "You are a natural language to function call system.\n"
            "Given this function registry:\n"
            f"{registry_json}\n"
            "Chose the appropriate function and its parameters based on the user input.\n"
            "{\n"
            f'    "prompt": "{p.prompt}",\n'
            f'    "name": "'
        )

        with Live(console=console, refresh_per_second=10) as live:
            log("Generating function name...")
            decode_function_name(model, context, function_names, live, result)
            log("Done generating function name.")
            
            # Extend context with generated name and start parameters
            context += f'{result.name}",\n    "parameters": {{\n        '
            
            # Find the chosen function definition
            function_def = function_registry.get(result.name)
            
            log("Generating function parameters...")
            if function_def:
                decode_function_arguments(model, context, function_def, live, result)
            log("Done generating function parameters.")

        results.results.append(result)
        
        log(f"Writing function call to output file: {output_path}")
        with open(output_path, "w") as f:
            f.write(results.dump())
        log("Done writing output.")

    total_time = time.perf_counter() - start_time
    show_summary(total_time, len(prompts))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user. Exiting...[/bold red]")
        sys.exit(0)
