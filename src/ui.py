"""Terminal UI components for displaying progress and results.

Provides rich console output for displaying function registries, prompts,
live function call generation, execution summaries, and error messages.
"""

from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
import json

from src.models import FunctionRegistry, FunctionCallResult
import sys

console = Console()
log = console.log


def print_error(msg: str) -> None:
    """Print an error message and exit."""
    console.print(f"[bold red]Error:[/] {msg}")
    sys.exit(1)


def print_header() -> None:
    """Print the application title banner."""
    title = Text(
        "\n☎ Call Me Maybe (Constrained Decoding Engine) ☎\n",
        justify="center",
        style="bold bright_cyan",
    )
    console.print(
        Panel(
            title,
            border_style="bright_blue",
            padding=(1, 4),
            title="[bold green]System Online[/]",
            title_align="left",
            subtitle="[dim]Ready for Generation[/]",
            subtitle_align="right",
        )
    )


def show_registry(registry: FunctionRegistry) -> None:
    """Display the loaded function registry in a nice table."""
    table = Table(
        title="Loaded Function Registry",
        show_header=True,
        header_style="bold magenta",
        expand=True
    )
    table.add_column("Function Name", style="cyan", width=20)
    table.add_column("Description", style="white")
    table.add_column("Parameters", style="green")

    for func in registry.functions:
        params_str = ", ".join(f"{
            name}: {param.type}" for name, param in func.parameters.items())
        table.add_row(func.name, func.description, params_str or "None")

    console.print(table)
    console.print()


def show_prompts(prompts: List[str]) -> None:
    """Display the loaded prompts."""
    console.print("[bold magenta]Loaded Prompts to Process:[/bold magenta]")
    for i, prompt in enumerate(prompts, start=1):
        console.print(f"  [cyan]{i}.[/cyan] {prompt}")
    console.print()


def live_update_function_call(live: Live, result: FunctionCallResult) -> None:
    """Update the live UI display with the current state of generation."""
    # Convert parameters to formatted JSON string
    try:
        params_json = json.dumps(result.parameters, indent=2)
    except Exception:
        params_json = str(result.parameters)

    content = Text()
    content.append("Prompt: ", style="bold cyan")
    content.append(f"{result.prompt}\n\n", style="white")

    content.append("Generating Function: ", style="bold magenta")
    content.append(f"{result.name}\n\n", style="green")

    content.append("Generating Parameters:\n", style="bold magenta")
    content.append(f"{params_json}", style="yellow")

    panel = Panel(
        content,
        title="[bold blue]Constrained Decoder Output[/]",
        border_style="bright_blue",
    )
    live.update(panel)


def show_summary(total_time: float, num_prompts: int) -> None:
    """Display the execution summary."""
    console.print("\n[bold green]Generation Complete[/bold green]")
    console.print(
         f"Procesed [cyan]{
            num_prompts}[/cyan] prompts in [yellow]{
                total_time:.2f}s[/yellow].")
