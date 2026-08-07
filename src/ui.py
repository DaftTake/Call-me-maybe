"""Terminal UI components."""

from __future__ import annotations

import json
import sys
from typing import Any, List

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.models import FunctionCallResult, FunctionRegistry

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
    """Display the loaded function registry in a table."""
    table = Table(
        title="Loaded Function Registry",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Function Name", style="cyan", width=20)
    table.add_column("Description", style="white")
    table.add_column("Parameters", style="green")

    for func in registry.functions:
        params = ", ".join(
            f"{name}: {param.type}"
            for name, param in func.parameters.items()
        )
        table.add_row(func.name, func.description, params or "None")

    console.print(table)
    console.print()


def show_prompts(prompts: List[str]) -> None:
    """Display the loaded prompts."""
    console.print("[bold magenta]Loaded Prompts to Process:[/bold magenta]")
    for i, prompt in enumerate(prompts, start=1):
        console.print(f"  [cyan]{i}.[/cyan] {prompt}")
    console.print()


def live_update_function_call(
    live: Live, result: FunctionCallResult
) -> None:
    """Update the live UI with the current generation state."""
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
        "Procesed "
        f"[cyan]{num_prompts}[/cyan] prompts in "
        f"[yellow]{total_time:.2f}s[/yellow]."
    )


def show_progress_bar(done: int, total: int, width: int = 30) -> None:
    """Print a simple ASCII progress bar (visualization bonus)."""
    filled = int(width * done / total) if total else 0
    bar = "#" * filled + "-" * (width - filled)
    console.print(f"[cyan]Progress:[/] [{bar}] {done}/{total}")


def show_token_breakdown(text: str, token_ids: List[int],
                         tokenizer: Any) -> None:
    """Show how a string maps to tokens (tokenizer integration bonus)."""
    table = Table(title="Token Breakdown", show_header=True)
    table.add_column("Token ID", style="cyan")
    table.add_column("Token", style="green")
    for tid in token_ids:
        table.add_row(str(tid), tokenizer.decode([tid]))
    console.print(table)
    console.print(f"Input: [yellow]{text}[/yellow]")
