"""User interface helpers for the callme project.

Provides a simple, consistent way to print error messages and terminate
the program gracefully on configuration or runtime errors.
"""

from __future__ import annotations

import sys


def print_error(message: str) -> None:
    """Print an error message to stderr and exit with a non-zero status.

    Args:
        message: The error message to display to the user.
    """
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)
