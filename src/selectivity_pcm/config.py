from pathlib import Path
from runpy import run_path
from typing import Any


def load_python_config(
    path: Path,
    variable: str,
) -> Any:
    """Load a named variable from a Python configuration file."""

    namespace = run_path(str(path))

    if variable not in namespace:
        raise KeyError(
            f"{variable!r} not found in config file: {path}"
        )

    return namespace[variable]