#!/usr/bin/env python
"""Deterministically convert a ``# %%`` percent-format script into a notebook.

The percent script stays the reviewable source of truth; the ``.ipynb`` is a
generated artifact.  Conversion is byte-deterministic: no timestamps, no
execution counts, and no random identifiers are emitted.

Usage::

    python scripts/nb_from_percent.py notebooks/dcdn_explained.py \
        --output notebooks/dcdn_explained.ipynb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CODE_MARKER = "# %%"
MARKDOWN_MARKER = "# %% [markdown]"


def _split_cells(text: str) -> list[tuple[str, list[str]]]:
    """Return ``(cell_type, source_lines)`` pairs in file order."""
    cells: list[tuple[str, list[str]]] = []
    current_type: str | None = None
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped in (MARKDOWN_MARKER, CODE_MARKER):
            if current_type is not None:
                cells.append((current_type, current))
            current_type = "markdown" if stripped == MARKDOWN_MARKER else "code"
            current = []
            continue
        current.append(line)
    if current_type is not None:
        cells.append((current_type, current))
    return cells


def _strip_markdown_comment(lines: list[str]) -> list[str]:
    """Drop the leading ``# `` from markdown cell lines."""
    out: list[str] = []
    for line in lines:
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return out


def _trim(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _as_source(lines: list[str]) -> list[str]:
    """nbformat stores source as a list of lines that keep their newlines."""
    return [f"{line}\n" for line in lines[:-1]] + lines[-1:] if lines else []


def build_notebook(text: str, kernel_name: str, kernel_display: str) -> dict:
    cells = []
    for cell_type, raw in _split_cells(text):
        lines = _trim(_strip_markdown_comment(raw) if cell_type == "markdown" else raw)
        if not lines:
            continue
        # Deterministic positional IDs.  nbformat >= 4.5 requires an id, and a
        # random one would make the generated notebook non-reproducible.
        cell: dict = {
            "cell_type": cell_type,
            "id": f"cell-{len(cells):03d}",
            "metadata": {},
            "source": _as_source(lines),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": kernel_display,
                "language": "python",
                "name": kernel_name,
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-name", default="python3")
    parser.add_argument("--kernel-display", default="Python 3")
    args = parser.parse_args()

    notebook = build_notebook(
        args.source.read_text(encoding="utf-8"),
        kernel_name=args.kernel_name,
        kernel_display=args.kernel_display,
    )
    args.output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    markdown = len(notebook["cells"]) - code
    print(f"{args.output}: {code} code cells, {markdown} markdown cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
