"""
src/utils.py — shared netlist parsing helpers for the SRAM fault-injection pipeline.
"""

from pathlib import Path
from typing import List


def load_cell_core(path: str) -> List[str]:
    """
    Read a cell-core .net fragment and return its lines as a list of strings.

    Each entry in the returned list corresponds to one line of the file,
    with the trailing newline stripped. Empty lines at the end of the file
    are preserved as empty strings so that round-trips through save_cell_core
    are lossless.

    Parameters
    ----------
    path : str
        Absolute or relative path to the .net file to read.

    Returns
    -------
    list of str
        One entry per line, newlines stripped.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    # Split on newlines; keep internal empty lines, drop a single trailing newline
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def save_cell_core(lines: List[str], path: str) -> None:
    """
    Write a list of strings back out to a .net file.

    Each string in *lines* is written as its own line, joined with newlines.
    A single trailing newline is appended so the file is POSIX-compliant and
    editors do not flag a missing final newline.  No other modifications are
    made — no whitespace normalisation, no reordering.

    Parameters
    ----------
    lines : list of str
        Cell-core lines to write, one element per line.
    path : str
        Destination file path.  Parent directories must already exist.
    """
    p = Path(path)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
