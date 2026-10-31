"""
CelLab – Library of predefined automata.

The library is simply the folder ``library/`` at the project root: every
``*.json`` file in it is one automaton in the JSON v2 format.  Nothing
here knows about any particular automaton; adding a new one means only
dropping a new JSON file into the folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from cellab.model import Automaton, load_automaton_file

# <project root>/library  (this file lives in <project root>/cellab/)
LIBRARY_DIR = Path(__file__).resolve().parent.parent / "library"


def list_automata(library_dir: Union[str, Path] = LIBRARY_DIR) -> List[str]:
    """Return the names (file names without ``.json``) of all automata
    in the library, sorted alphabetically."""
    folder = Path(library_dir)
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.json"))


def load_automaton(
    name: str, library_dir: Union[str, Path] = LIBRARY_DIR
) -> Automaton:
    """Load and validate one automaton from the library.

    ``name`` is the file name without extension, e.g. ``"forest_fire"``.
    A path to any ``.json`` file is also accepted, so the same function
    works for automata that are not in the library.
    """
    candidate = Path(name)
    if candidate.suffix == ".json" and candidate.exists():
        return load_automaton_file(candidate)

    path = Path(library_dir) / f"{name}.json"
    if not path.exists():
        available = ", ".join(list_automata(library_dir)) or "(none)"
        raise FileNotFoundError(
            f"No automaton named '{name}' in {library_dir}. "
            f"Available: {available}"
        )
    return load_automaton_file(path)
