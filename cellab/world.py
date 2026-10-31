"""
CelLab – Initial world generation.

A world is always a 2-D ``np.ndarray`` of dtype ``np.int8`` with
``shape = (H, W)``; each value is the index of a state in
``automaton.states``.  A 1-D automaton is simply a world with one row.

Three ways to create the initial world (project requirement: "set" or
"generated according to rules"):

* :func:`random_world`  – random, with a density per state and a seed;
* :func:`from_template` – a named pattern placed in the centre;
* :func:`from_text`     – read from text such as ``..F..`` (one row per line).

:func:`to_text` is the inverse of :func:`from_text` and is used to print
generations in the terminal.

Nothing here refers to a specific automaton or state name: everything
comes from the ``Automaton`` object (states, symbols, templates).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from cellab.model import Automaton

Shape = Tuple[int, int]
SeedLike = Union[int, np.random.Generator, None]


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _check_shape(shape: Sequence[int]) -> Shape:
    if len(shape) != 2:
        raise ValueError(f"World shape must be (H, W), got {tuple(shape)}.")
    h, w = int(shape[0]), int(shape[1])
    if h < 1 or w < 1:
        raise ValueError(f"World shape must be positive, got ({h}, {w}).")
    return h, w


def _state_index(automaton: Automaton, state: str) -> int:
    try:
        return automaton.states.index(state)
    except ValueError:
        raise ValueError(
            f"Unknown state '{state}'. Known states: {automaton.states}."
        ) from None


def _symbols(automaton: Automaton, symbols: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Return the state -> character mapping to use."""
    result = symbols if symbols is not None else automaton.symbols
    if result is None:
        raise ValueError(
            f"Automaton '{automaton.name}' has no 'symbols'; "
            f"pass symbols explicitly."
        )
    missing = [s for s in automaton.states if s not in result]
    if missing:
        raise ValueError(f"No symbol defined for state(s) {missing}.")
    return result


def _rng(seed: SeedLike) -> np.random.Generator:
    """Never uses the global NumPy random state."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def empty_world(shape: Sequence[int], automaton: Automaton) -> np.ndarray:
    """A world where every cell is in ``default_state``."""
    h, w = _check_shape(shape)
    fill = _state_index(automaton, automaton.default_state)
    return np.full((h, w), fill, dtype=np.int8)


def parse_shape(text: str) -> Shape:
    """``"30x60"`` -> ``(30, 60)``; ``"60"`` -> ``(1, 60)`` (a 1-D world)."""
    parts = text.lower().replace(" ", "").split("x")
    try:
        if len(parts) == 1:
            return _check_shape((1, int(parts[0])))
        if len(parts) == 2:
            return _check_shape((int(parts[0]), int(parts[1])))
    except ValueError as exc:
        raise ValueError(f"Invalid size '{text}': {exc}") from None
    raise ValueError(f"Invalid size '{text}'. Use HxW, e.g. 30x60.")


def parse_density(text: str) -> Dict[str, float]:
    """``"A=0.6,B=0.01"`` -> ``{"A": 0.6, "B": 0.01}``."""
    result: Dict[str, float] = {}
    for item in filter(None, (p.strip() for p in text.split(","))):
        if "=" not in item:
            raise ValueError(f"Invalid density '{item}'. Use STATE=value.")
        state, value = item.split("=", 1)
        result[state.strip()] = float(value)
    return result


# ────────────────────────────────────────────────────────────────────
# 1. Random world
# ────────────────────────────────────────────────────────────────────

def random_world(
    shape: Sequence[int],
    automaton: Automaton,
    density: Dict[str, float],
    seed: SeedLike = None,
) -> np.ndarray:
    """Random world: each cell gets state *s* with probability
    ``density[s]``; the remaining cells get ``default_state``.

    The same seed always gives the same world.
    """
    h, w = _check_shape(shape)
    total = 0.0
    for state, p in density.items():
        _state_index(automaton, state)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Density for '{state}' must be in [0, 1], got {p}.")
        total += p
    if total > 1.0 + 1e-9:
        raise ValueError(f"Densities add up to {total:.3f}, which is more than 1.")

    grid = empty_world((h, w), automaton)
    r = _rng(seed).random((h, w))
    low = 0.0
    for state, p in density.items():
        high = low + p
        grid[(r >= low) & (r < high)] = _state_index(automaton, state)
        low = high
    return grid


# ────────────────────────────────────────────────────────────────────
# 2. Templates
# ────────────────────────────────────────────────────────────────────

def from_template(name: str, shape: Sequence[int], automaton: Automaton) -> np.ndarray:
    """Place a pattern in the centre of an otherwise default world.

    ``name`` is either

    * a general template that works for any automaton:
      ``"center:STATE"`` (one cell in the centre) or ``"row:STATE"``
      (the middle row filled), or
    * the name of a template stored in the automaton's JSON
      (``automaton.templates``), e.g. ``"glider"``.
    """
    grid = empty_world(shape, automaton)
    h, w = grid.shape

    if ":" in name:
        kind, state = name.split(":", 1)
        idx = _state_index(automaton, state)
        if kind == "center":
            grid[h // 2, w // 2] = idx
        elif kind == "row":
            grid[h // 2, :] = idx
        else:
            raise ValueError(
                f"Unknown general template '{kind}'. Use center:STATE or row:STATE."
            )
        return grid

    if name not in automaton.templates:
        available = sorted(automaton.templates) + ["center:STATE", "row:STATE"]
        raise ValueError(
            f"Automaton '{automaton.name}' has no template '{name}'. "
            f"Available: {', '.join(available)}."
        )
    pattern = from_text("\n".join(automaton.templates[name]), automaton)
    ph, pw = pattern.shape
    if ph > h or pw > w:
        raise ValueError(
            f"Template '{name}' is {ph}x{pw} and does not fit in a {h}x{w} world."
        )
    top, left = (h - ph) // 2, (w - pw) // 2
    grid[top:top + ph, left:left + pw] = pattern
    return grid


# ────────────────────────────────────────────────────────────────────
# 3. Text  <->  world
# ────────────────────────────────────────────────────────────────────

def from_text(
    path_or_text: Union[str, Path],
    automaton: Automaton,
    symbols: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Read a world written with the automaton's symbols, one row per line,
    e.g.::

        ..F..
        .TTT.

    ``path_or_text`` may be a path to a text file or the text itself.
    Empty lines at the start/end are ignored.
    """
    sym = _symbols(automaton, symbols)
    char_to_index = {ch: automaton.states.index(st) for st, ch in sym.items()}

    candidate = Path(path_or_text) if "\n" not in str(path_or_text) else None
    if candidate is not None and candidate.is_file():
        text = candidate.read_text(encoding="utf-8")
    else:
        text = str(path_or_text)

    rows = [line.rstrip("\r") for line in text.split("\n")]
    while rows and not rows[0].strip():
        rows.pop(0)
    while rows and not rows[-1].strip():
        rows.pop()
    if not rows:
        raise ValueError("The world text is empty.")

    width = len(rows[0])
    grid = np.empty((len(rows), width), dtype=np.int8)
    for r, line in enumerate(rows):
        if len(line) != width:
            raise ValueError(
                f"Row {r + 1} has {len(line)} characters, expected {width} "
                f"(all rows must have the same length)."
            )
        for c, ch in enumerate(line):
            if ch not in char_to_index:
                raise ValueError(
                    f"Unknown symbol '{ch}' at row {r + 1}, column {c + 1}. "
                    f"Known symbols: {sym}."
                )
            grid[r, c] = char_to_index[ch]
    return grid


def to_text(
    grid: np.ndarray,
    automaton: Automaton,
    symbols: Optional[Dict[str, str]] = None,
) -> str:
    """Inverse of :func:`from_text`: one line per row, using the symbols."""
    sym = _symbols(automaton, symbols)
    chars = [sym[state] for state in automaton.states]
    return "\n".join("".join(chars[v] for v in row) for row in np.asarray(grid))


# ────────────────────────────────────────────────────────────────────
# Default world from the JSON
# ────────────────────────────────────────────────────────────────────

def default_world(automaton: Automaton, seed: SeedLike = None) -> np.ndarray:
    """Build the world described by ``automaton.default_world``, e.g.
    ``{"size": [30, 60], "init": "random", "density": {"A": 0.6}}`` or
    ``{"size": [20, 20], "init": "template", "template": "glider"}``.
    """
    spec = automaton.default_world
    if not spec:
        raise ValueError(f"Automaton '{automaton.name}' has no 'default_world'.")
    shape = tuple(spec.get("size", (20, 20)))
    init = spec.get("init", "random")
    if init == "random":
        return random_world(shape, automaton, spec.get("density", {}), seed)
    if init == "template":
        return from_template(spec["template"], shape, automaton)
    if init == "text":
        return from_text("\n".join(spec["rows"]), automaton)
    raise ValueError(f"Unknown init mode '{init}' in default_world.")
