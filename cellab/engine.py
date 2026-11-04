"""
CelLab – Simulation engine.

Fully vectorized routines for running cellular-automaton simulations.
Nothing here refers to a specific automaton or state name; every
parameter is data-driven.

Functions
---------
count_neighbors
    Count how many neighbours of each cell are in a given state, using
    ``scipy.signal.convolve2d`` for O(1)-per-cell performance regardless
    of grid size.
step
    Compute the next generation of the grid by applying the automaton's
    rules in order, using boolean masks (no cell-by-cell loops).
run
    Execute the simulation for a given number of generations, with an
    optional per-step callback and strict seed-based reproducibility.
"""

from __future__ import annotations

import operator
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from scipy.signal import convolve2d

from cellab.model import Automaton, Condition


# ────────────────────────────────────────────────────────────────────
# Kernel construction
# ────────────────────────────────────────────────────────────────────

_KERNELS: Dict[str, np.ndarray] = {
    "moore": np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]], dtype=np.int8,
    ),
    "von neumann": np.array(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]], dtype=np.int8,
    ),
    "left-center-right": np.array(
        [[1, 0, 1]], dtype=np.int8,
    ),
}


def _get_kernel(neighborhood: str) -> np.ndarray:
    """Return the convolution kernel for *neighborhood*.

    Raises ``ValueError`` for unrecognised neighbourhood strings.
    """
    try:
        return _KERNELS[neighborhood]
    except KeyError:
        raise ValueError(
            f"Unknown neighborhood '{neighborhood}'. "
            f"Expected one of: {', '.join(sorted(_KERNELS))}"
        ) from None


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

def count_neighbors(
    grid: np.ndarray,
    state_idx: int,
    neighborhood: str,
    boundary: str,
) -> np.ndarray:
    """Count the neighbours of every cell that are in state *state_idx*.

    Parameters
    ----------
    grid : np.ndarray
        2-D array of integer state indices with shape ``(H, W)``.
    state_idx : int
        The state index to count in each cell's neighbourhood.
    neighborhood : str
        One of ``"moore"``, ``"von neumann"``, or
        ``"left-center-right"``.  Determines the convolution kernel.
    boundary : str
        ``"toroidal"`` → wrap-around (``boundary="wrap"`` in
        ``convolve2d``).
        ``"fixed"`` → cells outside the grid are treated as *not*
        matching *state_idx* (``boundary="fill", fillvalue=0``).

    Returns
    -------
    np.ndarray
        Integer array with the same shape as *grid*.  Each element is
        the number of neighbours in state *state_idx*.

    Notes
    -----
    The function is fully vectorized: it builds a boolean mask of cells
    that match *state_idx*, casts it to ``int8``, then convolves it with
    the neighbourhood kernel.  No Python loops over individual cells.

    For ``boundary="fixed"`` the ``fillvalue=0`` is correct because the
    mask is 0/1: cells outside the grid contribute 0 to the count,
    meaning they are treated as *not* being in state *state_idx*.
    """
    # ── kernel ──────────────────────────────────────────────────────
    kernel = _get_kernel(neighborhood)

    # ── boolean mask → int8 ─────────────────────────────────────────
    mask = (grid == state_idx).astype(np.int8)

    # ── boundary → convolve2d kwargs ────────────────────────────────
    if boundary == "toroidal":
        result = convolve2d(mask, kernel, mode="same", boundary="wrap")
    elif boundary == "fixed":
        result = convolve2d(
            mask, kernel, mode="same", boundary="fill", fillvalue=0,
        )
    else:
        raise ValueError(
            f"Unknown boundary '{boundary}'. Expected 'toroidal' or 'fixed'."
        )

    return result.astype(int)


# ────────────────────────────────────────────────────────────────────
# Operator dispatch table
# ────────────────────────────────────────────────────────────────────

_OPS: Dict[str, Callable] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<":  operator.lt,
    "<=": operator.le,
    ">":  operator.gt,
    ">=": operator.ge,
}


def _eval_condition(
    condition: Condition,
    grid: np.ndarray,
    automaton: Automaton,
) -> np.ndarray:
    """Return a boolean mask where *condition* is satisfied.

    For comparison operators the mask is
    ``op(neighbor_count, value)``; for ``"in"`` it is
    ``np.isin(neighbor_count, value_list)``.
    """
    state_idx = automaton.states.index(condition.count)
    counts = count_neighbors(
        grid, state_idx, automaton.neighborhood, automaton.boundary,
    )

    if condition.op == "in":
        return np.isin(counts, condition.value)

    return _OPS[condition.op](counts, condition.value)


# ────────────────────────────────────────────────────────────────────
# Step function
# ────────────────────────────────────────────────────────────────────

def step(
    grid: np.ndarray,
    automaton: Automaton,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute the next generation of the cellular automaton.

    Parameters
    ----------
    grid : np.ndarray
        Current 2-D grid of integer state indices (``shape = (H, W)``).
    automaton : Automaton
        The automaton definition (states, rules, neighbourhood, …).
    rng : np.random.Generator
        Random generator used for stochastic rules.  Must be a seeded
        ``Generator`` — the global NumPy random state is never used.

    Returns
    -------
    np.ndarray
        New grid with the same shape and dtype as *grid*.

    Notes
    -----
    Rules are applied in declaration order.  The **first** rule whose
    conditions match a given cell wins; later rules cannot overwrite it.
    This is enforced with a ``changed_mask`` that grows as rules fire.

    The function is fully vectorized: every rule produces a boolean mask
    over the whole grid and the transition is applied in one array
    operation.  No Python loops iterate over individual cells.
    """
    new_grid = grid.copy()
    changed_mask = np.zeros(grid.shape, dtype=bool)

    for rule in automaton.rules:
        from_idx = automaton.states.index(rule.from_state)
        to_idx = automaton.states.index(rule.to_state)

        # ── base mask: cells in the source state that haven't changed ─
        mask = (grid == from_idx) & ~changed_mask

        # ── neighbour-count conditions (if any) ──────────────────────
        if rule.if_condition is not None:
            conditions: List[Condition] = (
                rule.if_condition
                if isinstance(rule.if_condition, list)
                else [rule.if_condition]
            )
            for cond in conditions:
                mask = mask & _eval_condition(cond, grid, automaton)

        # ── 1-D pattern condition (left-center-right) ────────────────
        if rule.pattern is not None:
            left_idx = automaton.states.index(rule.pattern[0])
            center_idx = automaton.states.index(rule.pattern[1])
            right_idx = automaton.states.index(rule.pattern[2])

            # np.roll shift=1  → each cell sees its LEFT neighbour
            # np.roll shift=-1 → each cell sees its RIGHT neighbour
            left_match = np.roll(grid, shift=1, axis=-1) == left_idx
            center_match = grid == center_idx
            right_match = np.roll(grid, shift=-1, axis=-1) == right_idx

            mask = mask & left_match & center_match & right_match

        # ── stochastic filter ────────────────────────────────────────
        if rule.probability is not None:
            mask = mask & (rng.random(grid.shape) < rule.probability)

        # ── apply transition ─────────────────────────────────────────
        new_grid[mask] = to_idx
        changed_mask |= mask

    return new_grid


# ────────────────────────────────────────────────────────────────────
# Run loop
# ────────────────────────────────────────────────────────────────────

def run(
    grid: np.ndarray,
    automaton: Automaton,
    generations: int,
    seed: Union[int, None] = None,
    on_step: Optional[Callable[[int, np.ndarray], None]] = None,
) -> np.ndarray:
    """Execute the simulation for *generations* steps.

    Parameters
    ----------
    grid : np.ndarray
        Initial 2-D grid of integer state indices (``shape = (H, W)``).
    automaton : Automaton
        The automaton definition (states, rules, neighbourhood, …).
    generations : int
        Number of simulation steps to run.
    seed : int | None
        Seed for the random number generator.  The same seed always
        produces the same sequence of grids, regardless of platform.
        ``None`` gives non-reproducible randomness.
    on_step : callable, optional
        If provided, called as ``on_step(t, grid)`` immediately after
        generation *t* is computed (``t`` runs from ``1`` to
        ``generations``).  Useful for recording history, rendering
        frames, or collecting statistics.

    Returns
    -------
    np.ndarray
        The grid after all *generations* have been applied.
    """
    rng = np.random.default_rng(seed)

    for t in range(1, generations + 1):
        grid = step(grid, automaton, rng)

        if on_step is not None:
            on_step(t, grid)

    return grid
