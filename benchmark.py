"""CelLab – Engine benchmark.

Runs Game of Life on a 200×200 grid for 1 000 generations and prints
the wall-clock time.  Uses the library JSON so nothing is hardcoded.
"""

import time

import numpy as np

from cellab.library import load_automaton
from cellab.world import random_world
from cellab.engine import run

# ── setup ────────────────────────────────────────────────────────────
automaton = load_automaton("game_of_life")
grid = random_world(
    (200, 200), automaton, {"ALIVE": 0.5}, seed=42,
)

# ── benchmark ────────────────────────────────────────────────────────
generations = 1_000

print(f"Running {automaton.name} on {grid.shape[0]}×{grid.shape[1]} "
      f"for {generations:,} generations …")

start = time.perf_counter()
final = run(grid, automaton, generations=generations, seed=42)
elapsed = time.perf_counter() - start

print(f"Done in {elapsed:.2f}s  "
      f"({generations / elapsed:,.0f} gen/s, "
      f"{elapsed / generations * 1000:.2f} ms/gen)")
