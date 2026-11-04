# AI Log — `engine.py` – `step` Function

**Date**: 2026-11-02T14:29:51+02:00  
**Model**: Claude Opus 4.6 (Thinking)  
**Author**: Antigravity (AI Coding Assistant)

---

## Prompt Summary

The user requested the core `step(grid, automaton, rng) -> np.ndarray` function
in `cellab/engine.py` — the function that computes the next generation of any
cellular automaton by applying rules in order using vectorized boolean masks.

### Key Requirements Given

| # | Requirement | Status |
|---|-------------|--------|
| 1 | **Zero Hardcoding** — no automaton or state name in code | ✅ Verified |
| 2 | **Vectorized** — boolean masks only, no cell-by-cell `for` loops | ✅ Implemented |
| 3 | `changed_mask` ensures first-matching-rule-wins semantics | ✅ Implemented |
| 4 | Condition evaluation using `count_neighbors` + operator dispatch | ✅ Implemented |
| 5 | All 7 operators: `==`, `!=`, `<`, `<=`, `>`, `>=`, `in` | ✅ Implemented |
| 6 | Stochastic rules via `rng.random(grid.shape) < probability` | ✅ Implemented |
| 7 | 1-D pattern matching via `np.roll(shift=1)` / `np.roll(shift=-1)` | ✅ Implemented |
| 8 | No global randomness — only the passed `rng` | ✅ Verified |

### Signature

```python
def step(
    grid: np.ndarray,
    automaton: Automaton,
    rng: np.random.Generator,
) -> np.ndarray:
```

### Helper Functions Added

| Function | Purpose |
|----------|---------|
| `_eval_condition(condition, grid, automaton)` | Evaluates one `Condition` → boolean mask via `count_neighbors` + operator dispatch |
| `_OPS` dict | Maps operator strings (`"=="`, `">="`, …) to `operator.eq`, `operator.ge`, … |

### Design Decisions

1. **Operator dispatch table** (`_OPS`).  Maps the 6 comparison operators to
   `operator.*` functions.  The `"in"` operator is handled separately with
   `np.isin(counts, value_list)`.  This avoids any `if/elif` chain and is
   trivially extensible.

2. **Rule iteration, not cell iteration.**  The outer loop is over *rules*
   (typically 3–6), not over cells (potentially millions).  Inside each
   iteration, all work is done by NumPy array operations over the full grid.

3. **`changed_mask` for first-match semantics.**  Once a cell is transitioned
   by rule *i*, all subsequent rules skip it.  This is critical for Game of
   Life where "die if < 2" and "survive if 2–3" must not collide, and for
   Brian's Brain where the DYING → DEAD rule must not be overridden.

4. **Pattern matching with `np.roll`.**  For 1-D automata (like elementary
   cellular automata), `np.roll(grid, 1, axis=-1)` gives each cell's left
   neighbour and `np.roll(grid, -1, axis=-1)` gives the right.  This is
   naturally toroidal; fixed-boundary 1-D automata would need extra masking
   (not required by the current library).

5. **`rng` is passed, not created.**  The caller controls the seed, ensuring
   reproducible simulations.

### Verification

**Game of Life — blinker oscillation (period 2):**

```
Gen 0:     Gen 1:     Gen 2 (= Gen 0):
.....      .....      .....
.....      ..#..      .....
.###.  →   ..#..  →   .###.
.....      ..#..      .....
.....      .....      .....
```

`assert np.array_equal(gen0, gen2)` — ✅ passed.

All **46 existing tests** continue to pass.

### Files Modified / Created

| File | Change |
|------|--------|
| `cellab/engine.py` | Added `_OPS`, `_eval_condition`, `step`; updated module docstring and imports |
| `ai_logs/engine_step_log.md` | This log file |
