# AI Log — `engine.py` – `run` Function

**Date**: 2026-11-04T14:30:40+02:00  
**Model**: Claude Opus 4.6 (Thinking)  
**Author**: Antigravity (AI Coding Assistant)

---

## Prompt Summary

The user requested the main execution loop `run(grid, automaton, generations,
seed, on_step)` in `cellab/engine.py` — the top-level function that drives
the simulation for a given number of generations with strict reproducibility
and an optional per-step callback.

### Key Requirements Given

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Initialize `rng = np.random.default_rng(seed)` — no global randomness | ✅ Implemented |
| 2 | Loop `generations` times calling `step(grid, automaton, rng)` | ✅ Implemented |
| 3 | Call `on_step(t, grid)` after each generation if callback provided | ✅ Implemented |
| 4 | Return the final grid | ✅ Implemented |
| 5 | Same seed always produces the same sequence of grids | ✅ Verified |

### Signature

```python
def run(
    grid: np.ndarray,
    automaton: Automaton,
    generations: int,
    seed: int | None = None,
    on_step: Callable[[int, np.ndarray], None] | None = None,
) -> np.ndarray:
```

### Design Decisions

1. **`t` starts at 1.**  The callback receives `t` in `range(1, generations+1)`,
   so `t=1` is the first computed generation, matching the natural "generation
   number" convention (generation 0 is the initial state).

2. **`on_step` receives the live grid.**  The callback gets the actual array
   (not a copy) for performance.  If callers need to store history they should
   copy inside their callback (as Member 3's history recorder will do).

3. **Single `rng` instance** is created once and passed through every `step`
   call, so stochastic rules consume the random stream in a deterministic
   order.

### Verification

- **Blinker period-2**: `run(blinker, life, 4, seed=0)` → final grid equals
  initial grid ✅
- **Callback**: `on_step` called exactly 4 times with `t = 1, 2, 3, 4` ✅
- **Reproducibility**: `run(..., seed=0)` called twice → identical results ✅
- **All 66 tests** pass (including other team members' tests) ✅

### Files Modified / Created

| File | Change |
|------|--------|
| `cellab/engine.py` | Added `run` function; updated module docstring and imports |
| `ai_logs/engine_run_log.md` | This log file |
