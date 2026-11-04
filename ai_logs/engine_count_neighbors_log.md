# AI Log — `engine.py` – `count_neighbors` Function

**Date**: 2026-11-02T14:00:52+02:00  
**Model**: Claude Opus 4.6 (Thinking)  
**Author**: Antigravity (AI Coding Assistant)

---

## Prompt Summary

The user requested the first function in `cellab/engine.py` — the simulation
engine helper `count_neighbors`.  This function counts, for every cell in the
grid, how many of its neighbours are in a given state.  It must be fully
vectorized using `scipy.signal.convolve2d` (no loops over cells).

### Key Requirements Given

| # | Requirement | Status |
|---|-------------|--------|
| 1 | **Zero Hardcoding** — no automaton or state name in code | ✅ Verified |
| 2 | Grid is a 2-D `np.ndarray` of integers | ✅ Implemented |
| 3 | **Moore** kernel: 3×3 of 1s with 0 in centre | ✅ Implemented |
| 4 | **Von Neumann** kernel: 3×3 cross (up/down/left/right = 1) | ✅ Implemented |
| 5 | **Left-center-right** kernel: `[[1, 0, 1]]` for 1-D automata | ✅ Implemented |
| 6 | `"toroidal"` → `convolve2d(mode="same", boundary="wrap")` | ✅ Implemented |
| 7 | `"fixed"` → `convolve2d(mode="same", boundary="fill", fillvalue=0)` | ✅ Implemented |
| 8 | Boolean mask `(grid == state_idx).astype(np.int8)` convolved with kernel | ✅ Implemented |
| 9 | No loops over individual cells | ✅ Verified |

### Signature

```python
def count_neighbors(
    grid: np.ndarray,
    state_idx: int,
    neighborhood: str,
    boundary: str,
) -> np.ndarray:
```

### Design Decisions

1. **Kernels are module-level constants** (`_KERNELS` dict) using `np.int8`
   arrays.  A private helper `_get_kernel()` looks up the kernel by
   neighbourhood string, raising `ValueError` for unknown names.

2. **Boolean mask approach.**  The function computes
   `(grid == state_idx).astype(np.int8)` — a 0/1 matrix — then convolves it
   with the kernel.  This naturally counts only the neighbours that match
   `state_idx`.

3. **`fillvalue=0` is correct for "fixed" boundary.**  Since the mask is
   binary (0 or 1), outer cells with `fillvalue=0` are treated as *not*
   matching the target state, which is the correct semantic: cells outside a
   fixed-boundary grid do not contribute to any neighbour count.

4. **Return dtype is `int`** (platform-native integer), obtained by casting the
   `convolve2d` output.  The caller can safely compare against integer
   thresholds from the rule conditions.

5. **No automaton-specific logic** — the function takes raw integers
   (`state_idx`, not a state name string), so it is completely generic.

### Verification

Smoke-tested all three neighbourhood / boundary combinations:

| Grid | Neighbourhood | Boundary | Result | Correct |
|------|---------------|----------|--------|---------|
| 3×3 cross of 1s | Moore | Toroidal | Corner cells see 5 neighbours | ✅ |
| 3×3 cross of 1s | Von Neumann | Fixed | Centre cell sees 4 neighbours | ✅ |
| 1×5 `[1,0,1,1,0]` | Left-center-right | Toroidal | `[0,2,1,1,2]` | ✅ |

All 29 existing tests continue to pass after adding the new file.

### Files Created

| File | Purpose |
|------|---------|
| `cellab/engine.py` | Simulation engine — `count_neighbors` helper function |
| `ai_logs/engine_count_neighbors_log.md` | This log file |
