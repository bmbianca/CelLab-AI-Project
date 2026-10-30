# AI Log — `model.py` Generation

**Date**: 2026-10-30T13:32:10+02:00  
**Model**: Claude Opus 4.6 (Thinking)  
**Author**: Antigravity (AI Coding Assistant)

---

## Prompt Summary

The user requested the implementation of `cellab/model.py` — the foundational
data-model layer for the CelLab Cellular Automata project, based on the team's
**JSON v2 format specification**.

### Key Requirements Given

| # | Requirement | Status |
|---|-------------|--------|
| 1 | **Zero Hardcoding** — No specific automaton names or state strings in code | ✅ Implemented |
| 2 | Create `Automaton`, `Rule`, and `Condition` Pydantic classes | ✅ Implemented |
| 3 | `Condition.op` supports `==`, `!=`, `<`, `<=`, `>`, `>=`, `in` (with list) | ✅ Implemented |
| 4 | `if` field can be a single `Condition` or a list (all must be met) | ✅ Implemented |
| 5 | States must be unique | ✅ Validated |
| 6 | `default_state` must exist within `states` | ✅ Validated |
| 7 | All states referenced in rules/conditions must be declared in `states` | ✅ Validated |
| 8 | `probability` ∈ [0, 1] | ✅ Validated via `ge=0.0, le=1.0` |
| 9 | `neighborhood` ∈ `{moore, von neumann, left-center-right}` | ✅ `Literal` type |
| 10 | `boundary` ∈ `{toroidal, fixed}` | ✅ `Literal` type |
| 11 | 1-D `pattern` condition (list of 3 states) for `left-center-right` | ✅ Implemented |
| 12 | `max_neighbors(neighborhood)` → 8 / 4 / 2, used in validation | ✅ Implemented |
| 13 | Condition values must not exceed `max_neighbors` | ✅ Validated |
| 14 | Expose `load_automaton_file(path) -> Automaton` | ✅ Implemented |

### Example JSON Used for Reference

```json
{
  "name": "Forest Fire",
  "states": ["EMPTY", "TREE", "FIRE"],
  "default_state": "EMPTY",
  "neighborhood": "moore",
  "boundary": "toroidal",
  "rules": [
    {"from": "FIRE", "to": "EMPTY"},
    {"from": "TREE", "to": "FIRE", "if": {"count": "FIRE", "op": ">=", "value": 1}},
    {"from": "TREE", "to": "FIRE", "probability": 0.001}
  ]
}
```

### Design Decisions

1. **Pydantic V2 API**: Used `model_validator`, `field_validator`, `model_validate`,
   and `Field(alias=...)` from Pydantic v2.
2. **Aliases**: `from` → `from_state`, `if` → `if_condition`, `to` → `to_state`
   (since `from` and `if` are Python reserved words). `populate_by_name=True`
   allows access via both the alias and the Python-safe name.
3. **No simulation logic**: The file contains only data definitions, validations,
   and the JSON loader. The engine and NumPy convolutions are deferred to
   separate modules.
4. **Cross-field validation** is consolidated in a single `model_validator` on
   `Automaton` for clarity and to ensure all fields are available.

### Files Created

| File | Purpose |
|------|---------|
| `cellab/__init__.py` | Package initialiser |
| `cellab/model.py` | Pydantic data model (Automaton, Rule, Condition) |
| `ai_logs/model_prompt_log.md` | This log file |
