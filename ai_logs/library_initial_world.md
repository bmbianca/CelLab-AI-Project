# AI Log — Library, Initial World & Offline Parser Generation

**Date**: 2026-10-31
**Model**: Claude Opus 4.8 (Claude Code)

---

## Prompt Summary

The user requested step-by-step guidance to implement **Member 2's part of
Deliverable 2** on top of Member 1's existing data-model layer
(`cellab/model.py`): the library of predefined automata, the initial-world
generation, and the verification of the offline NLP parser.

### Key Requirements Given

| # | Requirement | Status |
|---|-------------|--------|
| 1 | **Zero Hardcoding** — no automaton or state name anywhere in code | ✅ Verified |
| 2 | 3 mandatory automata in JSON v2 format (no `dimensions` field) | ✅ Implemented |
| 3 | Each automaton carries `description`, `symbols`, `templates`, `default_world` | ✅ Implemented |
| 4 | `list_automata()` / `load_automaton(name)` reading from `library/` | ✅ Implemented |
| 5 | `random_world` with per-state density and a seed | ✅ Implemented |
| 6 | Template-based world (patterns stored in the automaton's JSON, not in code) | ✅ Implemented |
| 7 | `from_text` / `to_text` round-trip for the `..F..` world format | ✅ Implemented |
| 8 | No global randomness — only a seeded `np.random.Generator` | ✅ Verified |
| 9 | Offline NLP parser = local LLM via Ollama (feedback "just use local") | ✅ Verified |

### Example JSON Used for Reference (Forest Fire)

```json
{
  "name": "Forest Fire",
  "states": ["EMPTY", "TREE", "FIRE"],
  "default_state": "EMPTY",
  "neighborhood": "moore",
  "boundary": "toroidal",
  "rules": [
    {"from": "FIRE",  "to": "EMPTY"},
    {"from": "TREE",  "to": "FIRE", "if": {"count": "FIRE", "op": ">=", "value": 1}},
    {"from": "TREE",  "to": "FIRE", "probability": 0.001},
    {"from": "EMPTY", "to": "TREE", "probability": 0.05}
  ],
  "symbols": {"EMPTY": ".", "TREE": "T", "FIRE": "F"}
}
```

### Design Decisions

1. **Nothing specific to any automaton lives in code.** Shape templates
   (glider, blinker) are stored in each automaton's JSON; a dedicated test runs
   an invented automaton with states `A`, `B`, `C` to prove the code is generic.
2. **Reproducible randomness.** `random_world` draws from a seeded
   `np.random.Generator`; the same seed yields the same world. The global NumPy
   random state is never used.
3. **World is always 2-D** (`np.int8`, `shape = (H, W)`); a 1-D automaton is a
   world with a single row.
4. **Offline parser via Ollama**, not a spaCy/regex grammar, per the teacher's
   feedback ("just use local").
5. **Minimal change to Member 1's file.** `model.py` only gained optional
   fields (`symbols`, `templates`, `default_world`) plus their validation; the
   existing model and loader were left untouched.

### Offline Parser Verification (Ollama)

Installed Ollama locally and pulled `qwen2.5:7b`. Command:

```
ollama run qwen2.5:7b "Return only JSON: a cellular automaton with states EMPTY, TREE, FIRE"
```

The model returned valid JSON listing the three states, confirming that the
offline (no-internet, no-cost) parser path works.

### Files Created

| File | Purpose |
|------|---------|
| `library/game_of_life.json` | Game of Life — deterministic, 2 states |
| `library/brians_brain.json` | Brian's Brain — 3 states |
| `library/forest_fire.json` | Forest Fire — stochastic (probabilities + seed) |
| `cellab/library.py` | Library loader: `list_automata`, `load_automaton` |
| `cellab/world.py` | Initial-world generation: random / template / text |
| `cellab/model.py` | Added optional `symbols`, `templates`, `default_world` fields |
| `tests/test_library.py` | Tests for the library loader and the extra fields |
| `tests/test_world.py` | Tests for world generation (incl. an invented automaton) |
| `library_initial_world.md` | This log file |
