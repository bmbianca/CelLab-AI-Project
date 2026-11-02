# Member 3 - Experiment, storage, and CLI

Implemented the experiment configuration and run/resume workflow:

- The experiment runner accepts an automaton, an integer-indexed 2D world,
  generation count, optional seed, history-storage setting, and a step
  function.
- Saved experiment folders contain `automaton.json`, `config.json`, and
  compressed `history.npz`. The checkpoint stores the initial/current worlds
  and NumPy generator state; optionally it also stores every generation.
- When full history is disabled, per-generation SHA-256 hashes and counts for
  each declared state remain in `config.json`.
- The CLI provides `run` and `resume` commands and expects the engine contract
  `cellab.engine.step(world, automaton, rng)`.
- Tests verify that 100 uninterrupted stochastic generations match 50
  generations followed by save, reload, and 50 more.

Validation: `py -3.12 -m pytest -q` passed (7 tests).

DEMO ONLY: using a placeholder RNG step, not Member 1's CA engine.
Completed generations: 50
Completed generations: 100
PASS: 50 + save/reload + 50 matches an uninterrupted 100-generation run.

## Follow-up — 2026-11-02

### User request

Complete only Member 3's work, following the full Deliverable 2 task document;
do not implement the simulation engine or change the responsibilities of the
other members.

### Changes completed

- Added a validated `ExperimentConfig` that retains the complete automaton,
  initialization parameters, generation count, seed, storage choice, and RNG
  state.
- Added `History` recording: BLAKE2b hash and per-state counts for every
  generation, with world snapshots kept only when history storage is enabled.
- Saved the initialization/RNG configuration in `config.json` and snapshots,
  summaries, and checkpoints in compressed `history.npz`; retained loading
  support for the previous format.
- Completed the CLI commands `list`, `show`, `run`, `resume`, and `info`, with
  text/template/random/file initialization, periodic display, and animation.
- Added `python -m cellab`, setup/usage documentation, and the Pydantic v2
  dependency constraint.
- Kept the CA engine external. The Member 3 demo uses a labeled placeholder
  step and is not presented as a test of automaton rules.

### Validation

- `py -3.12 -m pytest -q`: 49 tests passed.
- The save/resume/info demo passed: 50 generations + save/reload + 50 matched
  an uninterrupted 100-generation run.
- CLI `list` and `show` were exercised. `run`/`resume` continue to require
  `cellab.engine.step`, which is outside Member 3's assigned scope.
