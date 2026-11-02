# CelLab - Cellular Automata Laboratory

## Installation

Use Python 3.12 from the repository root:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m pytest -q
```

## Command-line interface

Run commands from the repository root with `python -m cellab` (or
`py -3.12 -m cellab` on Windows):

```powershell
py -3.12 -m cellab list
py -3.12 -m cellab show forest_fire
```

Start a random Forest Fire experiment. The seed is printed and saved, and the
same `numpy.random.Generator` is used for random initial-world generation and
the simulation. Omit `--seed` to generate, print, and save a new seed:

```powershell
py -3.12 -m cellab run forest_fire `
  --size 30x60 `
  --init random `
  --density TREE=0.6,FIRE=0.01 `
  --seed 42 `
  --generations 100 `
  --store `
  --show every:10 `
  --save runs\forest_fire
```

The initial world can instead be selected from a template or read from a text
file. Text files use the symbols defined by the automaton, one row per line:

```powershell
py -3.12 -m cellab run game_of_life `
  --init template --template glider --size 30x60 `
  --generations 100 --seed 42 --save runs\life

py -3.12 -m cellab run forest_fire `
  --init text --world initial_world.txt `
  --generations 100 --seed 42 --save runs\forest_fire_text
```

NumPy `.npy` initial worlds are also accepted with `--init file --world
initial_world.npy`. A custom automaton can be selected with `--file`:

```powershell
py -3.12 -m cellab run --file automaton.json `
  --init random --size 30x60 --generations 100 --save runs\custom
```

Resume an experiment, inspect its counts and repeated hashes, or display the
rules for a library automaton:

```powershell
py -3.12 -m cellab resume runs\forest_fire --generations 50 --show every:10
py -3.12 -m cellab info runs\forest_fire
```

Use `--animate` to redraw two-dimensional worlds in the terminal. One-row
worlds are displayed one generation per line as a space-time diagram.

## Saved experiment format

Each saved experiment directory contains:

- `automaton.json` — the complete automaton definition, independent of the
  library after saving.
- `config.json` — initialization parameters, seed, current generation,
  history-storage setting, and NumPy bit-generator state.
- `history.npz` — compressed initial/current worlds, per-generation BLAKE2b
  hashes and state counts, and (when `--store` is enabled) every generation.

The seed and generator state are restored by `resume`; a stochastic run thus
continues reproducibly after saving and loading. With history storage disabled,
the checkpoint and per-generation summaries remain available, but intermediate
world arrays are not stored.

## Member 3 workflow check

The self-check exercises the CLI's save/resume/info workflow and verifies that
50 generations, save/reload, and 50 more match an uninterrupted 100-generation
run:

```powershell
py -3.12 -m cellab.member3_demo experiments\member3_demo
```

It uses a clearly labeled placeholder step, so it checks Member 3's workflow,
not cellular-automaton rules.

## Simulation engine integration

The `run` and `resume` commands use `cellab.engine.step(world, automaton, rng)`,
which is the simulation-engine contract owned separately from Member 3's
experiment, storage, and CLI code. Until that module is present, `list`, `show`,
and `info` work, while simulation commands report that the engine is
unavailable. The experiment runner also accepts a compatible step function
directly for isolated storage and recovery tests.
