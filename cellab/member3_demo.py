"""Runnable demonstration of Member 3's run, storage, and resume workflow.

The included step function is a deterministic-seed random-cell demo only; it
is a stand-in for the real cellular-automaton engine owned by Member 1.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path
from typing import Sequence

import numpy as np

from cellab import cli
from cellab.experiment import run_experiment
from cellab.model import Automaton


SEED = 2026
SPLIT_GENERATIONS = 50


def _demo_step(
    world: np.ndarray,
    automaton: Automaton,
    rng: np.random.Generator,
) -> np.ndarray:
    flips = rng.random(world.shape) < 0.25
    return np.where(flips, (world + 1) % len(automaton.states), world)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    requested_directory = (
        Path(args[0]) if args else Path("experiments") / "member3_demo"
    )
    output_directory = _available_directory(requested_directory)
    if output_directory != requested_directory:
        print(
            f"{requested_directory} already exists; using "
            f"{output_directory} instead."
        )

    automaton = Automaton.model_validate(
        {
            "name": "Member 3 persistence demo",
            "states": ["A", "B"],
            "default_state": "A",
            "neighborhood": "moore",
            "boundary": "toroidal",
            "rules": [],
        }
    )
    initial_world = np.zeros((8, 12), dtype=np.int64)
    initial_world[2:6, 3:9] = 1
    module = types.ModuleType("cellab.engine")
    module.step = _demo_step
    previous_engine = sys.modules.get("cellab.engine")
    sys.modules["cellab.engine"] = module

    try:
        with tempfile.TemporaryDirectory(prefix="cellab-member3-") as temp_dir:
            input_dir = Path(temp_dir)
            automaton_file = input_dir / "automaton.json"
            world_file = input_dir / "world.npy"
            automaton_file.write_text(
                json.dumps(
                    automaton.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
            np.save(world_file, initial_world)

            print("DEMO ONLY: using a placeholder RNG step, not Member 1's CA engine.")
            print("Running 50 generations and saving the experiment...")
            run_status = cli.main(
                [
                    "run",
                    "--automaton",
                    str(automaton_file),
                    "--world",
                    str(world_file),
                    "--generations",
                    str(SPLIT_GENERATIONS),
                    "--seed",
                    str(SEED),
                    "--store-history",
                    "--output",
                    str(output_directory),
                ]
            )
            if run_status != 0:
                raise RuntimeError("The CLI failed to start the demonstration run.")

            print("Reloading the checkpoint and resuming for 50 more generations...")
            resume_status = cli.main(
                [
                    "resume",
                    str(output_directory),
                    "--generations",
                    str(SPLIT_GENERATIONS),
                ]
            )
            if resume_status != 0:
                raise RuntimeError("The CLI failed to resume the demonstration run.")
            print("Inspecting the saved experiment...")
            info_status = cli.main(["info", str(output_directory)])
            if info_status != 0:
                raise RuntimeError("The CLI failed to inspect the demonstration run.")

            full_run = run_experiment(
                automaton,
                initial_world,
                generations=2 * SPLIT_GENERATIONS,
                step=_demo_step,
                output_directory=input_dir / "uninterrupted",
                seed=SEED,
                store_history=False,
            )

        from cellab.storage import load_experiment

        resumed = load_experiment(output_directory)
        if not np.array_equal(resumed.world, full_run.world):
            raise RuntimeError("Resume verification failed: final worlds differ.")
        if resumed.summaries != full_run.summaries:
            raise RuntimeError("Resume verification failed: summaries differ.")

        print("PASS: 50 + save/reload + 50 matches an uninterrupted 100-generation run.")
        print(f"Saved files: {output_directory / 'automaton.json'}")
        print(f"             {output_directory / 'config.json'}")
        print(f"             {output_directory / 'history.npz'}")
        print(f"Final BLAKE2b: {resumed.summaries[-1]['hash']}")
        return 0
    finally:
        if previous_engine is None:
            sys.modules.pop("cellab.engine", None)
        else:
            sys.modules["cellab.engine"] = previous_engine


def _available_directory(requested: Path) -> Path:
    if not requested.exists():
        return requested
    suffix = 2
    while True:
        candidate = requested.with_name(f"{requested.name}_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


if __name__ == "__main__":
    raise SystemExit(main())
