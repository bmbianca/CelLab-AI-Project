"""Persistence helpers for resumable cellular-automaton experiments."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cellab.model import Automaton


@dataclass
class History:
    """Per-generation summaries and, optionally, complete world snapshots."""

    automaton: Automaton
    store_worlds: bool
    worlds: list[np.ndarray] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, generation: int, world: np.ndarray) -> None:
        expected_generation = len(self.summaries)
        if generation != expected_generation:
            raise ValueError(
                f"Expected generation {expected_generation}, got {generation}."
            )
        if self.store_worlds:
            self.worlds.append(world.copy())
        self.summaries.append(
            summarize_world(world, self.automaton, generation=generation)
        )


@dataclass
class StoredExperiment:
    automaton: Automaton
    initial_world: np.ndarray
    world: np.ndarray
    rng: np.random.Generator
    seed: int
    completed_generations: int
    store_history: bool
    history: list[np.ndarray]
    summaries: list[dict[str, Any]]
    initialization: dict[str, Any]


def summarize_world(
    world: np.ndarray,
    automaton: Automaton,
    generation: int,
) -> dict[str, Any]:
    """Return a reproducible hash and one cell count per declared state."""
    digest = hashlib.blake2b(np.ascontiguousarray(world).tobytes()).hexdigest()

    counts = np.bincount(world.ravel(), minlength=len(automaton.states))
    return {
        "generation": generation,
        "hash": digest,
        "state_counts": {
            state: int(counts[index])
            for index, state in enumerate(automaton.states)
        },
    }


def save_experiment(
    directory: str | Path,
    *,
    automaton: Automaton,
    initial_world: np.ndarray,
    world: np.ndarray,
    rng: np.random.Generator,
    seed: int,
    completed_generations: int,
    store_history: bool,
    history: list[np.ndarray],
    summaries: list[dict[str, Any]],
    initialization: dict[str, Any] | None = None,
) -> None:
    """Write the automaton, resume configuration, and compressed checkpoint."""
    path = Path(directory)
    validate_world(initial_world, automaton)
    validate_world(world, automaton)
    if initial_world.shape != world.shape:
        raise ValueError("Initial and current worlds must have the same shape.")
    if len(summaries) != completed_generations + 1:
        raise ValueError(
            "There must be one summary for each generation, including generation 0."
        )
    if [item.get("generation") for item in summaries] != list(
        range(completed_generations + 1)
    ):
        raise ValueError("Summary generation indexes must be consecutive.")
    if store_history:
        if len(history) != completed_generations + 1:
            raise ValueError(
                "Stored history must include the initial world and every generation."
            )
        for generation in history:
            validate_world(generation, automaton)
            if generation.shape != initial_world.shape:
                raise ValueError("All stored generations must have the same shape.")
        if not np.array_equal(history[0], initial_world):
            raise ValueError("Stored generation 0 must match the initial world.")
        if not np.array_equal(history[-1], world):
            raise ValueError("The last stored generation must match the checkpoint.")
    elif history:
        raise ValueError("World snapshots must be omitted when history storage is off.")

    from cellab.experiment import ExperimentConfig

    initialization_config = (
        {"mode": "provided"} if initialization is None else initialization
    )
    config_model = ExperimentConfig(
        automaton=automaton,
        initialization=initialization_config,
        generations=completed_generations,
        seed=seed,
        store_history=store_history,
        completed_generations=completed_generations,
        rng_state=_to_json_value(rng.bit_generator.state),
    )
    config = config_model.model_dump(mode="json", exclude={"automaton"})
    config["format_version"] = 2
    path.mkdir(parents=True, exist_ok=True)
    history_file = path / "history.npz"
    temporary_history = path / "history.npz.tmp"
    state_counts = np.asarray(
        [
            [summary["state_counts"][state] for state in automaton.states]
            for summary in summaries
        ],
        dtype=np.int64,
    )
    hashes = np.asarray([summary["hash"] for summary in summaries], dtype="U128")
    with temporary_history.open("wb") as file:
        arrays: dict[str, np.ndarray] = {
            "initial_world": initial_world,
            "current_world": world,
            "summary_generations": np.arange(
                completed_generations + 1, dtype=np.int64
            ),
            "hashes": hashes,
            "state_counts": state_counts,
        }
        if store_history:
            arrays["generations"] = np.stack(history)
        np.savez_compressed(file, **arrays)
    os.replace(temporary_history, history_file)

    _write_json(path / "automaton.json", automaton.model_dump(
        mode="json", by_alias=True, exclude_none=True
    ))
    _write_json(path / "config.json", config)


def load_experiment(directory: str | Path) -> StoredExperiment:
    """Load a saved experiment and restore its NumPy random-generator state."""
    path = Path(directory)
    with (path / "config.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    if config.get("format_version") not in (1, 2):
        raise ValueError(
            f"Unsupported experiment format version: "
            f"{config.get('format_version')!r}"
        )

    with (path / "automaton.json").open("r", encoding="utf-8") as file:
        automaton = Automaton.model_validate(json.load(file))

    with np.load(path / "history.npz", allow_pickle=False) as archive:
        initial_world = archive["initial_world"].copy()
        world = archive["current_world"].copy()
        stored_history = (
            archive["generations"].copy()
            if "generations" in archive.files
            else None
        )
        summary_generations = (
            archive["summary_generations"].copy()
            if "summary_generations" in archive.files
            else None
        )
        hashes = archive["hashes"].copy() if "hashes" in archive.files else None
        state_counts = (
            archive["state_counts"].copy()
            if "state_counts" in archive.files
            else None
        )
        summary_arrays_present = [
            "summary_generations" in archive.files,
            "hashes" in archive.files,
            "state_counts" in archive.files,
        ]

    validate_world(initial_world, automaton)
    validate_world(world, automaton)
    if initial_world.shape != world.shape:
        raise ValueError("Saved initial and current worlds have different shapes.")
    from cellab.experiment import ExperimentConfig

    config_model = ExperimentConfig.model_validate(
        {
            key: value
            for key, value in {
                **config,
                "automaton": automaton,
                "initialization": config.get(
                    "initialization", {"mode": "provided"}
                ),
            }.items()
            if key
            in {
                "automaton",
                "initialization",
                "generations",
                "seed",
                "store_history",
                "completed_generations",
                "rng_state",
            }
        }
    )
    if any(summary_arrays_present) and not all(summary_arrays_present):
        raise ValueError("Saved experiment has incomplete summary arrays.")
    if config.get("format_version") == 2:
        if not all(summary_arrays_present):
            raise ValueError("Saved experiment is missing its summary arrays.")
    if (
        summary_generations is not None
        and hashes is not None
        and state_counts is not None
    ):
        expected_generations = config_model.completed_generations + 1
        if (
            summary_generations.shape != (expected_generations,)
            or hashes.shape != (expected_generations,)
            or state_counts.shape != (expected_generations, len(automaton.states))
            or not np.array_equal(
                summary_generations, np.arange(expected_generations)
            )
        ):
            raise ValueError("Saved experiment summaries have inconsistent dimensions.")
        summaries = [
            {
                "generation": int(summary_generations[index]),
                "hash": str(hashes[index]),
                "state_counts": {
                    state: int(state_counts[index, state_index])
                    for state_index, state in enumerate(automaton.states)
                },
            }
            for index in range(expected_generations)
        ]
        if np.any(state_counts < 0):
            raise ValueError("Saved state counts cannot be negative.")
        if np.any(state_counts.sum(axis=1) != world.size):
            raise ValueError("Saved state counts do not match the world size.")
        if any(len(summary["hash"]) != 128 for summary in summaries):
            raise ValueError("Saved BLAKE2b hashes must contain 128 characters.")
    else:
        summaries = []
        for item in config.get("summaries", []):
            summary = dict(item)
            if "hash" not in summary and "sha256" in summary:
                summary["hash"] = summary.pop("sha256")
            summaries.append(summary)
    if len(summaries) != config_model.completed_generations + 1:
        raise ValueError("Saved experiment does not contain a complete summary history.")
    if config_model.store_history and stored_history is None:
        raise ValueError("Configuration requires full history, but none is stored.")
    if not config_model.store_history and stored_history is not None:
        raise ValueError("History snapshots exist although storage is disabled.")
    if stored_history is not None and stored_history.shape[0] != (
        config_model.completed_generations + 1
    ):
        raise ValueError("Saved generation history has an inconsistent length.")
    if stored_history is not None:
        if stored_history.ndim != 3 or stored_history.shape[1:] != world.shape:
            raise ValueError("Saved world snapshots have inconsistent dimensions.")
        for generation in stored_history:
            validate_world(generation, automaton)
        if not np.array_equal(stored_history[0], initial_world):
            raise ValueError("Saved generation 0 does not match the initial world.")
        if not np.array_equal(stored_history[-1], world):
            raise ValueError("The last saved generation does not match the checkpoint.")

    rng = np.random.default_rng(config_model.seed)
    rng.bit_generator.state = config_model.rng_state
    history = (
        [generation.copy() for generation in stored_history]
        if stored_history is not None
        else []
    )
    return StoredExperiment(
        automaton=automaton,
        initial_world=initial_world,
        world=world,
        rng=rng,
        seed=config_model.seed,
        completed_generations=config_model.completed_generations,
        store_history=config_model.store_history,
        history=history,
        summaries=summaries,
        initialization=config_model.initialization,
    )


def validate_world(world: np.ndarray, automaton: Automaton) -> None:
    if world.ndim != 2 or any(size == 0 for size in world.shape):
        raise ValueError("A world must be a non-empty, two-dimensional array.")
    if not np.issubdtype(world.dtype, np.integer):
        raise ValueError("World cells must be integer state indexes.")
    if np.any(world < 0) or np.any(world >= len(automaton.states)):
        raise ValueError("World contains a state index not declared by automaton.")


def _write_json(path: Path, data: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary_path, path)


def _to_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
