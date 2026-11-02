"""Run and resume reproducible cellular-automaton experiments."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from cellab.model import Automaton
from cellab.storage import (
    History,
    StoredExperiment,
    load_experiment,
    save_experiment,
    validate_world,
)

StepFunction = Callable[
    [np.ndarray, Automaton, np.random.Generator], np.ndarray
]
StepCallback = Callable[[int, np.ndarray], None]


class ExperimentConfig(BaseModel):
    """Validated, self-contained configuration for a saved experiment."""

    model_config = ConfigDict(extra="forbid")

    automaton: Automaton
    initialization: dict[str, Any] = Field(
        default_factory=lambda: {"mode": "provided"}
    )
    generations: StrictInt = Field(ge=0)
    seed: StrictInt = Field(ge=0)
    store_history: StrictBool
    completed_generations: StrictInt = Field(default=0, ge=0)
    rng_state: dict[str, Any] = Field(default_factory=dict)

    @field_validator("initialization")
    @classmethod
    def _validate_initialization(cls, value: dict[str, Any]) -> dict[str, Any]:
        mode = value.get("mode")
        allowed_modes = {
            "random",
            "template",
            "text",
            "file",
            "provided",
        }
        if mode not in allowed_modes:
            raise ValueError(
                "initialization.mode must be random, template, text, file, "
                "or provided."
            )
        return value


def run_experiment(
    automaton: Automaton,
    initial_world: np.ndarray,
    *,
    generations: int,
    step: StepFunction,
    output_directory: str | Path,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    store_history: bool = False,
    initialization: dict[str, Any] | None = None,
    on_step: StepCallback | None = None,
) -> StoredExperiment:
    """Run a new experiment and save a checkpoint ready to resume."""
    _validate_generation_count(generations)
    world = np.asarray(initial_world).copy()
    validate_world(world, automaton)
    actual_seed = secrets.randbits(64) if seed is None else seed
    if isinstance(actual_seed, bool) or not isinstance(actual_seed, int):
        raise ValueError("seed must be a non-negative integer.")
    if actual_seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    if rng is not None and seed is None:
        raise ValueError("seed must be provided when supplying an RNG.")
    if rng is None:
        rng = np.random.default_rng(actual_seed)
    initialization_config = (
        {"mode": "provided"} if initialization is None else initialization
    )
    ExperimentConfig(
        automaton=automaton,
        initialization=initialization_config,
        generations=generations,
        seed=actual_seed,
        store_history=store_history,
        completed_generations=0,
        rng_state=rng.bit_generator.state,
    )
    history = History(automaton, store_history)
    history.record(0, world)
    world = _advance(automaton, world, rng, step, generations, 0, history, on_step)
    save_experiment(
        output_directory,
        automaton=automaton,
        initial_world=np.asarray(initial_world),
        world=world,
        rng=rng,
        seed=actual_seed,
        completed_generations=generations,
        store_history=store_history,
        history=history.worlds,
        summaries=history.summaries,
        initialization=initialization_config,
    )
    return load_experiment(output_directory)


def resume_experiment(
    directory: str | Path,
    *,
    additional_generations: int,
    step: StepFunction,
    on_step: StepCallback | None = None,
) -> StoredExperiment:
    """Continue a saved run using its saved random-generator state."""
    _validate_generation_count(additional_generations)
    stored = load_experiment(directory)
    history = History(
        stored.automaton,
        stored.store_history,
        stored.history,
        stored.summaries,
    )
    stored.world = _advance(
        stored.automaton,
        stored.world,
        stored.rng,
        step,
        additional_generations,
        stored.completed_generations,
        history,
        on_step,
    )
    stored.history = history.worlds
    stored.summaries = history.summaries
    stored.completed_generations += additional_generations
    save_experiment(
        directory,
        automaton=stored.automaton,
        initial_world=stored.initial_world,
        world=stored.world,
        rng=stored.rng,
        seed=stored.seed,
        completed_generations=stored.completed_generations,
        store_history=stored.store_history,
        history=history.worlds,
        summaries=history.summaries,
        initialization=stored.initialization,
    )
    return stored


def _advance(
    automaton: Automaton,
    world: np.ndarray,
    rng: np.random.Generator,
    step: StepFunction,
    count: int,
    completed_generations: int,
    history: History,
    on_step: StepCallback | None = None,
) -> np.ndarray:
    for offset in range(1, count + 1):
        next_world = np.asarray(step(world.copy(), automaton, rng))
        validate_world(next_world, automaton)
        if next_world.shape != world.shape:
            raise ValueError(
                f"Step changed world shape from {world.shape} to "
                f"{next_world.shape}."
            )
        world = next_world.copy()
        history.record(completed_generations + offset, world)
        if on_step is not None:
            on_step(completed_generations + offset, world.copy())
    return world


def _validate_generation_count(generations: int) -> None:
    if (
        isinstance(generations, bool)
        or not isinstance(generations, int)
        or generations < 0
    ):
        raise ValueError("generations must be a non-negative integer.")
