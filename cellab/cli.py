"""Command-line interface for running and inspecting experiments."""

from __future__ import annotations

import argparse
import importlib
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cellab.experiment import StepCallback, StepFunction, resume_experiment, run_experiment
from cellab.library import list_automata, load_automaton
from cellab.model import Automaton, Condition, Rule, load_automaton_file
from cellab.storage import load_experiment, validate_world
from cellab.world import (
    from_template,
    from_text,
    parse_density,
    parse_shape,
    random_world,
    to_text,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "list":
            return _list_automata()
        if args.command == "show":
            return _show_automaton(args.automaton)
        if args.command == "info":
            return _show_experiment_info(args.experiment)
        if args.command == "run":
            return _run(args)
        if args.command == "resume":
            return _resume(args)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        parser.error(str(error))
    return 0


def _run(args: argparse.Namespace) -> int:
    automaton = _resolve_automaton(args.automaton_name, args.file)
    seed = secrets.randbits(64) if args.seed is None else args.seed
    if seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    rng = np.random.default_rng(seed)
    world, initialization = _initial_world(args, automaton, rng)
    validate_world(world, automaton)
    show_every = _show_interval(args.show)
    callback = _display_callback(automaton, show_every, args.animate)
    step = _load_step()
    if callback is not None:
        _display_generation(0, world, automaton, animate=args.animate)

    result = run_experiment(
        automaton,
        world,
        generations=args.generations,
        seed=seed,
        rng=rng,
        store_history=args.store,
        output_directory=args.save,
        step=step,
        initialization=initialization,
        on_step=callback,
    )
    print(f"Seed: {result.seed}")
    print(
        f"Completed generations: {result.completed_generations}; "
        f"world shape: {result.world.shape}"
    )
    print(f"Experiment saved to: {Path(args.save)}")
    return 0


def _resume(args: argparse.Namespace) -> int:
    stored = load_experiment(args.experiment)
    show_every = _show_interval(args.show)
    callback = _display_callback(stored.automaton, show_every, args.animate)
    if args.animate or (
        show_every is not None
        and stored.completed_generations % show_every == 0
    ):
        _display_generation(
            stored.completed_generations,
            stored.world,
            stored.automaton,
            animate=args.animate,
        )
    step = _load_step()
    result = resume_experiment(
        args.experiment,
        additional_generations=args.generations,
        step=step,
        on_step=callback,
    )
    print(
        f"Completed generations: {result.completed_generations}; "
        f"world shape: {result.world.shape}"
    )
    print(f"Experiment saved to: {Path(args.experiment)}")
    return 0


def _load_step() -> StepFunction:
    try:
        engine = importlib.import_module("cellab.engine")
    except ModuleNotFoundError as error:
        if error.name == "cellab.engine":
            raise RuntimeError(
                "The simulation engine is not available yet. "
                "It must provide cellab.engine.step(world, automaton, rng)."
            ) from error
        raise
    step = getattr(engine, "step", None)
    if not callable(step):
        raise RuntimeError(
            "cellab.engine must provide step(world, automaton, rng)."
        )
    return step


def _resolve_automaton(name: str | None, file_path: str | None) -> Automaton:
    if name and file_path:
        raise ValueError("Choose either an automaton name or --file, not both.")
    source = file_path or name
    if not source:
        raise ValueError("Provide an automaton name or a JSON file with --file.")
    candidate = Path(source)
    if candidate.is_file() or candidate.suffix.lower() == ".json":
        return load_automaton_file(candidate)
    return load_automaton(source)


def _initial_world(
    args: argparse.Namespace,
    automaton: Automaton,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = automaton.default_world or {}
    mode = args.init
    if mode is None:
        if args.world:
            mode = "file" if Path(args.world).suffix.lower() == ".npy" else "text"
        else:
            mode = spec.get("init", "random")
    elif mode == "default":
        mode = spec.get("init", "random")

    shape = _parse_size(args.size) if args.size else _default_shape(spec)
    if mode == "random":
        density = (
            parse_density(args.density)
            if args.density is not None
            else spec.get("density", {})
        )
        return random_world(shape, automaton, density, rng), {
            "mode": "random",
            "size": list(shape),
            "density": density,
        }
    if mode == "template":
        template = args.template or spec.get("template")
        if not template:
            raise ValueError(
                "Template initialization requires --template or a default "
                "template in the automaton JSON."
            )
        return from_template(template, shape, automaton), {
            "mode": "template",
            "size": list(shape),
            "template": template,
        }
    if mode == "text":
        if not args.world:
            raise ValueError("Text initialization requires --world with a text file.")
        world = from_text(args.world, automaton)
        return world, {
            "mode": "text",
            "size": list(world.shape),
            "source": str(args.world),
        }
    if mode == "file":
        if not args.world:
            raise ValueError("File initialization requires --world with a .npy file.")
        if Path(args.world).suffix.lower() != ".npy":
            raise ValueError("Array-file initialization requires a .npy file.")
        world = np.load(args.world, allow_pickle=False)
        return world, {"mode": "file", "source": str(args.world)}
    raise ValueError(f"Unknown initialization mode: {mode!r}.")


def _parse_size(value: str) -> tuple[int, int]:
    try:
        return parse_shape(value)
    except ValueError as error:
        raise ValueError(f"Invalid --size: {error}") from error


def _non_negative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _default_shape(spec: dict[str, Any]) -> tuple[int, int]:
    size = spec.get("size", (20, 20))
    if isinstance(size, str):
        return parse_shape(size)
    return parse_shape("x".join(str(part) for part in size))


def _show_interval(value: str | None) -> int | None:
    if value is None:
        return None
    prefix, separator, raw_interval = value.partition(":")
    if prefix != "every" or not separator:
        raise ValueError("--show must use the format every:N, for example every:10.")
    try:
        interval = int(raw_interval)
    except ValueError as error:
        raise ValueError("--show interval must be a positive integer.") from error
    if interval < 1:
        raise ValueError("--show interval must be a positive integer.")
    return interval


def _display_callback(
    automaton: Automaton,
    show_every: int | None,
    animate: bool,
) -> StepCallback | None:
    if show_every is None and not animate:
        return None

    def display(generation: int, world: np.ndarray) -> None:
        if animate or generation % show_every == 0:
            _display_generation(generation, world, automaton, animate=animate)

    return display


def _display_generation(
    generation: int,
    world: np.ndarray,
    automaton: Automaton,
    *,
    animate: bool,
) -> None:
    rendered = to_text(world, automaton)
    if world.shape[0] == 1:
        print(f"Generation {generation}: {rendered}")
        return
    if animate:
        sys.stdout.write("\x1b[2J\x1b[H")
    print(f"Generation {generation}:\n{rendered}")
    if animate:
        sys.stdout.flush()
        time.sleep(0.08)


def _list_automata() -> int:
    for name in list_automata():
        print(name)
    return 0


def _show_automaton(source: str) -> int:
    automaton = _resolve_automaton(source, None)
    print(automaton.name)
    if automaton.description:
        print(automaton.description)
    for rule in automaton.rules:
        print(_describe_rule(rule))
    return 0


def _describe_rule(rule: Rule) -> str:
    result = f"A cell in state {rule.from_state} becomes {rule.to_state}"
    conditions = rule.if_condition
    if conditions is not None:
        values = conditions if isinstance(conditions, list) else [conditions]
        result += " if " + " and ".join(_describe_condition(item) for item in values)
    if rule.pattern is not None:
        result += f" for neighborhood pattern {', '.join(rule.pattern)}"
    if rule.probability is not None:
        result += f" with probability {rule.probability:g}"
    return result + "."


def _describe_condition(condition: Condition) -> str:
    labels = {
        "==": "exactly",
        "!=": "a number other than",
        "<": "fewer than",
        "<=": "at most",
        ">": "more than",
        ">=": "at least",
        "in": "one of",
    }
    value = condition.value
    formatted = (
        ", ".join(str(item) for item in value)
        if isinstance(value, list)
        else str(value)
    )
    return (
        f"the number of neighboring cells in state {condition.count} is "
        f"{labels[condition.op]} {formatted}"
    )


def _show_experiment_info(directory: str) -> int:
    stored = load_experiment(directory)
    print(f"Automaton: {stored.automaton.name}")
    print(f"Completed generations: {stored.completed_generations}")
    print(f"World shape: {stored.world.shape}")
    print(f"Seed: {stored.seed}")
    print(f"Full history stored: {'yes' if stored.store_history else 'no'}")
    counts = stored.summaries[-1]["state_counts"]
    print("Current state counts:")
    for state in stored.automaton.states:
        print(f"  {state}: {counts[state]}")
    repeated = _first_repeated_hash(stored.summaries)
    if repeated is None:
        print("Repeated world hash: none")
    else:
        first_generation, repeated_generation = repeated
        print(
            f"Repeated world hash: generation {repeated_generation} "
            f"matches generation {first_generation}"
        )
    return 0


def _first_repeated_hash(
    summaries: list[dict[str, Any]],
) -> tuple[int, int] | None:
    seen: dict[str, int] = {}
    for summary in summaries:
        digest = summary.get("hash", summary.get("sha256"))
        if digest in seen:
            return seen[digest], int(summary["generation"])
        seen[digest] = int(summary["generation"])
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cellab",
        description="Run, resume, and inspect cellular-automaton experiments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="list automata in the library")

    show_parser = commands.add_parser("show", help="show an automaton and its rules")
    show_parser.add_argument("automaton", help="library name or automaton JSON file")

    run_parser = commands.add_parser("run", help="start and save an experiment")
    run_parser.add_argument(
        "automaton_name",
        nargs="?",
        help="automaton library name or JSON path",
    )
    run_source = run_parser.add_mutually_exclusive_group()
    run_source.add_argument(
        "--file",
        "--automaton",
        dest="file",
        help="load an automaton definition from JSON",
    )
    run_parser.add_argument(
        "--size",
        help="initial world size HxW (or W for one row)",
    )
    run_parser.add_argument(
        "--init",
        choices=("random", "template", "text", "file", "default"),
        help="initialization mode; defaults to the automaton's recommendation",
    )
    run_parser.add_argument(
        "--density",
        help="random state densities, for example TREE=0.6,FIRE=0.01",
    )
    run_parser.add_argument("--template", help="named initial-world template")
    run_parser.add_argument(
        "--world",
        help="text world file for --init text or NumPy .npy file for --init file",
    )
    run_parser.add_argument(
        "--generations",
        type=_non_negative_integer,
        required=True,
        help="number of generations to run",
    )
    run_parser.add_argument("--seed", type=int, help="reproducibility seed")
    run_parser.add_argument(
        "--store",
        "--store-history",
        dest="store",
        action="store_true",
        help="store every generation in the compressed history",
    )
    run_parser.add_argument(
        "--save",
        "--output",
        dest="save",
        required=True,
        help="experiment output directory",
    )
    run_parser.add_argument(
        "--show",
        metavar="every:N",
        help="print generation 0 and every Nth generation",
    )
    run_parser.add_argument(
        "--animate",
        action="store_true",
        help="redraw the current generation in the terminal",
    )

    resume_parser = commands.add_parser("resume", help="continue a saved experiment")
    resume_parser.add_argument("experiment", help="saved experiment directory")
    resume_parser.add_argument(
        "--generations",
        type=_non_negative_integer,
        required=True,
        help="additional generations to run",
    )
    resume_parser.add_argument("--show", metavar="every:N")
    resume_parser.add_argument("--animate", action="store_true")

    info_parser = commands.add_parser("info", help="summarize a saved experiment")
    info_parser.add_argument("experiment", help="saved experiment directory")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
