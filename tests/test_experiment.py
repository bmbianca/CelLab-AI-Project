import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from cellab import cli
from cellab.experiment import ExperimentConfig, resume_experiment, run_experiment
from cellab.library import load_automaton
from cellab.model import Automaton


AUTOMATON = Automaton.model_validate(
    {
        "name": "Test",
        "states": ["A", "B"],
        "default_state": "A",
        "neighborhood": "moore",
        "boundary": "toroidal",
        "rules": [],
    }
)
INITIAL_WORLD = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)


def identity_step(
    world: np.ndarray,
    _automaton: Automaton,
    _rng: np.random.Generator,
) -> np.ndarray:
    _ = _automaton, _rng
    return world.copy()


def random_step(world, automaton, rng):
    changes = rng.random(world.shape) < 0.3
    return np.where(changes, (world + 1) % len(automaton.states), world)


def test_resume_matches_uninterrupted_run(tmp_path):
    uninterrupted = run_experiment(
        AUTOMATON,
        INITIAL_WORLD,
        generations=100,
        seed=4291,
        store_history=True,
        output_directory=tmp_path / "whole",
        step=random_step,
    )
    split_directory = tmp_path / "split"
    first_half = run_experiment(
        AUTOMATON,
        INITIAL_WORLD,
        generations=50,
        seed=4291,
        store_history=True,
        output_directory=split_directory,
        step=random_step,
    )
    assert first_half.completed_generations == 50

    resumed = resume_experiment(
        split_directory,
        additional_generations=50,
        step=random_step,
    )

    np.testing.assert_array_equal(resumed.world, uninterrupted.world)
    assert resumed.completed_generations == 100
    assert len(resumed.history) == 101
    assert resumed.summaries == uninterrupted.summaries


def test_stochastic_automaton_resume_restores_rng_state(tmp_path):
    automaton = load_automaton("forest_fire")
    initial_world = np.zeros((5, 7), dtype=np.int8)

    def stochastic_step(world, definition, rng):
        return rng.integers(
            0, len(definition.states), size=world.shape, dtype=world.dtype
        )

    uninterrupted = run_experiment(
        automaton,
        initial_world,
        generations=100,
        seed=4291,
        store_history=True,
        output_directory=tmp_path / "forest-whole",
        step=stochastic_step,
    )
    split_directory = tmp_path / "forest-split"
    run_experiment(
        automaton,
        initial_world,
        generations=50,
        seed=4291,
        store_history=True,
        output_directory=split_directory,
        step=stochastic_step,
    )
    resumed = resume_experiment(
        split_directory,
        additional_generations=50,
        step=stochastic_step,
    )

    np.testing.assert_array_equal(resumed.world, uninterrupted.world)
    assert resumed.summaries == uninterrupted.summaries


def test_disabled_history_keeps_checkpoint_and_summaries(tmp_path):
    result = run_experiment(
        AUTOMATON,
        INITIAL_WORLD,
        generations=3,
        seed=7,
        store_history=False,
        output_directory=tmp_path / "compact",
        step=random_step,
    )

    assert result.history == []
    assert len(result.summaries) == 4
    assert len(result.summaries[0]["hash"]) == 128
    assert result.summaries[0]["state_counts"] == {"A": 3, "B": 3}
    with np.load(tmp_path / "compact" / "history.npz", allow_pickle=False) as data:
        assert data.files == [
            "initial_world",
            "current_world",
            "summary_generations",
            "hashes",
            "state_counts",
        ]
        np.testing.assert_array_equal(data["initial_world"], INITIAL_WORLD)
        np.testing.assert_array_equal(data["current_world"], result.world)
        np.testing.assert_array_equal(data["summary_generations"], [0, 1, 2, 3])
        assert data["state_counts"].shape == (4, len(AUTOMATON.states))
        assert data["hashes"].shape == (4,)
    config = json.loads((tmp_path / "compact" / "config.json").read_text())
    assert config["seed"] == 7
    assert config["generations"] == 3
    assert "summaries" not in config
    assert config["initialization"] == {"mode": "provided"}
    assert config["rng_state"]
    saved_automaton = json.loads(
        (tmp_path / "compact" / "automaton.json").read_text()
    )
    assert saved_automaton["name"] == AUTOMATON.name


def test_invalid_world_and_generation_count_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="state index"):
        run_experiment(
            AUTOMATON,
            np.array([[2]]),
            generations=1,
            step=random_step,
            output_directory=tmp_path / "invalid",
        )

    with pytest.raises(ValueError, match="non-negative integer"):
        run_experiment(
            AUTOMATON,
            INITIAL_WORLD,
            generations=-1,
            step=random_step,
            output_directory=tmp_path / "invalid-count",
        )


def test_cli_runs_and_resumes_with_engine_step(tmp_path, monkeypatch, capsys):
    engine_module = types.ModuleType("cellab.engine")
    engine_module.step = random_step
    monkeypatch.setitem(sys.modules, "cellab.engine", engine_module)

    automaton_path = tmp_path / "automaton.json"
    automaton_path.write_text(
        json.dumps(AUTOMATON.model_dump(mode="json", by_alias=True)),
        encoding="utf-8",
    )
    world_path = tmp_path / "world.npy"
    np.save(world_path, INITIAL_WORLD)
    experiment_path = tmp_path / "experiment"

    assert cli.main(
        [
            "run",
            "--automaton",
            str(automaton_path),
            "--world",
            str(world_path),
            "--generations",
            "2",
            "--seed",
            "11",
            "--output",
            str(experiment_path),
        ]
    ) == 0
    assert "Seed: 11" in capsys.readouterr().out

    assert cli.main(
        [
            "resume",
            str(experiment_path),
            "--generations",
            "2",
        ]
    ) == 0
    assert "Completed generations: 4" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("name", "mode", "template"),
    [
        ("game_of_life", "random", None),
        ("game_of_life", "template", "blinker"),
        ("game_of_life", "text", None),
        ("brians_brain", "random", None),
        ("brians_brain", "template", "glider"),
        ("brians_brain", "text", None),
        ("forest_fire", "random", None),
        ("forest_fire", "template", "spark"),
        ("forest_fire", "text", None),
    ],
)
def test_cli_initialization_modes_use_library_data(
    tmp_path, monkeypatch, capsys, name, mode, template
):
    engine_module = types.ModuleType("cellab.engine")
    engine_module.step = identity_step
    monkeypatch.setitem(sys.modules, "cellab.engine", engine_module)
    automaton = load_automaton(name)
    args = [
        "run",
        name,
        "--init",
        mode,
        "--generations",
        "2",
        "--seed",
        "17",
        "--save",
        str(tmp_path / name),
    ]
    if mode == "text":
        symbol = automaton.symbols[automaton.default_state]
        world_file = tmp_path / f"{name}.txt"
        world_file.write_text("\n".join([symbol * 4] * 3), encoding="utf-8")
        args.extend(["--world", str(world_file)])
    else:
        args.extend(["--size", "8x8"])
    if mode == "random":
        args.extend(["--density", f"{automaton.states[-1]}=0.1"])
    if template:
        args.extend(["--template", template])

    assert cli.main(args) == 0
    output = capsys.readouterr().out
    assert "Seed: 17" in output
    stored_config = json.loads(
        (tmp_path / name / "config.json").read_text(encoding="utf-8")
    )
    assert stored_config["initialization"]["mode"] == mode
    with np.load(tmp_path / name / "history.npz", allow_pickle=False) as archive:
        expected_shape = (3, 4) if mode == "text" else (8, 8)
        assert archive["initial_world"].shape == expected_shape


def test_cli_random_world_and_simulation_share_seeded_rng(
    tmp_path, monkeypatch, capsys
):
    automaton = load_automaton("forest_fire")
    engine_module = types.ModuleType("cellab.engine")
    engine_module.step = random_step
    monkeypatch.setitem(sys.modules, "cellab.engine", engine_module)
    seed = 391

    assert cli.main(
        [
            "run",
            "forest_fire",
            "--init",
            "random",
            "--size",
            "7x9",
            "--density",
            "TREE=0.4,FIRE=0.1",
            "--generations",
            "1",
            "--seed",
            str(seed),
            "--save",
            str(tmp_path / "shared-rng"),
        ]
    ) == 0
    capsys.readouterr()

    rng = np.random.default_rng(seed)
    tree = automaton.states.index("TREE")
    fire = automaton.states.index("FIRE")
    random_values = rng.random((7, 9))
    expected_initial = np.zeros((7, 9), dtype=np.int8)
    expected_initial[(random_values >= 0.0) & (random_values < 0.4)] = tree
    expected_initial[(random_values >= 0.4) & (random_values < 0.5)] = fire
    expected_next = random_step(expected_initial, automaton, rng)

    with np.load(tmp_path / "shared-rng" / "history.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["initial_world"], expected_initial)
        np.testing.assert_array_equal(data["current_world"], expected_next)


def test_cli_generates_and_saves_seed_when_omitted(tmp_path, monkeypatch, capsys):
    automaton = load_automaton("forest_fire")
    engine_module = types.ModuleType("cellab.engine")
    engine_module.step = identity_step
    monkeypatch.setitem(sys.modules, "cellab.engine", engine_module)

    assert cli.main(
        [
            "run",
            "forest_fire",
            "--init",
            "random",
            "--size",
            "6x7",
            "--density",
            "TREE=0.25",
            "--generations",
            "0",
            "--save",
            str(tmp_path / "generated-seed"),
        ]
    ) == 0
    output = capsys.readouterr().out
    config = json.loads(
        (tmp_path / "generated-seed" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    assert f"Seed: {config['seed']}" in output
    assert 0 <= config["seed"] < 2**64
    rng = np.random.default_rng(config["seed"])
    random_values = rng.random((6, 7))
    expected_initial = np.zeros((6, 7), dtype=np.int8)
    expected_initial[random_values < 0.25] = automaton.states.index("TREE")
    with np.load(
        tmp_path / "generated-seed" / "history.npz",
        allow_pickle=False,
    ) as archive:
        np.testing.assert_array_equal(archive["initial_world"], expected_initial)


def test_cli_show_every_prints_requested_generations(tmp_path, monkeypatch, capsys):
    engine_module = types.ModuleType("cellab.engine")
    engine_module.step = identity_step
    monkeypatch.setitem(sys.modules, "cellab.engine", engine_module)

    assert cli.main(
        [
            "run",
            "game_of_life",
            "--init",
            "template",
            "--template",
            "block",
            "--size",
            "8x8",
            "--generations",
            "3",
            "--seed",
            "4",
            "--show",
            "every:2",
            "--save",
            str(tmp_path / "display"),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "Generation 0:" in output
    assert "Generation 2:" in output
    assert "Generation 1:" not in output


def test_cli_list_show_and_info_commands(tmp_path, monkeypatch, capsys):
    engine_module = types.ModuleType("cellab.engine")
    engine_module.step = identity_step
    monkeypatch.setitem(sys.modules, "cellab.engine", engine_module)

    assert cli.main(["list"]) == 0
    assert "game_of_life" in capsys.readouterr().out

    assert cli.main(["show", "forest_fire"]) == 0
    show_output = capsys.readouterr().out
    assert "Forest Fire" in show_output
    assert "probability 0.001" in show_output

    assert cli.main(
        [
            "run",
            "game_of_life",
            "--init",
            "template",
            "--template",
            "block",
            "--size",
            "8x8",
            "--generations",
            "2",
            "--seed",
            "4",
            "--save",
            str(tmp_path / "stable"),
        ]
    ) == 0
    capsys.readouterr()
    assert cli.main(["info", str(tmp_path / "stable")]) == 0
    info_output = capsys.readouterr().out
    assert "Completed generations: 2" in info_output
    assert "matches generation 0" in info_output


def test_experiment_config_validates_and_contains_automaton():
    config = ExperimentConfig(
        automaton=AUTOMATON,
        initialization={"mode": "random", "size": [2, 3]},
        generations=12,
        seed=3,
        store_history=False,
    )

    assert config.automaton == AUTOMATON
    assert config.generations == 12


def test_python_module_entrypoint_lists_automata_in_subprocess():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "cellab", "list"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "game_of_life" in result.stdout
    assert "brians_brain" in result.stdout
    assert "forest_fire" in result.stdout
