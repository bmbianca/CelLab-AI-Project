import numpy as np
import pytest

from cellab.library import list_automata, load_automaton
from cellab.model import Automaton
from cellab.world import (
    default_world,
    from_template,
    from_text,
    parse_density,
    parse_shape,
    random_world,
    to_text,
)


@pytest.fixture
def forest():
    return load_automaton("forest_fire")


@pytest.fixture
def life():
    return load_automaton("game_of_life")


# ── random_world ────────────────────────────────────────────────────

def test_random_world_respects_density(forest):
    grid = random_world((100, 100), forest, {"TREE": 0.6, "FIRE": 0.01}, seed=1)
    tree = forest.states.index("TREE")
    fire = forest.states.index("FIRE")
    assert grid.dtype == np.int8 and grid.shape == (100, 100)
    assert abs((grid == tree).mean() - 0.6) < 0.02
    assert abs((grid == fire).mean() - 0.01) < 0.02


def test_same_seed_same_world_different_seed_different_world(forest):
    a = random_world((50, 50), forest, {"TREE": 0.5}, seed=42)
    b = random_world((50, 50), forest, {"TREE": 0.5}, seed=42)
    c = random_world((50, 50), forest, {"TREE": 0.5}, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_density_over_one_rejected(forest):
    with pytest.raises(ValueError, match="more than 1"):
        random_world((10, 10), forest, {"TREE": 0.8, "FIRE": 0.3}, seed=1)


def test_unknown_state_in_density_rejected(forest):
    with pytest.raises(ValueError, match="Unknown state"):
        random_world((10, 10), forest, {"WATER": 0.1}, seed=1)


# ── text <-> world ──────────────────────────────────────────────────

def test_from_text_reads_symbols(forest):
    grid = from_text("..F..\n.TTT.", forest)
    assert grid.shape == (2, 5)
    assert grid[0, 2] == forest.states.index("FIRE")
    assert grid[1, 1] == forest.states.index("TREE")


def test_text_round_trip(forest):
    grid = random_world((20, 30), forest, {"TREE": 0.5, "FIRE": 0.1}, seed=7)
    assert np.array_equal(from_text(to_text(grid, forest), forest), grid)


def test_from_text_reads_file(tmp_path, forest):
    path = tmp_path / "world.txt"
    path.write_text("T.F\nFFT\n", encoding="utf-8")
    assert from_text(str(path), forest).shape == (2, 3)


def test_unknown_symbol_rejected(forest):
    with pytest.raises(ValueError, match="Unknown symbol 'X'"):
        from_text("..X..", forest)


def test_unequal_rows_rejected(forest):
    with pytest.raises(ValueError, match="same length"):
        from_text("...\n..", forest)


# ── templates ───────────────────────────────────────────────────────

def test_template_is_centered(life):
    grid = from_template("blinker", (5, 5), life)
    alive = life.states.index("ALIVE")
    assert grid[2, 1] == grid[2, 2] == grid[2, 3] == alive
    assert (grid == alive).sum() == 3


def test_template_too_big_rejected(life):
    with pytest.raises(ValueError, match="does not fit"):
        from_template("glider", (2, 2), life)


def test_general_templates_work_for_any_automaton(forest):
    grid = from_template("center:FIRE", (5, 7), forest)
    assert grid[2, 3] == forest.states.index("FIRE")
    row = from_template("row:TREE", (1, 9), forest)  # a 1-D world
    assert (row == forest.states.index("TREE")).all()


def test_unknown_template_rejected(life):
    with pytest.raises(ValueError, match="no template"):
        from_template("spaceship", (10, 10), life)


# ── misc ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list_automata())
def test_default_world_of_every_library_automaton(name):
    automaton = load_automaton(name)
    grid = default_world(automaton, seed=0)
    assert grid.ndim == 2 and grid.dtype == np.int8
    assert grid.max() < len(automaton.states)


def test_parsers():
    assert parse_shape("30x60") == (30, 60)
    assert parse_shape("80") == (1, 80)
    assert parse_density("TREE=0.6, FIRE=0.01") == {"TREE": 0.6, "FIRE": 0.01}


def test_invented_automaton_needs_no_code_changes():
    """States the code has never seen (A, B, C) work the same way."""
    automaton = Automaton.model_validate({
        "name": "Invented",
        "states": ["A", "B", "C"],
        "default_state": "A",
        "neighborhood": "moore",
        "boundary": "toroidal",
        "rules": [],
        "symbols": {"A": " ", "B": "b", "C": "c"},
        "templates": {"pair": ["bc"]},
    })
    grid = from_template("pair", (3, 4), automaton)
    assert to_text(grid, automaton) == "    \n bc \n    "
