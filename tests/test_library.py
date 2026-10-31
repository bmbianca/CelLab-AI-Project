import json

import pytest
from pydantic import ValidationError

from cellab.library import list_automata, load_automaton
from cellab.model import Automaton


def test_library_has_the_three_required_automata():
    names = list_automata()
    for required in ["game_of_life", "brians_brain", "forest_fire"]:
        assert required in names


@pytest.mark.parametrize("name", list_automata())
def test_every_library_file_is_valid(name):
    """Every JSON in library/ loads and has the Member 2 extra fields."""
    automaton = load_automaton(name)
    assert automaton.description
    assert automaton.symbols is not None
    assert set(automaton.symbols) == set(automaton.states)


def test_unknown_name_gives_clear_error():
    with pytest.raises(FileNotFoundError) as exc_info:
        load_automaton("does_not_exist")
    assert "Available" in str(exc_info.value)


def test_load_from_any_json_path(tmp_path):
    """An automaton outside the library (invented states A, B) also loads."""
    data = {
        "name": "Invented",
        "states": ["A", "B"],
        "default_state": "A",
        "neighborhood": "von neumann",
        "boundary": "fixed",
        "rules": [{"from": "A", "to": "B", "if": {"count": "B", "op": ">=", "value": 1}}],
        "symbols": {"A": "a", "B": "b"},
    }
    path = tmp_path / "invented.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_automaton(str(path)).name == "Invented"


def test_bad_symbols_rejected():
    base = load_automaton("game_of_life").model_dump(by_alias=True)
    base["symbols"] = {"DEAD": ".", "ALIVE": "."}  # same symbol twice
    with pytest.raises(ValidationError):
        Automaton.model_validate(base)


def test_template_with_unknown_symbol_rejected():
    base = load_automaton("game_of_life").model_dump(by_alias=True)
    base["templates"] = {"bad": ["#X#"]}
    with pytest.raises(ValidationError):
        Automaton.model_validate(base)
