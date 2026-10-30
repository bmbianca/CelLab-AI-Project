import pytest
from pydantic import ValidationError
from cellab.model import Automaton

# A valid baseline dictionary (JSON loaded into Python)
VALID_AUTOMATON = {
    "name": "Forest Fire Test",
    "states": ["EMPTY", "TREE", "FIRE"],
    "default_state": "EMPTY",
    "neighborhood": "moore",
    "boundary": "toroidal",
    "rules": [
        {"from": "FIRE", "to": "EMPTY"},
        {"from": "TREE", "to": "FIRE", "probability": 0.05}
    ]
}


def test_valid_automaton():
    """A correctly formatted automaton should parse without errors."""
    automaton = Automaton.model_validate(VALID_AUTOMATON)
    assert automaton.name == "Forest Fire Test"
    assert len(automaton.rules) == 2


def test_invalid_probability_rejected():
    """Probabilities outside [0, 1] must be rejected."""
    invalid_data = VALID_AUTOMATON.copy()
    invalid_data["rules"] = [
        {"from": "TREE", "to": "FIRE", "probability": 1.5}
    ]

    with pytest.raises(ValidationError) as exc_info:
        Automaton.model_validate(invalid_data)

    assert "probability" in str(exc_info.value)


def test_undeclared_state_rejected():
    """Using a state not defined in 'states' must be rejected."""
    invalid_data = VALID_AUTOMATON.copy()
    invalid_data["rules"] = [
        {"from": "MAGIC_STATE", "to": "EMPTY"}
    ]

    with pytest.raises(ValidationError) as exc_info:
        Automaton.model_validate(invalid_data)

    assert "not declared in states" in str(exc_info.value)