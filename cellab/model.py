"""
CelLab – Data-model layer (JSON v2 format).

Defines Pydantic models for cellular-automaton rule files and a loader
function.  No simulation logic lives here; the engine and NumPy kernels
belong in separate modules.

Every validation is *data-driven* — nothing is hardcoded to a specific
automaton or state name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────

NeighborhoodType = Literal["moore", "von neumann", "left-center-right"]
BoundaryType = Literal["toroidal", "fixed"]
OperatorType = Literal["==", "!=", "<", "<=", ">", ">=", "in"]


def max_neighbors(neighborhood: NeighborhoodType) -> int:
    """Return the maximum number of neighbors for a given neighborhood type.

    * ``"moore"``              → 8
    * ``"von neumann"``        → 4
    * ``"left-center-right"``  → 2

    Raises ``ValueError`` for unrecognized neighborhood strings.
    """
    mapping = {
        "moore": 8,
        "von neumann": 4,
        "left-center-right": 2,
    }
    if neighborhood not in mapping:
        raise ValueError(
            f"Unknown neighborhood '{neighborhood}'. "
            f"Expected one of: {', '.join(mapping)}"
        )
    return mapping[neighborhood]


# ────────────────────────────────────────────────────────────────────
# Condition
# ────────────────────────────────────────────────────────────────────

class Condition(BaseModel):
    """A single neighbor-count predicate.

    Example JSON:

        {"count": "FIRE", "op": ">=", "value": 1}

    ``value`` may be an ``int`` for comparison operators or a ``list[int]``
    when ``op`` is ``"in"``.
    """

    count: str = Field(
        ...,
        description="The state whose neighbours are counted.",
    )
    op: OperatorType = Field(
        ...,
        description="Comparison operator to apply.",
    )
    value: Union[int, List[int]] = Field(
        ...,
        description=(
            "Threshold value (int) or list of acceptable values "
            "(list[int]) when op='in'."
        ),
    )

    # -- validators ------------------------------------------------

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_value(cls, v, info):
        """Allow ``value`` to be a plain int or a list depending on ``op``."""
        return v  # typing union handles this; further checks in model_validator

    @model_validator(mode="after")
    def _validate_op_value_consistency(self) -> "Condition":
        """``op='in'`` requires a list; every other op requires a scalar."""
        if self.op == "in":
            if not isinstance(self.value, list):
                raise ValueError(
                    "When op='in', value must be a list of integers."
                )
        else:
            if isinstance(self.value, list):
                raise ValueError(
                    f"When op='{self.op}', value must be an integer, "
                    f"not a list."
                )
        return self


# ────────────────────────────────────────────────────────────────────
# Rule
# ────────────────────────────────────────────────────────────────────

class Rule(BaseModel):
    """A single transition rule.

    A rule can be:

    * **unconditional** – just ``from`` / ``to``.
    * **conditional**   – includes an ``if`` (single ``Condition`` or list).
    * **probabilistic** – includes a ``probability`` ∈ [0, 1].
    * **pattern-based** – includes ``pattern`` (list of states for 1-D
      neighborhoods).

    Combinations are allowed (e.g., a probabilistic plus conditional rule).
    """

    from_state: str = Field(
        ...,
        alias="from",
        description="Source state for this transition.",
    )
    to_state: str = Field(
        ...,
        alias="to",
        description="Target state for this transition.",
    )

    # Optional condition(s) — a single object OR a list of objects.
    if_condition: Optional[Union[Condition, List[Condition]]] = Field(
        default=None,
        alias="if",
        description=(
            "One or more conditions that must ALL be satisfied for the "
            "rule to fire."
        ),
    )

    probability: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probability of transition (0–1). Validated to be in range.",
    )

    pattern: Optional[List[str]] = Field(
        default=None,
        description=(
            "Exact pattern for 1-D (left-center-right) automata. "
            "Must be a list of exactly 3 state names."
        ),
    )

    model_config = {"populate_by_name": True}

    # -- validators ------------------------------------------------

    @field_validator("pattern", mode="after")
    @classmethod
    def _validate_pattern_length(cls, v):
        """Pattern lists must have exactly 3 elements (left, center, right)."""
        if v is not None and len(v) != 3:
            raise ValueError(
                f"Pattern must contain exactly 3 states "
                f"(left, center, right), got {len(v)}."
            )
        return v


# ────────────────────────────────────────────────────────────────────
# Automaton
# ────────────────────────────────────────────────────────────────────

class Automaton(BaseModel):
    """Top-level model representing a complete automaton definition.

    This is the direct Pydantic mapping of the JSON v2 format.
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable name for this automaton.",
    )
    states: List[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of possible cell states.",
    )
    default_state: str = Field(
        ...,
        description="State assigned to every cell at initialisation.",
    )
    neighborhood: NeighborhoodType = Field(
        ...,
        description="Neighbourhood topology to use.",
    )
    boundary: BoundaryType = Field(
        ...,
        description="Boundary condition for the grid.",
    )
    rules: List[Rule] = Field(
        ...,
        description="Ordered list of transition rules.",
    )

    model_config = {"populate_by_name": True}

    # -- validators ------------------------------------------------

    @field_validator("states", mode="after")
    @classmethod
    def _states_must_be_unique(cls, v):
        """All state names must be distinct."""
        seen = set()
        for s in v:
            if s in seen:
                raise ValueError(f"Duplicate state name: '{s}'.")
            seen.add(s)
        return v

    @model_validator(mode="after")
    def _cross_field_validation(self) -> "Automaton":
        """Run all cross-field consistency checks in a single pass."""
        state_set = set(self.states)
        max_n = max_neighbors(self.neighborhood)

        # 1. default_state ∈ states
        if self.default_state not in state_set:
            raise ValueError(
                f"default_state '{self.default_state}' is not declared "
                f"in states {self.states}."
            )

        # 2. Validate every rule
        for idx, rule in enumerate(self.rules):
            prefix = f"rules[{idx}]"

            # from / to must be known states
            if rule.from_state not in state_set:
                raise ValueError(
                    f"{prefix}: 'from' state '{rule.from_state}' "
                    f"is not declared in states."
                )
            if rule.to_state not in state_set:
                raise ValueError(
                    f"{prefix}: 'to' state '{rule.to_state}' "
                    f"is not declared in states."
                )

            # Validate condition(s)
            conditions: List[Condition] = []
            if rule.if_condition is not None:
                if isinstance(rule.if_condition, list):
                    conditions = rule.if_condition
                else:
                    conditions = [rule.if_condition]

            for ci, cond in enumerate(conditions):
                cond_prefix = f"{prefix}.if[{ci}]"

                # count state must exist
                if cond.count not in state_set:
                    raise ValueError(
                        f"{cond_prefix}: count state '{cond.count}' "
                        f"is not declared in states."
                    )

                # value(s) must not exceed max_neighbors
                values = (
                    cond.value if isinstance(cond.value, list) else [cond.value]
                )
                for val in values:
                    if val < 0:
                        raise ValueError(
                            f"{cond_prefix}: value {val} must be ≥ 0."
                        )
                    if val > max_n:
                        raise ValueError(
                            f"{cond_prefix}: value {val} exceeds "
                            f"max_neighbors ({max_n}) for neighborhood "
                            f"'{self.neighborhood}'."
                        )

            # Validate pattern states (1-D only)
            if rule.pattern is not None:
                if self.neighborhood != "left-center-right":
                    raise ValueError(
                        f"{prefix}: 'pattern' is only valid with the "
                        f"'left-center-right' neighborhood."
                    )
                for pi, ps in enumerate(rule.pattern):
                    if ps not in state_set:
                        raise ValueError(
                            f"{prefix}.pattern[{pi}]: state '{ps}' "
                            f"is not declared in states."
                        )

        return self


# ────────────────────────────────────────────────────────────────────
# Public loader
# ────────────────────────────────────────────────────────────────────

def load_automaton_file(path) -> Automaton:
    """Read a JSON v2 automaton file from *path* and return a validated
    :class:`Automaton` instance.

    Parameters
    ----------
    path : str | pathlib.Path
        Filesystem path to a ``.json`` automaton definition file.

    Returns
    -------
    Automaton
        A fully-validated Pydantic model.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    pydantic.ValidationError
        If the data fails schema or cross-field validation.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Automaton file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    return Automaton.model_validate(data)
