import numpy as np
import pytest

from cellab.engine import count_neighbors, run, step
from cellab.library import load_automaton
from cellab.model import Automaton


# ════════════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════════════

@pytest.fixture
def cross_grid():
    """3×3 grid with a Von Neumann cross of 1s (state index 1)::

        0 1 0
        1 1 1
        0 1 0
    """
    return np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]], dtype=np.int8)


@pytest.fixture
def row_1d():
    """1×5 single-row grid for left-center-right tests: [1,0,1,1,0]."""
    return np.array([[1, 0, 1, 1, 0]], dtype=np.int8)


@pytest.fixture
def life():
    return load_automaton("game_of_life")


@pytest.fixture
def forest():
    return load_automaton("forest_fire")


# ════════════════════════════════════════════════════════════════════
#  count_neighbors tests
# ════════════════════════════════════════════════════════════════════

# ── Moore neighbourhood ────────────────────────────────────────────

class TestMoore:

    def test_toroidal_centre_cell(self, cross_grid):
        result = count_neighbors(cross_grid, 1, "moore", "toroidal")
        # Centre (1,1) has 4 orthogonal neighbours = 1 but 4 diagonal = 0
        assert result[1, 1] == 4

    def test_toroidal_corner_cell(self, cross_grid):
        result = count_neighbors(cross_grid, 1, "moore", "toroidal")
        # Corner (0,0) wraps: its 8 Moore neighbours include the
        # toroidal edges.  Expected = 5.
        assert result[0, 0] == 5

    def test_fixed_centre_cell(self, cross_grid):
        result = count_neighbors(cross_grid, 1, "moore", "fixed")
        # Same centre as toroidal (no boundary cells involved).
        assert result[1, 1] == 4

    def test_fixed_corner_cell(self, cross_grid):
        result = count_neighbors(cross_grid, 1, "moore", "fixed")
        # Corner (0,0) in fixed mode: 3 of 8 neighbours are outside
        # (fill 0).  Visible neighbours: (0,1)=1, (1,0)=1, (1,1)=1 → 3.
        assert result[0, 0] == 3

    def test_shape_preserved(self, cross_grid):
        result = count_neighbors(cross_grid, 1, "moore", "toroidal")
        assert result.shape == cross_grid.shape

    def test_all_zeros_for_absent_state(self, cross_grid):
        # State 2 doesn't appear in the grid → every count is 0.
        result = count_neighbors(cross_grid, 2, "moore", "toroidal")
        assert (result == 0).all()

    def test_all_ones_grid(self):
        grid = np.ones((4, 4), dtype=np.int8)
        result = count_neighbors(grid, 1, "moore", "toroidal")
        # Every cell has 8 neighbours, all in state 1.
        assert (result == 8).all()


# ── Von Neumann neighbourhood ──────────────────────────────────────

class TestVonNeumann:

    def test_toroidal_centre(self, cross_grid):
        result = count_neighbors(cross_grid, 1, "von neumann", "toroidal")
        # Centre (1,1): up=1, down=1, left=1, right=1 → 4.
        assert result[1, 1] == 4

    def test_toroidal_corner(self, cross_grid):
        result = count_neighbors(cross_grid, 1, "von neumann", "toroidal")
        # (0,0): up=wrap(2,0)=0, down=(1,0)=1, left=wrap(0,2)=0,
        # right=(0,1)=1 → 2.
        assert result[0, 0] == 2

    def test_fixed_edge(self, cross_grid):
        result = count_neighbors(cross_grid, 1, "von neumann", "fixed")
        # (0,1): up=fill(0), down=(1,1)=1, left=(0,0)=0, right=(0,2)=0 → 1.
        assert result[0, 1] == 1

    def test_fixed_centre(self, cross_grid):
        result = count_neighbors(cross_grid, 1, "von neumann", "fixed")
        assert result[1, 1] == 4


# ── Left-center-right neighbourhood ────────────────────────────────

class TestLeftCenterRight:

    def test_toroidal(self, row_1d):
        result = count_neighbors(row_1d, 1, "left-center-right", "toroidal")
        # Kernel [[1, 0, 1]]: counts left + right.
        # [1,0,1,1,0]:
        #   idx 0: left=wrap(0) right=0  → 0
        #   idx 1: left=1      right=1  → 2
        #   idx 2: left=0      right=1  → 1
        #   idx 3: left=1      right=0  → 1
        #   idx 4: left=1      right=wrap(1)  → 2
        np.testing.assert_array_equal(result, [[0, 2, 1, 1, 2]])

    def test_fixed(self, row_1d):
        result = count_neighbors(row_1d, 1, "left-center-right", "fixed")
        # Same but edges fill with 0.
        #   idx 0: left=fill(0) right=0  → 0
        #   idx 4: left=1      right=fill(0) → 1
        np.testing.assert_array_equal(result, [[0, 2, 1, 1, 1]])


# ── Counting different states ──────────────────────────────────────

class TestMultiState:
    """Grid with 3 states (0, 1, 2) — proves state_idx is generic."""

    @pytest.fixture
    def multi_grid(self):
        return np.array([[0, 1, 2],
                         [2, 0, 1],
                         [1, 2, 0]], dtype=np.int8)

    def test_count_state_0(self, multi_grid):
        result = count_neighbors(multi_grid, 0, "moore", "toroidal")
        # Centre cell (1,1): neighbours are 0,1,2,2,1,1,2,0 → two 0s.
        assert result[1, 1] == 2

    def test_count_state_2(self, multi_grid):
        result = count_neighbors(multi_grid, 2, "moore", "toroidal")
        # Centre cell (1,1): neighbours are 0,1,2,2,1,1,2,0 → three 2s.
        assert result[1, 1] == 3


# ── Error handling ──────────────────────────────────────────────────

class TestErrors:

    def test_unknown_neighborhood_rejected(self, cross_grid):
        with pytest.raises(ValueError, match="Unknown neighborhood"):
            count_neighbors(cross_grid, 1, "hexagonal", "toroidal")

    def test_unknown_boundary_rejected(self, cross_grid):
        with pytest.raises(ValueError, match="Unknown boundary"):
            count_neighbors(cross_grid, 1, "moore", "infinite")


# ════════════════════════════════════════════════════════════════════
#  step / run tests
# ════════════════════════════════════════════════════════════════════

# ── 1. Blinker (period-2 oscillator) ───────────────────────────────

class TestBlinker:
    """A horizontal line of 3 alive cells oscillates to vertical and back."""

    def test_blinker_returns_after_two_steps(self, life):
        # DEAD=0, ALIVE=1 — placed in a 5×5 grid
        ALIVE = life.states.index("ALIVE")
        grid = np.zeros((5, 5), dtype=np.int8)
        grid[2, 1:4] = ALIVE  # horizontal line at row 2

        rng = np.random.default_rng(0)
        g1 = step(grid, life, rng)

        # After 1 step: should be vertical
        assert g1[2, 2] == ALIVE
        assert g1[1, 2] == ALIVE
        assert g1[3, 2] == ALIVE
        assert g1[2, 1] == life.states.index("DEAD")

        g2 = step(g1, life, rng)

        # After 2 steps: back to original
        np.testing.assert_array_equal(g2, grid)

    def test_blinker_via_run(self, life):
        ALIVE = life.states.index("ALIVE")
        grid = np.zeros((5, 5), dtype=np.int8)
        grid[2, 1:4] = ALIVE

        final = run(grid, life, generations=2, seed=0)
        np.testing.assert_array_equal(final, grid)


# ── 2. Block (still life) ─────────────────────────────────────────

class TestBlock:
    """A 2×2 block of alive cells is stable — never changes."""

    def test_block_unchanged_after_one_step(self, life):
        ALIVE = life.states.index("ALIVE")
        grid = np.zeros((6, 6), dtype=np.int8)
        grid[2:4, 2:4] = ALIVE  # 2×2 block in the middle

        rng = np.random.default_rng(0)
        result = step(grid, life, rng)

        np.testing.assert_array_equal(result, grid)

    def test_block_unchanged_after_100_steps(self, life):
        ALIVE = life.states.index("ALIVE")
        grid = np.zeros((6, 6), dtype=np.int8)
        grid[2:4, 2:4] = ALIVE

        final = run(grid, life, generations=100, seed=0)
        np.testing.assert_array_equal(final, grid)


# ── 3. Glider (spaceship) ─────────────────────────────────────────

class TestGlider:
    """A glider translates (1, 1) every 4 generations on a toroidal grid."""

    def test_glider_shifts_by_1_1_after_4_steps(self, life):
        ALIVE = life.states.index("ALIVE")
        grid = np.zeros((10, 10), dtype=np.int8)

        # Standard glider pattern:
        #   .#.
        #   ..#
        #   ###
        grid[0, 1] = ALIVE
        grid[1, 2] = ALIVE
        grid[2, 0] = ALIVE
        grid[2, 1] = ALIVE
        grid[2, 2] = ALIVE

        final = run(grid, life, generations=4, seed=0)

        # Same pattern shifted by (+1, +1)
        expected = np.zeros_like(grid)
        expected[1, 2] = ALIVE
        expected[2, 3] = ALIVE
        expected[3, 1] = ALIVE
        expected[3, 2] = ALIVE
        expected[3, 3] = ALIVE

        np.testing.assert_array_equal(final, expected)


# ── 4. Boundary behaviour ─────────────────────────────────────────

class TestBoundary:
    """A pattern near the edge behaves differently under toroidal vs fixed."""

    def _make_life(self, boundary: str) -> Automaton:
        """Game of Life with the requested boundary condition."""
        return Automaton.model_validate({
            "name": "Life",
            "states": ["DEAD", "ALIVE"],
            "default_state": "DEAD",
            "neighborhood": "moore",
            "boundary": boundary,
            "rules": [
                {"from": "ALIVE", "to": "DEAD",
                 "if": {"count": "ALIVE", "op": "<", "value": 2}},
                {"from": "ALIVE", "to": "DEAD",
                 "if": {"count": "ALIVE", "op": ">", "value": 3}},
                {"from": "DEAD", "to": "ALIVE",
                 "if": {"count": "ALIVE", "op": "==", "value": 3}},
            ],
        })

    def test_toroidal_vs_fixed_differ_at_edge(self):
        # Place a glider in the top-left corner so it touches the boundary.
        # On a 6×6 grid the glider moves (+1,+1) every 4 steps and wraps
        # around at ~16 steps on a toroidal grid but dies at the edge on
        # a fixed grid.
        grid = np.zeros((6, 6), dtype=np.int8)
        grid[0, 1] = 1
        grid[1, 2] = 1
        grid[2, 0] = 1
        grid[2, 1] = 1
        grid[2, 2] = 1

        life_wrap = self._make_life("toroidal")
        life_fixed = self._make_life("fixed")

        result_wrap = run(grid.copy(), life_wrap, generations=20, seed=0)
        result_fixed = run(grid.copy(), life_fixed, generations=20, seed=0)

        assert not np.array_equal(result_wrap, result_fixed), \
            "Toroidal and fixed boundaries should give different results " \
            "when a pattern interacts with the edge."


# ── 5. Stochastic reproducibility (Forest Fire) ───────────────────

class TestStochasticReproducibility:

    def test_same_seed_same_result(self, forest):
        grid = np.zeros((20, 20), dtype=np.int8)
        tree = forest.states.index("TREE")
        grid[5:15, 5:15] = tree  # 10×10 block of trees

        a = run(grid.copy(), forest, generations=50, seed=42)
        b = run(grid.copy(), forest, generations=50, seed=42)

        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_result(self, forest):
        grid = np.zeros((20, 20), dtype=np.int8)
        tree = forest.states.index("TREE")
        grid[5:15, 5:15] = tree

        a = run(grid.copy(), forest, generations=50, seed=42)
        c = run(grid.copy(), forest, generations=50, seed=99)

        assert not np.array_equal(a, c), \
            "Different seeds should (almost certainly) produce different grids."


# ── 6. 1-D automaton (shape 1×W) ──────────────────────────────────

class Test1D:
    """A 1-D automaton with left-center-right neighbourhood runs through
    the same step() / run() code without shape errors."""

    @pytest.fixture
    def automaton_1d(self):
        """Minimal 1-D automaton with a pattern rule."""
        return Automaton.model_validate({
            "name": "Simple 1D",
            "states": ["OFF", "ON"],
            "default_state": "OFF",
            "neighborhood": "left-center-right",
            "boundary": "toroidal",
            "rules": [
                # ON cell with ON left and ON right stays ON (no-op,
                # but exercises the pattern path)
                {"from": "OFF", "to": "ON",
                 "pattern": ["ON", "OFF", "ON"]},
            ],
        })

    def test_step_on_single_row(self, automaton_1d):
        # 1×8 grid
        grid = np.array([[1, 0, 1, 0, 0, 1, 0, 1]], dtype=np.int8)
        rng = np.random.default_rng(0)
        result = step(grid, automaton_1d, rng)

        assert result.shape == (1, 8), "Shape must stay 1×W"
        assert result.dtype == grid.dtype

    def test_run_on_single_row(self, automaton_1d):
        grid = np.array([[1, 0, 1, 0, 0, 1, 0, 1]], dtype=np.int8)
        final = run(grid, automaton_1d, generations=10, seed=0)

        assert final.shape == (1, 8)

    def test_pattern_rule_fires_correctly(self, automaton_1d):
        # Pattern [ON, OFF, ON] → OFF becomes ON.
        # Grid:  ON OFF ON OFF ...
        #         1   0   1   0  0  0  0  0
        # Cell 1 has left=ON, center=OFF, right=ON → fires → becomes ON.
        grid = np.array([[1, 0, 1, 0, 0, 0, 0, 0]], dtype=np.int8)
        rng = np.random.default_rng(0)
        result = step(grid, automaton_1d, rng)

        assert result[0, 1] == 1, "Pattern [ON, OFF, ON] should fire at index 1"


# ── 7. on_step callback ───────────────────────────────────────────

class TestOnStepCallback:

    def test_callback_receives_every_generation(self, life):
        ALIVE = life.states.index("ALIVE")
        grid = np.zeros((5, 5), dtype=np.int8)
        grid[2, 1:4] = ALIVE  # blinker

        recorded = []
        run(grid, life, generations=5, seed=0,
            on_step=lambda t, g: recorded.append((t, g.copy())))

        assert [t for t, _ in recorded] == [1, 2, 3, 4, 5]
        # Blinker period 2: gen 2 == gen 0, gen 4 == gen 0
        np.testing.assert_array_equal(recorded[1][1], grid)
        np.testing.assert_array_equal(recorded[3][1], grid)
