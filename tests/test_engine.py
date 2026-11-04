import numpy as np
import pytest

from cellab.engine import count_neighbors


# ── fixtures ────────────────────────────────────────────────────────

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
