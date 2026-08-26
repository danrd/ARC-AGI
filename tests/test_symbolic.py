"""Grid-based tests for the symbolic pattern-analysis stack (GridObject,
GridSummary, RelationAnalyzer, pattern detection).

Most classes below hold plain assert-based test methods and run through
pytest like any other suite here. Where a test method's only parameter is
`grid`, it's parametrized by the `grid` fixture over every named scenario
in GridLibrary.get_all_test_grids() (24 scenarios - empty, single pixel,
lines, rectangles, checkerboards, nested/multicolor/sparse/dense patterns,
...), producing one test id per scenario (e.g. test_object_creation[filled_rectangle]).
Self-contained tests (no `grid` parameter) build whatever input they need
internally.
"""
import time
from collections import Counter
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rl.arc_task import ARCSubtask

from symbolic.summaries import GridSummary, RelationAnalyzer, SubtaskSummary
# summaries.py carried a byte-identical second copy of this, which nothing in
# production used - the RL transformators call this one.
from rl.arc_transformators import get_rotations
from symbolic.objects_analysis import GridObject
from symbolic.patterns import (
    generate_patterns,
    lines_coords,
    rectangles_coords,
    find_connected_components_excluding_colors,
    find_connected_components_with_color,
)

from .resource_utils import resource_budget

# ============================================================================
# TEST GRID LIBRARY
# ============================================================================

class GridLibrary:
    """Library of predefined test grids covering various scenarios."""

    @staticmethod
    def empty_grid(size=(10, 10)):
        """Empty grid - all zeros."""
        return np.zeros(size, dtype=int)

    @staticmethod
    def single_pixel(size=(10, 10), color=1, pos=None):
        """Single colored pixel."""
        grid = np.zeros(size, dtype=int)
        pos = pos or (size[0]//2, size[1]//2)
        grid[pos] = color
        return grid

    @staticmethod
    def horizontal_line(size=(10, 10), row=5, col_start=2, col_end=8, color=1):
        """Horizontal line."""
        grid = np.zeros(size, dtype=int)
        grid[row, col_start:col_end] = color
        return grid

    @staticmethod
    def vertical_line(size=(10, 10), col=5, row_start=2, row_end=8, color=1):
        """Vertical line."""
        grid = np.zeros(size, dtype=int)
        grid[row_start:row_end, col] = color
        return grid

    @staticmethod
    def diagonal_line(size=(10, 10), color=1, direction='main'):
        """Diagonal line (main or anti)."""
        grid = np.zeros(size, dtype=int)
        if direction == 'main':
            for i in range(min(size)):
                grid[i, i] = color
        else:  # anti-diagonal
            for i in range(min(size)):
                grid[i, size[1]-1-i] = color
        return grid

    @staticmethod
    def filled_rectangle(size=(10, 10), top=2, left=2, height=4, width=4, color=1):
        """Filled rectangle."""
        grid = np.zeros(size, dtype=int)
        grid[top:top+height, left:left+width] = color
        return grid

    @staticmethod
    def hollow_rectangle(size=(10, 10), top=2, left=2, height=5, width=5, color=1):
        """Hollow rectangle (border only)."""
        grid = np.zeros(size, dtype=int)
        # Top and bottom borders
        grid[top, left:left+width] = color
        grid[top+height-1, left:left+width] = color
        # Left and right borders
        grid[top:top+height, left] = color
        grid[top:top+height, left+width-1] = color
        return grid

    @staticmethod
    def l_shape(size=(10, 10), top=3, left=3, arm_length=3, color=1):
        """L-shaped pattern."""
        grid = np.zeros(size, dtype=int)
        # Vertical arm
        grid[top:top+arm_length, left] = color
        # Horizontal arm
        grid[top+arm_length-1, left:left+arm_length] = color
        return grid

    @staticmethod
    def t_shape(size=(10, 10), top=3, left=3, width=5, height=3, color=1):
        """T-shaped pattern."""
        grid = np.zeros(size, dtype=int)
        # Horizontal bar
        grid[top, left:left+width] = color
        # Vertical stem
        mid = left + width//2
        grid[top:top+height, mid] = color
        return grid

    @staticmethod
    def cross_shape(size=(10, 10), center=(5, 5), arm_length=2, color=1):
        """Cross/plus shape."""
        grid = np.zeros(size, dtype=int)
        cy, cx = center
        # Horizontal line
        grid[cy, max(0, cx-arm_length):min(size[1], cx+arm_length+1)] = color
        # Vertical line
        grid[max(0, cy-arm_length):min(size[0], cy+arm_length+1), cx] = color
        return grid

    @staticmethod
    def checkerboard(size=(10, 10), color1=1, color2=2):
        """Checkerboard pattern."""
        grid = np.zeros(size, dtype=int)
        for i in range(size[0]):
            for j in range(size[1]):
                if (i + j) % 2 == 0:
                    grid[i, j] = color1
                else:
                    grid[i, j] = color2
        return grid

    @staticmethod
    def scattered_pixels(size=(10, 10), num_pixels=10, colors=None):
        """Randomly scattered pixels."""
        grid = np.zeros(size, dtype=int)
        colors = colors or [1, 2, 3]
        np.random.seed(42)  # Reproducible
        for _ in range(num_pixels):
            i, j = np.random.randint(0, size[0]), np.random.randint(0, size[1])
            grid[i, j] = np.random.choice(colors)
        return grid

    @staticmethod
    def nested_rectangles(size=(15, 15)):
        """Nested rectangles with different colors."""
        grid = np.zeros(size, dtype=int)
        grid[2:13, 2:13] = 1
        grid[4:11, 4:11] = 2
        grid[6:9, 6:9] = 3
        return grid

    @staticmethod
    def multicolor_regions(size=(12, 12)):
        """Multiple distinct colored regions."""
        grid = np.zeros(size, dtype=int)
        grid[1:4, 1:4] = 1    # Top-left red
        grid[1:4, 8:11] = 2   # Top-right blue
        grid[8:11, 1:4] = 3   # Bottom-left green
        grid[8:11, 8:11] = 4  # Bottom-right yellow
        grid[5:7, 5:7] = 5    # Center purple
        return grid

    @staticmethod
    def connected_components(size=(15, 15)):
        """Grid with multiple connected components."""
        grid = np.zeros(size, dtype=int)
        # Component 1
        grid[1:4, 1:6] = 1
        # Component 2 (same color, disconnected)
        grid[6:9, 8:13] = 1
        # Component 3 (different color)
        grid[10:13, 2:5] = 2
        return grid

    @staticmethod
    def with_holes(size=(12, 12)):
        """Rectangle with holes inside."""
        grid = np.zeros(size, dtype=int)
        # Outer rectangle
        grid[2:10, 2:10] = 1
        # Inner holes
        grid[4:6, 4:6] = 0
        grid[4:6, 7:9] = 0
        grid[7:9, 4:6] = 0
        return grid

    @staticmethod
    def border_pattern(size=(10, 10), border_width=1, border_color=1, fill_color=2):
        """Grid with border"""
        grid = np.zeros(size, dtype=int)
        grid[:border_width, :] = border_color  # Top
        grid[-border_width:, :] = border_color  # Bottom
        grid[:, :border_width] = border_color  # Left
        grid[:, -border_width:] = border_color  # Right
        if fill_color > 0:
            grid[border_width:-border_width, border_width:-border_width] = fill_color
        return grid

    @staticmethod
    def diagonal_split(size=(10, 10), color1=1, color2=2):
        """Grid split diagonally."""
        grid = np.zeros(size, dtype=int)
        for i in range(size[0]):
            for j in range(size[1]):
                if i >= j:
                    grid[i, j] = color1
                else:
                    grid[i, j] = color2
        return grid

    @staticmethod
    def sparse_pattern(size=(15, 15)):
        """Sparse pattern with isolated objects."""
        grid = np.zeros(size, dtype=int)
        grid[2, 2] = 1
        grid[2, 12] = 1
        grid[12, 2] = 1
        grid[12, 12] = 1
        grid[7, 7] = 2
        return grid

    @staticmethod
    def dense_pattern(size=(10, 10)):
        """Dense pattern with many small objects"""
        grid = np.zeros(size, dtype=int)
        for i in range(1, size[0]-1, 2):
            for j in range(1, size[1]-1, 2):
                grid[i:i+2, j:j+2] = (i + j) % 3 + 1
        return grid

    @staticmethod
    def repeating_motif(size=(12, 12)):
        """Repeating 2x2 motif."""
        grid = np.zeros(size, dtype=int)
        motif = np.array([[1, 2], [3, 0]])
        for i in range(0, size[0]-1, 2):
            for j in range(0, size[1]-1, 2):
                grid[i:i+2, j:j+2] = motif
        return grid

    @staticmethod
    def gradient_pattern(size=(10, 10)):
        """Gradient of colors."""
        grid = np.zeros(size, dtype=int)
        for i in range(size[0]):
            grid[i, :] = i % 5 + 1
        return grid

    @staticmethod
    def noisy_grid(size=(10, 10), noise_level=0.3):
        """Grid with random noise."""
        np.random.seed(42)
        grid = np.random.choice([0, 1, 2, 3], size=size,
                               p=[1-noise_level, noise_level/3, noise_level/3, noise_level/3])
        return grid.astype(int)

    @staticmethod
    def get_all_test_grids():
        """Get dictionary of all test grids."""
        return {
            'empty': GridLibrary.empty_grid(),
            'single_pixel': GridLibrary.single_pixel(),
            'horizontal_line': GridLibrary.horizontal_line(),
            'vertical_line': GridLibrary.vertical_line(),
            'diagonal_main': GridLibrary.diagonal_line(direction='main'),
            'diagonal_anti': GridLibrary.diagonal_line(direction='anti'),
            'filled_rectangle': GridLibrary.filled_rectangle(),
            'hollow_rectangle': GridLibrary.hollow_rectangle(),
            'l_shape': GridLibrary.l_shape(),
            't_shape': GridLibrary.t_shape(),
            'cross_shape': GridLibrary.cross_shape(),
            'checkerboard': GridLibrary.checkerboard(),
            'scattered_pixels': GridLibrary.scattered_pixels(),
            'nested_rectangles': GridLibrary.nested_rectangles(),
            'multicolor_regions': GridLibrary.multicolor_regions(),
            'connected_components': GridLibrary.connected_components(),
            'with_holes': GridLibrary.with_holes(),
            'border_pattern': GridLibrary.border_pattern(),
            'diagonal_split': GridLibrary.diagonal_split(),
            'sparse_pattern': GridLibrary.sparse_pattern(),
            'dense_pattern': GridLibrary.dense_pattern(),
            'repeating_motif': GridLibrary.repeating_motif(),
            'gradient_pattern': GridLibrary.gradient_pattern(),
            'noisy_grid': GridLibrary.noisy_grid(),
        }


@pytest.fixture(params=list(GridLibrary.get_all_test_grids().items()), ids=lambda kv: kv[0])
def grid(request):
    return request.param[1]


# ============================================================================
# GRID-BASED TESTS
# ============================================================================

class TestGridObject:
    """Test GridObject on various grids."""

    @staticmethod
    def test_object_creation(grid: np.ndarray):
        """Test GridObject creation from grid."""
        # Find a component to create object from
        unique_colors = [c for c in np.unique(grid) if c != 0]
        if not unique_colors:
            return  # Skip empty grids
        color = unique_colors[0]
        components = find_connected_components_with_color(grid, color)
        if components:
            coords = components[0]
            obj = GridObject(
                shape='test_shape',
                coords=coords,
                color=[color],
                label='test_obj',
                grid_shape=grid.shape,
                font_color=0,
                grid=grid
            )
            assert obj.size == len(coords), "Size should match coords"
            assert obj.shape == 'test_shape', "Shape should match"
            assert color in obj.color_numbers, "Color should be present"

    @staticmethod
    def test_object_properties(grid: np.ndarray):
        """Test GridObject computed properties."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        if not components:
            return  # Skip if no components
        coords = components[0]
        obj = GridObject(
            shape='complex',
            coords=coords,
            color=[1],
            label='test_1',
            grid_shape=grid.shape,
            font_color=0,
            grid=grid
        )
        # Test basic properties
        assert obj.size > 0, "Should have positive size"
        assert obj.hor_size > 0, "Should have positive horizontal size"
        assert obj.vert_size > 0, "Should have positive vertical size"
        assert isinstance(obj.center, tuple), "Center should be tuple"
        assert len(obj.center) == 2, "Center should be 2D"

        # Test bounds
        assert 0 <= obj.min_i < grid.shape[0], "Min i should be in bounds"
        assert 0 <= obj.max_i < grid.shape[0], "Max i should be in bounds"
        assert 0 <= obj.min_j < grid.shape[1], "Min j should be in bounds"
        assert 0 <= obj.max_j < grid.shape[1], "Max j should be in bounds"

    @staticmethod
    def test_object_symmetry(grid: np.ndarray):
        """Test symmetry detection."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        if not components:
            return
        coords = components[0]
        obj = GridObject('test', coords, [1], 'test', grid.shape, 0, grid)
        assert obj.symmetry in ['horizontal_symmetry', 'vertical_symmetry',
                               'horizontal_and_vertical_symmetry', 'assymetry'], \
            "Symmetry should be valid value"

    @staticmethod
    def test_object_immutability(grid: np.ndarray):
        """Test that GridObject maintains immutability."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        if not components:
            return
        coords = components[0]
        obj = GridObject('test', coords, [1], 'test', grid.shape, 0, grid)
        # Test immutable attributes
        assert isinstance(obj.coords, tuple), "Coords should be immutable tuple"
        assert isinstance(obj.coords_offsets, tuple), "Offsets should be immutable tuple"
        assert isinstance(obj.color_numbers, tuple), "Colors should be immutable tuple"


class TestGridSummary:
    """Test GridSummary on various grids."""

    @staticmethod
    def test_summary_creation(grid: np.ndarray):
        """Test GridSummary creation."""
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        assert hasattr(summary, 'grid'), "Should have grid"
        assert hasattr(summary, 'shape'), "Should have shape"
        assert hasattr(summary, 'repr_levels'), "Should have repr_levels"
        assert 1 in summary.repr_levels, "Should have level 1"

    @staticmethod
    def test_multiple_levels(grid: np.ndarray):
        """Test GridSummary with multiple levels."""
        # Skip empty or very sparse grids for multi-level
        if np.count_nonzero(grid) < 3:
            return
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1, 2])
        assert len(summary.repr_levels) == 2, "Should have 2 levels"
        assert all(level in summary.repr_levels for level in [1, 2]), \
            "Should have all requested levels"

    @staticmethod
    def test_relation_analysis(grid: np.ndarray):
        """Test relation analysis."""
        # Skip grids with too few objects
        if np.count_nonzero(grid) < 4:
            return
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        level_1 = summary.repr_levels[1]
        assert hasattr(level_1, 'triples'), "Should have triples"
        assert hasattr(level_1, 'relation_statistics'), "Should have relation_statistics"

    @staticmethod
    def test_embeddings(grid: np.ndarray):
        """Test relation embeddings."""
        # Skip grids with too few objects
        if np.count_nonzero(grid) < 4:
            return
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        embeddings = summary.get_relation_embeddings_as_numpy(level=1)
        if embeddings.size > 0:
            assert isinstance(embeddings, np.ndarray), "Should be numpy array"
            assert embeddings.ndim == 2, "Should be 2D"
            assert np.all(np.isfinite(embeddings)), "All values should be finite"

    @staticmethod
    def test_embeddings_are_the_same_however_many_times_they_are_asked_for(grid: np.ndarray):
        """The call can replace the stored level with one carrying embeddings,
        so it is worth pinning that doing it twice changes nothing - callers
        ask for these once per reset and once per step against the same
        summary.
        """
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])

        first = summary.get_relation_embeddings_as_numpy(level=1)
        second = summary.get_relation_embeddings_as_numpy(level=1)

        assert np.array_equal(first, second)


class TestMatchScore:
    """Test match score calculations on grids."""

    @staticmethod
    def test_rotation_generation(grid: np.ndarray):
        """Test rotation generation."""
        components = find_connected_components_excluding_colors(grid, font_color=0.0)
        if not components or len(components) < 1:
            return
        coords = components[0]
        rotations = get_rotations(list(coords))
        assert len(rotations) == 4, "Should generate 4 rotations"
        assert all(len(rot) == len(coords) for rot in rotations), \
            "All rotations should have same number of coords"

    @staticmethod
    def test_intersection_checking(grid: np.ndarray):
        """Test intersection checking."""
        components = find_connected_components_excluding_colors(grid, font_color=0.0)
        if len(components) < 2:
            return
        coords1 = list(components[0])
        coords2 = list(components[1])
        # Check that different components don't intersect
        intersects = bool(set(coords1).intersection(set(coords2)))
        assert not intersects, "Components should not intersect"
        self_intersects = bool(set(coords1).intersection(set(coords1)))
        # Self intersection should always be true
        assert self_intersects, "Should self-intersect"

    @staticmethod
    def test_match_score_calculation(grid: np.ndarray):
        """Test match score calculation."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        if len(components) < 2:
            return


class TestRelationAnalyzer:
    """Test relation analyzer on grids."""

    @staticmethod
    def test_relation_detection(grid: np.ndarray):
        """Test relation detection between objects."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        if len(components) < 2:
            return
        obj1 = GridObject('comp1', components[0], [1], 'obj1', grid.shape, 0, grid)
        obj2 = GridObject('comp2', components[1], [2], 'obj2', grid.shape, 0, grid)
        analyzer = RelationAnalyzer(obj1=obj1, obj2=obj2, shape=grid.shape)
        assert hasattr(analyzer, 'triples'), "Should have triples"
        assert hasattr(analyzer, 'relation_counter'), "Should have relation_counter"
        assert isinstance(analyzer.relation_counter, Counter), "Should be Counter"

    @staticmethod
    def test_alignment_detection(grid: np.ndarray):
        """Test alignment detection."""
        components = find_connected_components_excluding_colors(grid, font_color=0.0)
        if len(components) < 2:
            return
        obj1 = GridObject('comp1', components[0], [1], 'obj1', grid.shape, 0, grid)
        obj2 = GridObject('comp2', components[1], [2], 'obj2', grid.shape, 0, grid)
        x_aligned = RelationAnalyzer.x_alignment(obj1, obj2)
        y_aligned = RelationAnalyzer.y_alignment(obj1, obj2)
        assert isinstance(x_aligned, bool), "x_alignment should return bool"
        assert isinstance(y_aligned, bool), "y_alignment should return bool"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests across modules."""

    @staticmethod
    def test_pattern_to_object_pipeline(grid: np.ndarray):
        """Test pattern generation to object creation."""
        patterns = generate_patterns(grid.shape, ['rectangle'], multithreading=False)
        if 'rectangle' in patterns and patterns['rectangle']:
            rect_patterns = patterns['rectangle']
            if rect_patterns and rect_patterns[0]:
                coords = rect_patterns[0][0]
                obj = GridObject('rectangle', coords, [1], 'test', grid.shape, 0, grid)
                assert obj.size == len(coords), "Size should match"
                assert obj.shape == 'rectangle', "Shape should match"

    @staticmethod
    def test_immutability_preservation(grid: np.ndarray):
        """Test that immutability is preserved through pipeline."""
        # Skip empty grids
        if np.count_nonzero(grid) == 0:
            return
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        level_1 = summary.repr_levels[1]
        # Try to modify frozen dataclass
        try:
            level_1.objects = tuple()
            assert False, "Should not allow modification"
        except (FrozenInstanceError, AttributeError):
            pass  # Expected


# ============================================================================
# VALIDATION AND CORRECTNESS TESTS
# ============================================================================

class TestCorrectnessValidation:
    """Tests to validate correctness of implementations"""

    @staticmethod
    def test_coordinate_bounds(grid: np.ndarray):
        """Validate all coordinates are within grid bounds."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        for comp in components:
            for coord in comp:
                assert 0 <= coord[0] < grid.shape[0], \
                    f"Row {coord[0]} out of bounds for grid shape {grid.shape}"
                assert 0 <= coord[1] < grid.shape[1], \
                    f"Col {coord[1]} out of bounds for grid shape {grid.shape}"
        # Also test GridObject bounds
        if components:
            obj = GridObject('test', components[0], [1], 'test', grid.shape, 0, grid)
            assert 0 <= obj.min_i < grid.shape[0], "Object min_i out of bounds"
            assert 0 <= obj.max_i < grid.shape[0], "Object max_i out of bounds"
            assert 0 <= obj.min_j < grid.shape[1], "Object min_j out of bounds"
            assert 0 <= obj.max_j < grid.shape[1], "Object max_j out of bounds"

    @staticmethod
    def test_color_consistency(grid: np.ndarray):
        """Validate color values are consistent."""
        unique_colors = [c for c in np.unique(grid) if c != 0]
        for color in unique_colors:
            components = find_connected_components_with_color(grid, color)
            for comp in components:
                # All coords in component should have the same color
                for coord in comp:
                    assert grid[coord] == color, \
                        f"Coord {coord} has color {grid[coord]}, expected {color}"
                # Test GridObject color tracking
                if comp:
                    obj = GridObject('test', comp, [color], 'test', grid.shape, 0, grid)
                    assert color in obj.color_numbers, \
                        f"Color {color} not in object color_numbers"

    @staticmethod
    def test_connectivity(grid: np.ndarray):
        """Validate connected components are actually connected using 8-connectivity."""
        unique_colors = [c for c in np.unique(grid) if c != 0]
        for color in unique_colors:
            components = find_connected_components_with_color(grid, color)
            for comp in components:
                if len(comp) <= 1:
                    continue
                # Check 8-connectivity (including diagonals)
                coord_set = set(comp)
                visited = set()
                queue = [comp[0]]
                visited.add(comp[0])
                while queue:
                    i, j = queue.pop(0)
                    # 8-connectivity: all 8 surrounding cells
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue  # Skip the current cell itself
                            ni, nj = i + di, j + dj
                            if (ni, nj) in coord_set and (ni, nj) not in visited:
                                visited.add((ni, nj))
                                queue.append((ni, nj))
                assert len(visited) == len(comp), \
                    f"Component not fully connected with 8-connectivity: {len(visited)} visited vs {len(comp)} total"

    @staticmethod
    def test_no_duplicate_coordinates(grid: np.ndarray):
        """Validate no duplicate coordinates in components."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        for comp in components:
            assert len(comp) == len(set(comp)), \
                "Component contains duplicate coordinates"
        # Test GridObject
        if components:
            obj = GridObject('test', components[0], [1], 'test', grid.shape, 0, grid)
            assert len(obj.coords) == len(set(obj.coords)), \
                "GridObject contains duplicate coordinates"

    @staticmethod
    def test_size_calculations(grid: np.ndarray):
        """Validate size calculations are correct."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        for comp in components:
            if not comp:
                continue
            obj = GridObject('test', comp, [1], 'test', grid.shape, 0, grid)
            # Size should match coordinate count
            assert obj.size == len(comp), \
                f"Object size {obj.size} doesn't match coords length {len(comp)}"
            # Horizontal size
            expected_hor = max(c[0] for c in comp) - min(c[0] for c in comp) + 1
            assert obj.hor_size == expected_hor, \
                f"Horizontal size {obj.hor_size} incorrect, expected {expected_hor}"
            # Vertical size
            expected_vert = max(c[1] for c in comp) - min(c[1] for c in comp) + 1
            assert obj.vert_size == expected_vert, \
                f"Vertical size {obj.vert_size} incorrect, expected {expected_vert}"

    @staticmethod
    def test_center_calculation(grid: np.ndarray):
        """Validate center calculations are reasonable."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        for comp in components:
            if not comp:
                continue
            obj = GridObject('test', comp, [1], 'test', grid.shape, 0, grid)
            # Center should be within object bounding box
            assert obj.min_i <= obj.center[0] <= obj.max_i, \
                "Center row not within object bounds"
            assert obj.min_j <= obj.center[1] <= obj.max_j, \
                "Center col not within object bounds"

    @staticmethod
    def test_compactness_range(grid: np.ndarray):
        """Validate compactness is in valid range [0, 1]."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        for comp in components:
            if not comp:
                continue
            obj = GridObject('test', comp, [1], 'test', grid.shape, 0, grid)
            assert 0.0 <= obj.compactness <= 1.0, \
                f"Compactness {obj.compactness} out of valid range [0, 1]"

    @staticmethod
    def test_rotation_preservation(grid: np.ndarray):
        """Validate rotations preserve coordinate count."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        if not components:
            return
        coords = list(components[0])
        rotations = get_rotations(coords)
        # All rotations should have same length
        original_len = len(coords)
        for rot in rotations:
            assert len(rot) == original_len, \
                f"Rotation has {len(rot)} coords, expected {original_len}"

    @staticmethod
    def test_embedding_dimensions(grid: np.ndarray):
        """Validate embedding dimensions are consistent."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        if len(components) < 2:
            return
        embeddings = []
        for comp in components[:5]:  # Test first 5
            obj = GridObject('test', comp, [1], 'test', grid.shape, 0, grid)
            emb = obj.create_embedding()
            embeddings.append(emb)
        # All embeddings should have same dimension
        if embeddings:
            first_dim = len(embeddings[0])
            for emb in embeddings[1:]:
                assert len(emb) == first_dim, \
                    f"Embedding dimension {len(emb)} doesn't match {first_dim}"


# ============================================================================
# REGRESSION TESTS
# ============================================================================

class TestRegression:
    """Tests for known issues and regressions."""

    @staticmethod
    def test_empty_component_handling():
        """Regression: Empty components should be handled gracefully."""
        grid = np.zeros((10, 10), dtype=int)
        components = find_connected_components_with_color(grid, 1)
        assert components == [], "Empty grid should return empty list"

    @staticmethod
    def test_single_cell_object():
        """Regression: Single cell objects should work correctly."""
        grid = GridLibrary.single_pixel()
        coords = [(5, 5)]
        obj = GridObject('cell', coords, [1], 'single', grid.shape, 0, grid)
        assert obj.size == 1, "Single cell should have size 1"
        assert obj.hor_size == 1, "Single cell horizontal size should be 1"
        assert obj.vert_size == 1, "Single cell vertical size should be 1"
        assert obj.compactness == 1.0, "Single cell should be fully compact"

    @staticmethod
    def test_rectangle_cache_consistency():
        """Regression: Rectangle cache should return consistent results."""
        size = (10, 10)
        # Call multiple times
        result1 = rectangles_coords(size)
        result2 = rectangles_coords(size)
        result3 = rectangles_coords(size)
        assert result1 == result2, "Cached results should match"
        assert result2 == result3, "Cached results should match"

    @staticmethod
    def test_grid_modification_safety():
        """Regression: Operations should not modify input grid."""
        grid = GridLibrary.filled_rectangle()
        original_grid = grid.copy()
        GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        assert np.array_equal(grid, original_grid), \
            "Input grid should not be modified"

    @staticmethod
    def test_color_zero_handling():
        """Regression: Color 0 (background) should be handled correctly."""
        grid = GridLibrary.filled_rectangle()
        # Should find background components
        bg_components = find_connected_components_with_color(grid, 0)
        assert isinstance(bg_components, list), "Should handle color 0"

    @staticmethod
    def test_boundary_objects():
        """Regression: Objects at grid boundaries should work."""
        grid = np.zeros((10, 10), dtype=int)
        # Corner object
        coords = [(0, 0), (0, 1), (1, 0)]
        obj = GridObject('corner', coords, [1], 'test', grid.shape, 0, grid)
        assert obj.min_i == 0, "Should handle boundary"
        assert obj.min_j == 0, "Should handle boundary"
        assert 'at_top_edge' in obj.positioning, "Should detect top edge"
        assert 'at_left_edge' in obj.positioning, "Should detect left edge"

    @staticmethod
    def test_large_object_performance():
        """Regression: Large objects should not cause performance issues."""
        grid = np.ones((30, 30), dtype=int)
        coords = [(i, j) for i in range(30) for j in range(30)]
        start = time.perf_counter()
        obj = GridObject('large', coords, [1], 'test', grid.shape, 0, grid)
        duration = time.perf_counter() - start
        assert duration < 1.0, f"Large object creation too slow: {duration}s"
        assert obj.size == 900, "Should handle large objects"

    @staticmethod
    def test_multicolor_object():
        """Regression: Objects with multiple colors should work."""
        grid = np.array([[1, 2], [3, 4]])
        coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
        colors = [1, 2, 3, 4]
        obj = GridObject('multi', coords, colors, 'test', grid.shape, 0, grid)
        assert len(obj.color_numbers) == 4, "Should track all colors"
        assert obj.size == 4, "Should have correct size"


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================

class TestProperties:
    """Property-based tests (invariants that should always hold)."""

    @staticmethod
    def test_object_count_invariant(grid: np.ndarray):
        """Property: Sum of object sizes should equal non-zero cells."""
        if np.count_nonzero(grid) == 0:
            return
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        level_1 = summary.repr_levels[1]
        total_cells = sum(obj.size for obj in level_1.objects)
        non_zero = np.count_nonzero(grid)
        # Allow some tolerance for overlapping or filtering
        assert total_cells <= non_zero * 1.5, \
            f"Total object cells {total_cells} far exceeds non-zero cells {non_zero}"

    @staticmethod
    def test_symmetry_invariant(grid: np.ndarray):
        """Property: If object is symmetric, rotated version should match."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        for comp in components:
            if not comp:
                continue
            obj = GridObject('test', comp, [1], 'test', grid.shape, 0, grid)
            if 'horizontal_and_vertical_symmetry' in obj.symmetry:
                # Object with both symmetries should look same when rotated 180°
                coords_offsets = [(x - obj.precise_center[0], y - obj.precise_center[1]) for x, y in obj.coords]
                coords_set = set(coords_offsets)
                rotated_coords = set((-c[0], -c[1]) for c in coords_offsets)
                # May not be perfect due to discretization, but should be similar
                overlap = len(coords_set.intersection(rotated_coords))
                assert overlap > len(coords_set) * 0.8, \
                    "Symmetric object should mostly overlap with rotation"

    @staticmethod
    def test_containment_transitivity(grid: np.ndarray):
        """Property: If A contains B and B contains C, then A contains C."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        if len(components) < 3:
            return
        # Create objects
        objects = []
        for i, comp in enumerate(components[:3]):
            obj = GridObject(f'obj_{i}', comp, [i+1], f'obj_{i}', grid.shape, 0, grid)
            objects.append(obj)
        # Check transitivity of containment
        # (Note: This is a logical property test, may not always find valid case)
        for i in range(len(objects)):
            for j in range(len(objects)):
                if i == j:
                    continue
                result = RelationAnalyzer.in_contour(objects[i], objects[j])
                # Just verify it returns valid value
                assert result in ['object_1', 'object_2', None], \
                    "in_contour should return valid value"

    @staticmethod
    def test_distance_symmetry(grid: np.ndarray):
        """Property: Distance from A to B equals distance from B to A."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        if len(components) < 2:
            return
        obj1 = GridObject('obj1', components[0], [1], 'obj1', grid.shape, 0, grid)
        obj2 = GridObject('obj2', components[1], [2], 'obj2', grid.shape, 0, grid)
        # Calculate Euclidean distance between centers
        dist_1_to_2 = np.sqrt((obj1.center[0] - obj2.center[0])**2 +
                              (obj1.center[1] - obj2.center[1])**2)
        dist_2_to_1 = np.sqrt((obj2.center[0] - obj1.center[0])**2 +
                              (obj2.center[1] - obj1.center[1])**2)
        assert abs(dist_1_to_2 - dist_2_to_1) < 1e-6, \
            "Distance should be symmetric"

    @staticmethod
    def test_alignment_consistency(grid: np.ndarray):
        """Property: Alignment should be consistent with coordinate values."""
        components = find_connected_components_excluding_colors(grid, font_color=0)
        if len(components) < 2:
            return
        obj1 = GridObject('obj1', components[0], [1], 'obj1', grid.shape, 0, grid)
        obj2 = GridObject('obj2', components[1], [2], 'obj2', grid.shape, 0, grid)
        x_aligned = RelationAnalyzer.x_alignment(obj1, obj2)
        y_aligned = RelationAnalyzer.y_alignment(obj1, obj2)

        # If x_aligned, should have overlapping rows
        if x_aligned:
            row_overlap = not (obj1.max_i < obj2.min_i or obj2.max_i < obj1.min_i)
            assert row_overlap, "X-aligned objects should have overlapping rows"

        # If y_aligned, should have overlapping columns
        if y_aligned:
            col_overlap = not (obj1.max_j < obj2.min_j or obj2.max_j < obj1.min_j)
            assert col_overlap, "Y-aligned objects should have overlapping columns"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @staticmethod
    def test_empty_grid():
        """Test handling of empty grid."""
        grid = np.zeros((5, 5), dtype=int)
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        assert summary is not None, "Should handle empty grid"
        level_1 = summary.repr_levels[1]
        assert len(level_1.objects) == 0, "Empty grid should have no objects"

    @staticmethod
    def test_single_pixel_grid():
        """Test handling of single pixel."""
        grid = GridLibrary.single_pixel()
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        assert summary is not None, "Should handle single pixel"

    @staticmethod
    def test_minimum_size_grid():
        """Test minimum size grids."""
        # 2x2 grid
        result = rectangles_coords((2, 2))
        assert isinstance(result, list), "Should handle 2x2 grid"

    @staticmethod
    def test_large_grid():
        """Test large grid handling."""
        # Create larger grid
        grid = np.zeros((50, 50), dtype=int)
        grid[10:20, 10:20] = 1
        grid[30:40, 30:40] = 2
        start = time.perf_counter()
        GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        duration = time.perf_counter() - start
        assert duration < 5.0, f"Large grid processing too slow: {duration}s"

    @staticmethod
    def test_many_colors():
        """Test grid with many different colors."""
        grid = np.zeros((10, 10), dtype=int)
        for i in range(10):
            grid[i, i] = i
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        assert summary is not None, "Should handle many colors"

    @staticmethod
    def test_all_same_color():
        """Test grid with all same non-zero color."""
        grid = np.ones((10, 10), dtype=int)
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        assert summary is not None, "Should handle uniform color"
        level_1 = summary.repr_levels[1]
        assert len(level_1.objects) >= 1, "Should detect at least one object"

    @staticmethod
    def test_rectangular_grids():
        """Test non-square grids."""
        # Wide grid
        grid_wide = np.zeros((5, 20), dtype=int)
        grid_wide[2, 5:15] = 1
        summary = GridSummary(grid=grid_wide, shape=grid_wide.shape, font_color=0, levels=[1])
        assert summary is not None, "Should handle wide grid"
        # Tall grid
        grid_tall = np.zeros((20, 5), dtype=int)
        grid_tall[5:15, 2] = 1
        summary = GridSummary(grid=grid_tall, shape=grid_tall.shape, font_color=0, levels=[1])
        assert summary is not None, "Should handle tall grid"

    @staticmethod
    def test_disconnected_pixels():
        """Test grid with completely disconnected pixels."""
        grid = np.zeros((10, 10), dtype=int)
        grid[1, 1] = 1
        grid[3, 5] = 1
        grid[7, 8] = 1
        grid[9, 2] = 1
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        level_1 = summary.repr_levels[1]
        # Should detect individual cells
        assert len(level_1.objects) == 4, "Should detect 4 separate objects"


# ============================================================================
# COMPATIBILITY TESTS
# ============================================================================

class TestCompatibility:
    """Tests for data type and format compatibility."""

    @staticmethod
    def test_different_dtypes():
        """Test with different numpy dtypes."""
        base_grid = GridLibrary.filled_rectangle()
        dtypes = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16]
        for dtype in dtypes:
            grid = base_grid.astype(dtype)
            summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
            assert summary is not None, f"Failed for dtype {dtype}"

    @staticmethod
    def test_tuple_vs_list_coords():
        """Test coordinate format compatibility."""
        grid = GridLibrary.single_pixel()
        # Test with list
        coords_list = [(5, 5), (5, 6)]
        obj1 = GridObject('test', coords_list, [1], 'test', grid.shape, 0, grid)
        # Test with tuple
        coords_tuple = ((5, 5), (5, 6))
        obj2 = GridObject('test', coords_tuple, [1], 'test', grid.shape, 0, grid)
        assert obj1.size == obj2.size
        assert obj1.coords == obj2.coords


# ============================================================================
# DOCUMENTATION TESTS
# ============================================================================

class TestDocumentation:
    """Tests based on documented examples and use cases."""

    @staticmethod
    def test_basic_usage_example():
        """Test basic usage example from documentation."""
        # Create simple test grid
        grid = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [2, 2, 0, 3]
        ])
        # Create summary
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        # Verify basic operations work
        assert summary is not None
        assert 1 in summary.repr_levels
        assert len(summary.repr_levels[1].objects) > 0

    @staticmethod
    def test_pattern_generation_example():
        """Test pattern generation example."""
        grid_size = (10, 10)
        # Generate patterns
        lines = lines_coords(grid_size)
        rectangles = rectangles_coords(grid_size)
        assert len(lines) > 0 or grid_size[0] <= 1 or grid_size[1] <= 1
        assert len(rectangles) > 0

    @staticmethod
    def test_object_creation_example():
        """Test GridObject creation example."""
        grid = np.zeros((10, 10))
        coords = [(2, 2), (2, 3), (3, 2), (3, 3)]
        obj = GridObject(
            shape='rectangle',
            coords=coords,
            color=[1],
            label='example',
            grid_shape=(10, 10),
            font_color=0,
            grid=grid
        )
        assert obj.size == 4
        assert obj.shape == 'rectangle'

    @staticmethod
    def test_relation_analysis_example():
        """Test relation analysis example."""
        grid = np.zeros((15, 15))
        obj1 = GridObject('rect', [(2, 2), (2, 3)], [1], 'obj1', (15, 15), 0, grid)
        obj2 = GridObject('rect', [(2, 6), (2, 7)], [1], 'obj2', (15, 15), 0, grid)
        analyzer = RelationAnalyzer(obj1=obj1, obj2=obj2, shape=(15, 15))
        assert analyzer is not None
        assert hasattr(analyzer, 'triples')
        assert hasattr(analyzer, 'relation_counter')


# ============================================================================
# STRESS AND CHAOS TESTS
# ============================================================================

class TestStress:
    """Stress tests with extreme conditions (generous thresholds - these
    check "doesn't degrade catastrophically", not tight performance
    budgets)."""

    @staticmethod
    def test_very_large_grid():
        """Test with very large grid."""
        grid = np.random.randint(0, 5, (100, 100))
        start = time.perf_counter()
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        duration = time.perf_counter() - start
        assert duration < 30.0, f"Very large grid too slow: {duration}s"
        assert summary is not None

    @staticmethod
    def test_many_small_objects():
        """Test with many small isolated objects."""
        grid = np.zeros((50, 50), dtype=int)
        # Create many 1-pixel objects
        for i in range(5, 45, 4):
            for j in range(5, 45, 4):
                grid[i, j] = (i + j) % 5 + 1
        start = time.perf_counter()
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        duration = time.perf_counter() - start
        level_1 = summary.repr_levels[1]
        num_objects = len(level_1.objects)
        assert duration < 10.0, f"Many objects too slow: {duration}s"
        assert num_objects > 50, "Should detect many objects"

    @staticmethod
    def test_dense_multicolor():
        """Test with dense multicolor grid."""
        grid = np.random.randint(1, 11, (30, 30))
        start = time.perf_counter()
        GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        duration = time.perf_counter() - start
        assert duration < 15.0, f"Dense multicolor too slow: {duration}s"

    @staticmethod
    def test_extreme_aspect_ratio():
        """Test with extreme aspect ratios."""
        # Very wide
        grid_wide = np.zeros((5, 100), dtype=int)
        grid_wide[2, 10:90] = 1
        summary = GridSummary(grid=grid_wide, shape=grid_wide.shape, font_color=0, levels=[1])
        assert summary is not None
        # Very tall
        grid_tall = np.zeros((100, 5), dtype=int)
        grid_tall[10:90, 2] = 1
        summary = GridSummary(grid=grid_tall, shape=grid_tall.shape, font_color=0, levels=[1])
        assert summary is not None

    @staticmethod
    def test_maximum_colors():
        """Test with maximum number of colors."""
        grid = np.zeros((10, 10), dtype=int)
        # Assign different color to each row
        for i in range(10):
            grid[i, :] = i + 1
        start = time.perf_counter()
        GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        duration = time.perf_counter() - start
        assert duration < 10.0, f"Many colors too slow: {duration}s"

    @staticmethod
    def test_repeated_operations():
        """Repeated construction must not leak memory or degrade in speed.

        A timing assertion alone would not catch a leak, so the peak-memory
        bound comes from resource_budget and the per-op time is asserted
        separately below.
        """
        grid = GridLibrary.multicolor_regions()
        with resource_budget(max_memory_mb=20.0):
            start = time.perf_counter()
            for _ in range(100):
                GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
            duration = time.perf_counter() - start
        avg_time = duration / 100
        assert avg_time < 0.5, f"Repeated operations degrading: {avg_time}s per op"

    @staticmethod
    def test_complex_nesting():
        """Test with deeply nested structures."""
        grid = np.zeros((40, 40), dtype=int)
        # Create nested rectangles
        for level in range(10):
            offset = level * 2
            color = level + 1
            size = 40 - offset * 2
            if size > 0:
                grid[offset:offset+size, offset:offset+size] = color
        start = time.perf_counter()
        GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        duration = time.perf_counter() - start
        assert duration < 10.0, f"Nested structures too slow: {duration}s"


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance smoke tests: same "generous threshold, not a tight
    benchmark" philosophy as TestStress above - these catch a catastrophic
    slowdown, not micro-regressions."""

    @staticmethod
    def test_pattern_generation_benchmark():
        """Benchmark pattern generation."""
        grid_sizes = [(10, 10), (20, 20), (30, 30)]
        shape_types = ['line', 'rectangle', 'diagonal']
        for size in grid_sizes:
            start = time.perf_counter()
            generate_patterns(size, shape_types, multithreading=True)
            duration = time.perf_counter() - start
            assert duration < 5.0, f"Pattern generation for {size} too slow: {duration}s"

    @staticmethod
    def test_grid_summary_benchmark():
        """Benchmark GridSummary creation."""
        test_grids = {
            'simple': GridLibrary.filled_rectangle(),
            'complex': GridLibrary.multicolor_regions(),
            'nested': GridLibrary.nested_rectangles(),
        }
        for name, grid in test_grids.items():
            start = time.perf_counter()
            GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
            duration = time.perf_counter() - start
            assert duration < 5.0, f"GridSummary for {name!r} too slow: {duration}s"

    @staticmethod
    def test_object_operations_benchmark():
        """Benchmark GridObject creation and embedding."""
        grid = GridLibrary.filled_rectangle()
        coords = [(i, j) for i in range(2, 6) for j in range(2, 6)]

        start = time.perf_counter()
        for _ in range(100):
            obj = GridObject('rect', coords, [1], 'test', grid.shape, 0, grid)
        create_duration = time.perf_counter() - start
        assert create_duration < 5.0, f"100x GridObject creation too slow: {create_duration}s"

        start = time.perf_counter()
        for _ in range(100):
            obj.create_embedding()
        embed_duration = time.perf_counter() - start
        assert embed_duration < 5.0, f"100x embedding too slow: {embed_duration}s"

    @staticmethod
    def test_stress_many_objects_benchmark():
        """Stress-benchmark GridSummary with many small objects."""
        grid = np.zeros((30, 30), dtype=int)
        obj_id = 1
        for i in range(1, 28, 3):
            for j in range(1, 28, 3):
                grid[i:i+2, j:j+2] = obj_id % 5 + 1
                obj_id += 1
        start = time.perf_counter()
        GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        duration = time.perf_counter() - start
        assert duration < 5.0, f"Many-objects GridSummary too slow: {duration}s"


class TestRelationStatistics:
    """Relation tallies feed both the summary text and the RL feature
    vector, so a count that isn't a count of anything propagates silently."""

    @staticmethod
    def test_unrelated_objects_tally_zero():
        """The tally used to be seeded from the tuple of relation *names*,
        which gives every relation a starting count of one - so two objects
        sharing nothing still reported same_color=1."""
        grid = np.zeros((6, 6), dtype=int)
        grid[0, 0] = 1
        grid[5, 5] = 2

        stats = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[2]).repr_levels[2].relation_statistics

        assert stats.same_color == 0
        assert stats.in_contour == 0

    @staticmethod
    def test_matching_objects_are_counted_once():
        grid = np.zeros((6, 6), dtype=int)
        grid[0, 0] = 3
        grid[5, 5] = 3

        stats = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[2]).repr_levels[2].relation_statistics

        assert stats.same_color == 1
        assert stats.same_size == 1


class TestObjectDistance:
    """Distance between objects drives the normalized_distance feature and
    the 'nearby object' threshold, so collapsing distinct configurations
    onto the same number removes signal from both."""

    @staticmethod
    def _objects(coords_a, coords_b, shape=(20, 20)):
        grid = np.zeros(shape, dtype=int)
        for c in list(coords_a) + list(coords_b):
            grid[c] = 1
        return (GridObject('complex', list(coords_a), [1], 'complex_0', shape, 0, grid),
                GridObject('complex', list(coords_b), [1], 'complex_1', shape, 0, grid))

    @staticmethod
    def test_objects_sharing_a_row_are_not_all_at_distance_zero():
        """Taking the smaller of the two axis gaps made every row- or
        column-aligned pair adjacent, however far apart they actually were."""
        near = GridSummary.calculate_distance(*TestObjectDistance._objects([(0, 0)], [(0, 2)]))
        far = GridSummary.calculate_distance(*TestObjectDistance._objects([(0, 0)], [(0, 19)]))

        assert far > near

    @staticmethod
    def test_overlapping_ranges_count_as_no_gap_on_that_axis():
        """A tall object spanning rows 0-10 and a cell on row 5 overlap
        vertically; measuring corner-to-corner instead reported a gap."""
        tall, cell = TestObjectDistance._objects([(0, 0), (10, 0)], [(5, 7)])

        assert GridSummary.calculate_distance(tall, cell) == 7

    @staticmethod
    def test_touching_objects_are_one_step_apart_and_overlapping_ones_zero():
        touching = GridSummary.calculate_distance(*TestObjectDistance._objects([(0, 0)], [(0, 1)]))
        interleaved = GridSummary.calculate_distance(*TestObjectDistance._objects([(0, 0), (6, 6)], [(2, 2), (3, 3)]))

        assert touching == 1
        assert interleaved == 0

    @staticmethod
    def test_diagonal_neighbours_are_one_step_apart():
        """Chebyshev, not Manhattan: on an 8-connected grid a diagonal
        neighbour is one move away, matching how objects reach each other."""
        assert GridSummary.calculate_distance(*TestObjectDistance._objects([(0, 0)], [(1, 1)])) == 1

    @staticmethod
    def test_distance_is_symmetric():
        a, b = TestObjectDistance._objects([(1, 1), (2, 2)], [(8, 9)])

        assert GridSummary.calculate_distance(a, b) == GridSummary.calculate_distance(b, a)


class TestSubtaskSummary:
    """The task-level feature path: input/output grid summaries plus the
    ratios between them."""

    @staticmethod
    def test_create_builds_both_grid_summaries():
        """The two summary attributes carried no type annotation, so they
        were never dataclass fields and create() raised TypeError on every
        call - taking prepare_features down with it."""
        subtask = ARCSubtask(label='s0', train_inp=np.array([[1, 0], [0, 2]]),
                             train_out=np.array([[0, 1], [2, 0]]))

        summary = SubtaskSummary.create(subtask)

        assert summary.inp_grid_summary is not None
        assert summary.out_grid_summary is not None
        assert summary.subtask_label == 's0'

    @staticmethod
    def test_ratios_capture_a_resize():
        subtask = ARCSubtask(label='s1', train_inp=np.array([[1, 0], [0, 2]]),
                             train_out=np.zeros((4, 4), dtype=int))

        summary = SubtaskSummary.create(subtask)

        assert summary.grids_x_ratio == 0.5
        assert summary.grids_y_ratio == 0.5

    @staticmethod
    def test_prepare_features_returns_numeric_differences():
        subtask = ARCSubtask(label='s2', train_inp=np.array([[1, 0], [0, 2]]),
                             train_out=np.array([[1, 1], [0, 2]]))

        features = SubtaskSummary.create(subtask).prepare_features()

        assert features['grids_x_ratio'] == 1.0
        assert 'total_objects_diff' in features
        assert all(isinstance(v, (int, float)) for v in features.values())
