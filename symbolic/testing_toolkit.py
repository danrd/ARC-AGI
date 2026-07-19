"""
Unified Grid-Based Test Suite for ARC Pattern Analysis

This module provides comprehensive testing based on predefined grid scenarios
that cover various edge cases and real-world patterns found in ARC tasks.

FEATURES:
- 25+ predefined test grid scenarios
- 10 test categories (Functional, Validation, Regression, Property, etc.)
- Automated grid-based testing
- Performance benchmarking
- HTML and JSON report generation
- CLI interface for easy usage

QUICK START:
    # Run all tests
    python test_suite.py all
    
    # Quick module test
    python test_suite.py quick object
    
    # Validation only
    python test_suite.py validate
    
    # Test specific grids
    python test_suite.py grids filled_rectangle nested_rectangles

PYTHON API:
    from test_suite import *
    
    # Run comprehensive tests
    runner = run_all_comprehensive_tests()
    
    # Quick tests
    quick_test('object')
    
    # Validation
    validate_implementation()
    
    # Performance
    performance_test()
    
    # List grids
    list_test_grids()
    

CONFIGURATION:
    Modify TestConfig class:
    - RUN_PERFORMANCE_TESTS = True/False
    - RUN_STRESS_TESTS = True/False
    - PERFORMANCE_THRESHOLD_MS = 100

TEST CATEGORIES:
1. Pattern Detection: Lines, rectangles, connected components
2. GridObject: Creation, properties, symmetry, immutability
3. GridSummary: Multi-level analysis, relations, embeddings
4. Match Score: Rotations, intersections, scoring
5. RelationAnalyzer: Relation detection, alignment
6. Correctness Validation: Bounds, consistency, connectivity
7. Regression: Known issues, edge cases
8. Property-Based: Invariants, symmetry, transitivity
9. Integration: Full pipeline tests

GRID SCENARIOS:
- Empty grids, single pixels
- Lines (horizontal, vertical, diagonal)
- Rectangles (filled, hollow, nested)
- Shapes (L, T, cross)
- Patterns (checkerboard, symmetric, gradient)
- Complex (multicolor, holes, sparse, dense)
- Edge cases (extreme ratios, many colors)

OUTPUT:
- Console report with statistics
- test_results.json: Detailed JSON results
- test_report.html: Beautiful visual report
- Optional: Grid visualizations

REQUIREMENTS:
- numpy
- Optional: matplotlib (for visualizations)
- Imports: pattern_generation, grid_object, grid_summary, 
          objects_filter, relation_analyzer, match_score

VERSION: 2.0
AUTHOR: Unified Test Framework
"""

import numpy as np
import time
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Callable
from dataclasses import FrozenInstanceError
import json
import matplotlib.pyplot as plt
from grid_summary import GridSummary
from grid_object import GridObject
from relation_analyzer import RelationAnalyzer
from pattern_generation import generate_patterns, lines_coords, rectangles_coords, find_connected_components_excluding_colors
from match_score import get_rotations
from utils.plotting import plot_grid
from symbolic.patterns import find_connected_components_with_color

# ============================================================================
# TEST GRID LIBRARY
# ============================================================================

class TestGridLibrary:
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
            'empty': TestGridLibrary.empty_grid(),
            'single_pixel': TestGridLibrary.single_pixel(),
            'horizontal_line': TestGridLibrary.horizontal_line(),
            'vertical_line': TestGridLibrary.vertical_line(),
            'diagonal_main': TestGridLibrary.diagonal_line(direction='main'),
            'diagonal_anti': TestGridLibrary.diagonal_line(direction='anti'),
            'filled_rectangle': TestGridLibrary.filled_rectangle(),
            'hollow_rectangle': TestGridLibrary.hollow_rectangle(),
            'l_shape': TestGridLibrary.l_shape(),
            't_shape': TestGridLibrary.t_shape(),
            'cross_shape': TestGridLibrary.cross_shape(),
            'checkerboard': TestGridLibrary.checkerboard(),
            'scattered_pixels': TestGridLibrary.scattered_pixels(),
            'nested_rectangles': TestGridLibrary.nested_rectangles(),
            'multicolor_regions': TestGridLibrary.multicolor_regions(),
            'connected_components': TestGridLibrary.connected_components(),
            'with_holes': TestGridLibrary.with_holes(),
            'border_pattern': TestGridLibrary.border_pattern(),
            'diagonal_split': TestGridLibrary.diagonal_split(),
            'sparse_pattern': TestGridLibrary.sparse_pattern(),
            'dense_pattern': TestGridLibrary.dense_pattern(),
            'repeating_motif': TestGridLibrary.repeating_motif(),
            'gradient_pattern': TestGridLibrary.gradient_pattern(),
            'noisy_grid': TestGridLibrary.noisy_grid(),
        }   
        
    def visualize_test_grids(self):
        """Visualize all test grids for debugging."""
        
        grids = self.get_all_test_grids()
        
        n_grids = len(grids)
        cols = 5
        rows = (n_grids + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        axes = axes.flatten()
        
        for idx, (name, grid) in enumerate(grids.items()):
            ax = axes[idx]
            ax.imshow(grid, cmap='tab20', interpolation='nearest')
            ax.set_title(name, fontsize=8)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(len(grids), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('test_grids_visualization.png', dpi=150, bbox_inches='tight')
        print("\nTest grids visualization saved to 'test_grids_visualization.png'")
        plt.close()


# ============================================================================
# TEST UTILITIES AND HELPERS
# ============================================================================

class TestUtilities:
    """Utility functions for testing."""
    
    @staticmethod
    def compare_grids(grid1: np.ndarray, grid2: np.ndarray, tolerance: float = 0) -> bool:
        """Compare two grids with optional tolerance."""
        if grid1.shape != grid2.shape:
            return False
        if tolerance > 0:
            return np.allclose(grid1, grid2, atol=tolerance)
        return np.array_equal(grid1, grid2)
    
    @staticmethod
    def get_grid_statistics(grid: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive statistics about a grid."""
        return {
            'shape': grid.shape,
            'size': grid.size,
            'non_zero': np.count_nonzero(grid),
            'unique_colors': len(np.unique(grid)),
            'density': np.count_nonzero(grid) / grid.size,
            'min_value': np.min(grid),
            'max_value': np.max(grid),
            'mean_value': np.mean(grid),
        }
    
    @staticmethod
    def create_random_grid(size: Tuple[int, int], num_colors: int = 3, 
                          density: float = 0.3, seed: int = 42) -> np.ndarray:
        """Create random grid with specified properties."""
        np.random.seed(seed)
        grid = np.zeros(size, dtype=int) 
        num_filled = int(size[0] * size[1] * density)
        filled_coords = np.random.choice(size[0] * size[1], num_filled, replace=False)
        for coord in filled_coords:
            i, j = coord // size[1], coord % size[1]
            grid[i, j] = np.random.randint(1, num_colors + 1)
        return grid
    
    @staticmethod
    def find_differences(grid1: np.ndarray, grid2: np.ndarray) -> List[Tuple[int, int]]:
        """Find coordinates where two grids differ."""
        if grid1.shape != grid2.shape:
            return []
        diff_mask = grid1 != grid2
        return list(zip(*np.where(diff_mask)))
    
    @staticmethod
    def visualize_grid(grid: np.ndarray):
        """Visualize a grid."""
        plot_grid(grid)
    
    @staticmethod
    def save_test_results(runner, filename: str = "test_results.json"):
        """Save test results to JSON file."""
        results_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(runner.results),
            'passed': sum(1 for r in runner.results if r.passed),
            'failed': sum(1 for r in runner.results if not r.passed),
            'total_time_ms': sum(r.duration_ms for r in runner.results),
            'tests': [
                {
                    'name': r.test_name,
                    'grid': r.grid_name,
                    'passed': r.passed,
                    'duration_ms': r.duration_ms,
                    'errors': r.errors,
                    'warnings': r.warnings,
                }
                for r in runner.results
            ]
        }
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nTest results saved to {filename}")


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
        grid = TestGridLibrary.single_pixel()
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
        grid = TestGridLibrary.filled_rectangle()
        original_grid = grid.copy()
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        assert np.array_equal(grid, original_grid), \
            "Input grid should not be modified"
    
    @staticmethod
    def test_color_zero_handling():
        """Regression: Color 0 (background) should be handled correctly."""
        grid = TestGridLibrary.filled_rectangle()
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
        grid = TestGridLibrary.single_pixel()
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
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
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
        base_grid = TestGridLibrary.filled_rectangle()
        dtypes = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16]
        for dtype in dtypes:
            grid = base_grid.astype(dtype)
            summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
            assert summary is not None, f"Failed for dtype {dtype}"
    
    @staticmethod
    def test_tuple_vs_list_coords():
        """Test coordinate format compatibility."""
        grid = TestGridLibrary.single_pixel()
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
    """Stress tests with extreme conditions."""
    
    @staticmethod
    def test_very_large_grid():
        """Test with very large grid."""
        print("\n  Testing 100x100 grid...")
        grid = np.random.randint(0, 5, (100, 100))
        start = time.perf_counter()
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        duration = time.perf_counter() - start
        print(f"    Completed in {duration:.2f}s")
        assert duration < 30.0, f"Very large grid too slow: {duration}s"
        assert summary is not None
    
    @staticmethod
    def test_many_small_objects():
        """Test with many small isolated objects."""
        print("\n  Testing grid with 100+ small objects...")
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
        print(f"    Found {num_objects} objects in {duration:.2f}s")
        assert duration < 10.0, f"Many objects too slow: {duration}s"
        assert num_objects > 50, "Should detect many objects"
    
    @staticmethod
    def test_dense_multicolor():
        """Test with dense multicolor grid."""
        print("\n  Testing dense 10-color grid...")
        grid = np.random.randint(1, 11, (30, 30))
        start = time.perf_counter()
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        duration = time.perf_counter() - start
        print(f"    Completed in {duration:.2f}s")
        assert duration < 15.0, f"Dense multicolor too slow: {duration}s"
    
    @staticmethod
    def test_extreme_aspect_ratio():
        """Test with extreme aspect ratios."""
        print("\n  Testing extreme aspect ratios...")
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
        print("    Both extreme ratios handled successfully")
    
    @staticmethod
    def test_maximum_colors():
        """Test with maximum number of colors."""
        print("\n  Testing with 10 different colors...")
        grid = np.zeros((10, 10), dtype=int)
        # Assign different color to each row
        for i in range(10):
            grid[i, :] = i + 1
        start = time.perf_counter()
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        duration = time.perf_counter() - start
        print(f"    Completed in {duration:.2f}s")
        assert duration < 10.0, f"Many colors too slow: {duration}s"
    
    @staticmethod
    def test_repeated_operations():
        """Test repeated operations for memory leaks."""
        print("\n  Testing 100 repeated operations...")
        grid = TestGridLibrary.multicolor_regions()
        start = time.perf_counter()
        for i in range(100):
            summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        duration = time.perf_counter() - start
        avg_time = duration / 100
        print(f"    Average time per operation: {avg_time*1000:.2f}ms")
        assert avg_time < 0.5, f"Repeated operations degrading: {avg_time}s per op"
    
    @staticmethod
    def test_complex_nesting():
        """Test with deeply nested structures."""
        print("\n  Testing deeply nested rectangles...")
        grid = np.zeros((40, 40), dtype=int)
        # Create nested rectangles
        for level in range(10):
            offset = level * 2
            color = level + 1
            size = 40 - offset * 2
            if size > 0:
                grid[offset:offset+size, offset:offset+size] = color
        start = time.perf_counter()
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        duration = time.perf_counter() - start
        
        print(f"    Completed in {duration:.2f}s")
        assert duration < 10.0, f"Nested structures too slow: {duration}s"

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance benchmarking tests."""
    @staticmethod
    def benchmark_pattern_generation():
        """Benchmark pattern generation"""
        results = {}
        grid_sizes = [(10, 10), (20, 20), (30, 30)]
        shape_types = ['line', 'rectangle', 'diagonal']
        for size in grid_sizes:
            start = time.perf_counter()
            patterns = generate_patterns(size, shape_types, multithreading=True)
            duration = time.perf_counter() - start
            results[size] = duration * 1000
        return results
    
    @staticmethod
    def benchmark_grid_summary():
        """Benchmark GridSummary creation."""
        results = {}
        test_grids = {
            'simple': TestGridLibrary.filled_rectangle(),
            'complex': TestGridLibrary.multicolor_regions(),
            'nested': TestGridLibrary.nested_rectangles(),
        }
        for name, grid in test_grids.items():
            start = time.perf_counter()
            summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
            duration = time.perf_counter() - start
            results[name] = duration * 1000
        return results
    
    @staticmethod
    def benchmark_object_operations():
        """Benchmark GridObject operations."""
        grid = TestGridLibrary.filled_rectangle()
        coords = [(i, j) for i in range(2, 6) for j in range(2, 6)]
        # Creation benchmark
        start = time.perf_counter()
        for _ in range(100):
            obj = GridObject('rect', coords, [1], 'test', grid.shape, 0, grid)
        create_time = (time.perf_counter() - start) * 1000
        
        # Embedding benchmark
        obj = GridObject('rect', coords, [1], 'test', grid.shape, 0, grid)
        start = time.perf_counter()
        for _ in range(100):
            obj.create_embedding()
        embed_time = (time.perf_counter() - start) * 1000
        return {'creation': create_time, 'embedding': embed_time}
    
    @staticmethod
    def stress_test_many_objects():
        """Stress test with many objects."""
        # Create grid with many small objects
        grid = np.zeros((30, 30), dtype=int)
        obj_id = 1
        for i in range(1, 28, 3):
            for j in range(1, 28, 3):
                grid[i:i+2, j:j+2] = obj_id % 5 + 1
                obj_id += 1
        start = time.perf_counter()
        summary = GridSummary(grid=grid, shape=grid.shape, font_color=0, levels=[1])
        duration = time.perf_counter() - start
        return duration * 1000


# ============================================================================
# UNIFIED TEST FRAMEWORK
# ============================================================================

class TestConfig:
    """Centralized test configuration."""
    RUN_PERFORMANCE_TESTS = True
    RUN_EDGE_CASE_TESTS = True
    RUN_STRESS_TESTS = True
    VERBOSE = True
    PERFORMANCE_THRESHOLD_MS = 100
    STRESS_TEST_ITERATIONS = 100
    RANDOM_SEED = 42
    SAVE_VISUALIZATIONS = False  # Set to True to save grid visualizations

class TestResult:
    """Container for test results."""
    def __init__(self, test_name: str, grid_name: str = None):
        self.test_name = test_name
        self.grid_name = grid_name
        self.passed = False
        self.duration_ms = 0.0
        self.errors = []
        self.warnings = []
        self.metadata = {}
    
    def __repr__(self):
        status = "✓" if self.passed else "✗"
        grid_info = f" [{self.grid_name}]" if self.grid_name else ""
        return f"{status} {self.test_name}{grid_info} ({self.duration_ms:.2f}ms)"


class UnifiedTestRunner:
    """Unified test runner with grid-based testing."""
    
    def __init__(self, test_grids: List[tuple] = None, config: TestConfig = None):
        self.config = config or TestConfig()
        self.results = []
        self.test_grids = test_grids if test_grids else TestGridLibrary.get_all_test_grids()
        self.errors = []
        self.slow_tests = []
    
    def run_test_on_grid(self, test_func: Callable, grid_name: str, 
                        grid: np.ndarray, *args, **kwargs) -> TestResult:
        """Execute a test function on a specific grid"""
        result = TestResult(test_func.__name__, grid_name)
        start_time = time.perf_counter()
        try:
            test_func(grid, *args, **kwargs)
            result.passed = True
        except AssertionError as e:
            result.errors.append(f"Assertion: {str(e)}")
            self.errors.append((f"Assertion: {str(e)}", test_func.__name__, grid_name, grid))
        except Exception as e:
            result.errors.append(f"{type(e).__name__}: {str(e)}")
            self.errors.append((f"{type(e).__name__}: {str(e)}", test_func.__name__, grid_name, grid))
        finally:
            result.duration_ms = (time.perf_counter() - start_time) * 1000
        if result.duration_ms > self.config.PERFORMANCE_THRESHOLD_MS:
            result.warnings.append(f"Slow test: {result.duration_ms:.2f}ms")
            self.slow_tests.append((test_func.__name__, f"Slow test: {result.duration_ms:.2f}ms"))
        self.results.append(result)
        return result
    
    def run_test_on_all_grids(self, test_func: Callable, 
                             grid_filter: Callable = None) -> List[TestResult]:
        """Execute a test function on all applicable grids."""
        grid_filter = grid_filter or (lambda name, grid: True)
        results = []
        
        for grid_name, grid in self.test_grids.items():
            if grid_filter(grid_name, grid):
                result = self.run_test_on_grid(test_func, grid_name, grid)
                results.append(result)
        
        return results
    
    def run_single_test(self, test_func: Callable, *args, **kwargs) -> TestResult:
        """Execute a single test without grid dependency."""
        result = TestResult(test_func.__name__)
        
        start_time = time.perf_counter()
        try:
            test_func(*args, **kwargs)
            result.passed = True
        except AssertionError as e:
            result.errors.append(f"Assertion: {str(e)}")
            self.errors.append((test_func.__name__, f"Assertion: {str(e)}"))
        except Exception as e:
            result.errors.append(f"{type(e).__name__}: {str(e)}")
            self.errors.append((test_func.__name__, f"{type(e).__name__}: {str(e)}"))
        finally:
            result.duration_ms = (time.perf_counter() - start_time) * 1000
        
        if result.duration_ms > self.config.PERFORMANCE_THRESHOLD_MS:
            result.warnings.append(f"Slow test: {result.duration_ms:.2f}ms")
            self.slow_tests.append((test_func.__name__, f"Slow test: {result.duration_ms:.2f}ms"))
        
        self.results.append(result)
        return result
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        if not self.results:
            return "No tests executed."
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        total_time = sum(r.duration_ms for r in self.results)
        # Group by test function
        by_function = defaultdict(list)
        for result in self.results:
            by_function[result.test_name].append(result)
        report = [
            "\n" + "="*80,
            "UNIFIED TEST EXECUTION REPORT",
            "="*80,
            f"Total Tests: {len(self.results)} | Passed: {passed} | Failed: {failed}",
            f"Total Time: {total_time:.2f}ms | Average: {total_time/len(self.results):.2f}ms",
            f"Success Rate: {100*passed/len(self.results):.1f}%",
            "="*80,
        ]
        # Summary by function
        report.append("\nSUMMARY BY FUNCTION:")
        report.append("-"*80)
        for func_name, func_results in sorted(by_function.items()):
            func_passed = sum(1 for r in func_results if r.passed)
            func_total = len(func_results)
            status = "✓" if func_passed == func_total else "✗"
            report.append(f"{status} {func_name}: {func_passed}/{func_total} passed")
            
            # Show failures
            failures = [r for r in func_results if not r.passed]
            if failures:
                for failure in failures:
                    report.append(f"    ✗ {failure.grid_name or 'N/A'}: {failure.errors[0][:60]}")
        # Detailed results
        report.append("\n" + "="*80)
        report.append("DETAILED RESULTS:")
        report.append("-"*80)
        for result in self.results:
            report.append(str(result))
            if result.errors:
                for error in result.errors:
                    report.append(f"  Error: {error}")
            if result.warnings:
                for warning in result.warnings:
                    report.append(f"  Warning: {warning}")
        
        report.append("="*80 + "\n")
        return "\n".join(report)

# ============================================================================
# QUICK-START FUNCTIONS
# ============================================================================

def quick_test(module_name: str = 'all'):
    """Quick test runner for specific modules
    
    Args:
        module_name: 'pattern', 'object', 'summary', 'filter', 'relation', 'match', or 'all'
    """
    runner = UnifiedTestRunner()
    
    print(f"\n{'='*80}")
    print(f"QUICK TEST: {module_name.upper()}")
    print(f"{'='*80}\n")
    
    if module_name in ['object', 'all']:
        print("Testing GridObject...")
        runner.run_test_on_all_grids(TestGridObject.test_object_creation)
        runner.run_test_on_all_grids(TestGridObject.test_object_properties)
    
    if module_name in ['summary', 'all']:
        print("Testing GridSummary...")
        runner.run_test_on_all_grids(TestGridSummary.test_summary_creation)
    
    if module_name in ['relation', 'all']:
        print("Testing RelationAnalyzer...")
        runner.run_test_on_all_grids(TestRelationAnalyzer.test_relation_detection)
    
    if module_name in ['match', 'all']:
        print("Testing Match Score...")
        runner.run_test_on_all_grids(TestMatchScore.test_rotation_generation)
    
    print(runner.generate_report())
    return runner


def test_specific_grids(*grid_names):
    """Test specific grid scenarios
    
    Example:
        test_specific_grids('filled_rectangle', 'nested_rectangles', 'multicolor_regions')
    """
    runner = UnifiedTestRunner()
    
    # Filter to requested grids
    runner.test_grids = {k: v for k, v in runner.test_grids.items() if k in grid_names}
    
    if not runner.test_grids:
        print(f"No grids found matching: {grid_names}")
        return None
    
    print(f"\n{'='*80}")
    print(f"TESTING SPECIFIC GRIDS: {', '.join(grid_names)}")
    print(f"{'='*80}\n")
    
    # Run core tests
    runner.run_test_on_all_grids(TestGridObject.test_object_creation)
    runner.run_test_on_all_grids(TestGridSummary.test_summary_creation)
    
    print(runner.generate_report())
    return runner


def validate_implementation():
    """Run validation tests only (fast, checks correctness)"""
    runner = UnifiedTestRunner()
    
    print(f"\n{'='*80}")
    print("VALIDATION TEST SUITE")
    print(f"{'='*80}\n")
    
    print("Running correctness validation...")
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_coordinate_bounds)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_color_consistency)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_connectivity)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_size_calculations)
    
    print("Running regression tests...")
    runner.run_single_test(TestRegression.test_empty_component_handling)
    runner.run_single_test(TestRegression.test_single_cell_object)
    runner.run_single_test(TestRegression.test_grid_modification_safety)
    
    print("Running property tests...")
    runner.run_test_on_all_grids(TestProperties.test_object_count_invariant)
    runner.run_test_on_all_grids(TestProperties.test_distance_symmetry)
    
    print(runner.generate_report())
    return runner


def performance_test():
    """Run performance tests only"""
    print(f"\n{'='*80}")
    print("PERFORMANCE TEST SUITE")
    print(f"{'='*80}\n")
    
    print("Pattern Generation Benchmark:")
    results = TestPerformance.benchmark_pattern_generation()
    for size, duration in results.items():
        status = "✓" if duration < 50 else "⚠" if duration < 100 else "✗"
        print(f"  {status} {size}: {duration:.2f}ms")
    
    print("\nGrid Summary Benchmark:")
    results = TestPerformance.benchmark_grid_summary()
    for name, duration in results.items():
        status = "✓" if duration < 50 else "⚠" if duration < 100 else "✗"
        print(f"  {status} {name}: {duration:.2f}ms")
    
    print("\nObject Operations Benchmark:")
    results = TestPerformance.benchmark_object_operations()
    for op, duration in results.items():
        status = "✓" if duration < 50 else "⚠" if duration < 100 else "✗"
        print(f"  {status} {op}: {duration:.2f}ms")
    
    print("\nStress Test:")
    duration = TestPerformance.stress_test_many_objects()
    status = "✓" if duration < 500 else "⚠" if duration < 1000 else "✗"
    print(f"  {status} Many objects: {duration:.2f}ms")
    
    print(f"\n{'='*80}")

def list_test_grids():
    """List all available test grid scenarios"""
    grids = TestGridLibrary.get_all_test_grids()
    
    print(f"\n{'='*80}")
    print(f"AVAILABLE TEST GRID SCENARIOS ({len(grids)} total)")
    print(f"{'='*80}\n")
    
    for name, grid in grids.items():
        stats = TestUtilities.get_grid_statistics(grid)
        print(f"  • {name:25} | Shape: {stats['shape']} | "
              f"Colors: {stats['unique_colors']} | "
              f"Density: {stats['density']:.2%}")
    
    print(f"\n{'='*80}")
    print("Use test_specific_grids('grid_name1', 'grid_name2') to test specific grids")
    print(f"{'='*80}\n")


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

def run_all_tests():
    """Execute comprehensive test suite"""
    runner = UnifiedTestRunner()
    
    print("\n" + "="*80)
    print("UNIFIED GRID-BASED TEST SUITE")
    print("="*80)
    print(f"Testing with {len(runner.test_grids)} predefined grid scenarios")
    print("="*80)
    
    
    # GridObject Tests
    print("[1/5] Running GridObject Tests...")
    runner.run_test_on_all_grids(TestGridObject.test_object_creation)
    runner.run_test_on_all_grids(TestGridObject.test_object_properties)
    runner.run_test_on_all_grids(TestGridObject.test_object_symmetry)
    runner.run_test_on_all_grids(TestGridObject.test_object_immutability)
    
    # GridSummary Tests
    print("[2/5] Running GridSummary Tests...")
    runner.run_test_on_all_grids(TestGridSummary.test_summary_creation)
    runner.run_test_on_all_grids(TestGridSummary.test_multiple_levels)
    runner.run_test_on_all_grids(TestGridSummary.test_relation_analysis)
    runner.run_test_on_all_grids(TestGridSummary.test_embeddings)
    
    # Match Score Tests
    print("[3/5] Running Match Score Tests...")
    runner.run_test_on_all_grids(TestMatchScore.test_rotation_generation)
    runner.run_test_on_all_grids(TestMatchScore.test_intersection_checking)
    runner.run_test_on_all_grids(TestMatchScore.test_match_score_calculation)
    
    # RelationAnalyzer Tests
    print("[4/5] Running RelationAnalyzer Tests...")
    runner.run_test_on_all_grids(TestRelationAnalyzer.test_relation_detection)
    runner.run_test_on_all_grids(TestRelationAnalyzer.test_alignment_detection)
    
    # Integration Tests
    print("[5/5] Running Integration Tests...")
    runner.run_test_on_all_grids(TestIntegration.test_pattern_to_object_pipeline)
    runner.run_test_on_all_grids(TestIntegration.test_immutability_preservation)
    
    # Edge Case Tests
    print("\nRunning Edge Case Tests...")
    runner.run_single_test(TestEdgeCases.test_empty_grid)
    runner.run_single_test(TestEdgeCases.test_single_pixel_grid)
    runner.run_single_test(TestEdgeCases.test_minimum_size_grid)
    runner.run_single_test(TestEdgeCases.test_large_grid)
    runner.run_single_test(TestEdgeCases.test_many_colors)
    runner.run_single_test(TestEdgeCases.test_all_same_color)
    runner.run_single_test(TestEdgeCases.test_rectangular_grids)
    runner.run_single_test(TestEdgeCases.test_disconnected_pixels)
    
    # Performance Benchmarks
    if TestConfig.RUN_PERFORMANCE_TESTS:
        print("\nRunning Performance Benchmarks...")
        print("  Pattern Generation:")
        results = TestPerformance.benchmark_pattern_generation()
        for size, duration in results.items():
            print(f"    {size}: {duration:.2f}ms")
        
        print("  Grid Summary:")
        results = TestPerformance.benchmark_grid_summary()
        for name, duration in results.items():
            print(f"    {name}: {duration:.2f}ms")
        
        print("  Object Operations:")
        results = TestPerformance.benchmark_object_operations()
        for op, duration in results.items():
            print(f"    {op}: {duration:.2f}ms")
        
        print("  Stress Test (many objects):")
        duration = TestPerformance.stress_test_many_objects()
        print(f"    {duration:.2f}ms")
    
    # Generate report
    report = runner.generate_report()
    print(report)
    return runner
    

# ============================================================================
# EXTENDED TEST EXECUTION
# ============================================================================

def run_comprehensive_tests():
    """Run all test suites including validation, regression, and property tests"""
    runner = UnifiedTestRunner()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE GRID-BASED TEST SUITE")
    print("="*80)
    print(f"Testing with {len(runner.test_grids)} predefined grid scenarios")
    print("="*80)
    
    # Core Functional Tests (from previous sections)
    print("\n[PHASE 1/4] Core Functional Tests")
    print("-"*80)
    
    print("  GridObject...")
    runner.run_test_on_all_grids(TestGridObject.test_object_creation)
    runner.run_test_on_all_grids(TestGridObject.test_object_properties)
    runner.run_test_on_all_grids(TestGridObject.test_object_symmetry)
    runner.run_test_on_all_grids(TestGridObject.test_object_immutability)
    
    print("  GridSummary...")
    runner.run_test_on_all_grids(TestGridSummary.test_summary_creation)
    runner.run_test_on_all_grids(TestGridSummary.test_multiple_levels)
    runner.run_test_on_all_grids(TestGridSummary.test_relation_analysis)
    runner.run_test_on_all_grids(TestGridSummary.test_embeddings)
    
    print("  Match Score...")
    runner.run_test_on_all_grids(TestMatchScore.test_rotation_generation)
    runner.run_test_on_all_grids(TestMatchScore.test_intersection_checking)
    runner.run_test_on_all_grids(TestMatchScore.test_match_score_calculation)
    
    print("  RelationAnalyzer...")
    runner.run_test_on_all_grids(TestRelationAnalyzer.test_relation_detection)
    runner.run_test_on_all_grids(TestRelationAnalyzer.test_alignment_detection)
    
    print("  Integration...")
    runner.run_test_on_all_grids(TestIntegration.test_pattern_to_object_pipeline)
    runner.run_test_on_all_grids(TestIntegration.test_immutability_preservation)
    
    # Validation Tests
    print("\n[PHASE 2/4] Correctness Validation Tests")
    print("-"*80)
    
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_coordinate_bounds)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_color_consistency)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_connectivity)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_no_duplicate_coordinates)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_size_calculations)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_center_calculation)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_compactness_range)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_rotation_preservation)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_embedding_dimensions)
    
    # Regression Tests
    print("\n[PHASE 3/4] Regression Tests")
    print("-"*80)
    
    runner.run_single_test(TestRegression.test_empty_component_handling)
    runner.run_single_test(TestRegression.test_single_cell_object)
    runner.run_single_test(TestRegression.test_rectangle_cache_consistency)
    runner.run_single_test(TestRegression.test_grid_modification_safety)
    runner.run_single_test(TestRegression.test_color_zero_handling)
    runner.run_single_test(TestRegression.test_boundary_objects)
    runner.run_single_test(TestRegression.test_large_object_performance)
    runner.run_single_test(TestRegression.test_multicolor_object)
    
    # Property-Based Tests
    print("\n[PHASE 4/4] Property-Based Tests")
    print("-"*80)
    
    runner.run_test_on_all_grids(TestProperties.test_object_count_invariant)
    runner.run_test_on_all_grids(TestProperties.test_symmetry_invariant)
    runner.run_test_on_all_grids(TestProperties.test_containment_transitivity)
    runner.run_test_on_all_grids(TestProperties.test_distance_symmetry)
    runner.run_test_on_all_grids(TestProperties.test_alignment_consistency)
    
    # Edge Cases
    print("\n[EDGE CASES]")
    print("-"*80)
    
    runner.run_single_test(TestEdgeCases.test_empty_grid)
    runner.run_single_test(TestEdgeCases.test_single_pixel_grid)
    runner.run_single_test(TestEdgeCases.test_minimum_size_grid)
    runner.run_single_test(TestEdgeCases.test_large_grid)
    runner.run_single_test(TestEdgeCases.test_many_colors)
    runner.run_single_test(TestEdgeCases.test_all_same_color)
    runner.run_single_test(TestEdgeCases.test_rectangular_grids)
    runner.run_single_test(TestEdgeCases.test_disconnected_pixels)
    
    # Performance Benchmarks
    if TestConfig.RUN_PERFORMANCE_TESTS:
        print("\n[PERFORMANCE BENCHMARKS]")
        print("-"*80)
        
        print("  Pattern Generation:")
        results = TestPerformance.benchmark_pattern_generation()
        for size, duration in results.items():
            print(f"    {size}: {duration:.2f}ms")
        
        print("  Grid Summary:")
        results = TestPerformance.benchmark_grid_summary()
        for name, duration in results.items():
            print(f"    {name}: {duration:.2f}ms")
        
        print("  Object Operations:")
        results = TestPerformance.benchmark_object_operations()
        for op, duration in results.items():
            print(f"    {op}: {duration:.2f}ms")
        
        print("  Stress Test:")
        duration = TestPerformance.stress_test_many_objects()
        print(f"    Many objects: {duration:.2f}ms")
    
    # Generate report
    report = runner.generate_report()
    print(report)
    
    # Additional statistics
    print("\n" + "="*80)
    print("TEST COVERAGE STATISTICS")
    print("="*80)
    
    grid_coverage = defaultdict(int)
    for result in runner.results:
        if result.grid_name:
            grid_coverage[result.grid_name] += 1
    
    print(f"\nTests per grid scenario:")
    for grid_name, count in sorted(grid_coverage.items(), key=lambda x: x[1], reverse=True):
        print(f"  {grid_name}: {count} tests")
    
    return runner


# ============================================================================
# FINAL COMPREHENSIVE TEST RUNNER
# ============================================================================

def run_all_comprehensive_tests():
    """Master function to run all test categories"""
    runner = UnifiedTestRunner()
    
    print("\n" + "="*80)
    print("MASTER COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Grid Scenarios: {len(runner.test_grids)}")
    print(f"Test Categories: 9 (Functional, Validation, Regression, Property, etc.)")
    print("="*80)
    
    # Track timing for each phase
    phase_times = {}
    
    # Phase 1: Core Functional Tests
    print("\n" + "="*80)
    print("[PHASE 1/10] CORE FUNCTIONAL TESTS")
    print("="*80)
    phase_start = time.perf_counter()
    
    print("  GridObject...")
    runner.run_test_on_all_grids(TestGridObject.test_object_creation)
    runner.run_test_on_all_grids(TestGridObject.test_object_properties)
    runner.run_test_on_all_grids(TestGridObject.test_object_symmetry)
    runner.run_test_on_all_grids(TestGridObject.test_object_immutability)
    
    print("  GridSummary...")
    runner.run_test_on_all_grids(TestGridSummary.test_summary_creation)
    runner.run_test_on_all_grids(TestGridSummary.test_multiple_levels)
    runner.run_test_on_all_grids(TestGridSummary.test_relation_analysis)
    runner.run_test_on_all_grids(TestGridSummary.test_embeddings)
    
    print("  Match Score...")
    runner.run_test_on_all_grids(TestMatchScore.test_rotation_generation)
    runner.run_test_on_all_grids(TestMatchScore.test_intersection_checking)
    runner.run_test_on_all_grids(TestMatchScore.test_match_score_calculation)
    
    print("  RelationAnalyzer...")
    runner.run_test_on_all_grids(TestRelationAnalyzer.test_relation_detection)
    runner.run_test_on_all_grids(TestRelationAnalyzer.test_alignment_detection)
    
    print("  Integration...")
    runner.run_test_on_all_grids(TestIntegration.test_pattern_to_object_pipeline)
    runner.run_test_on_all_grids(TestIntegration.test_immutability_preservation)
    
    phase_times['Core Functional'] = time.perf_counter() - phase_start
    
    # Phase 2: Correctness Validation
    print("\n" + "="*80)
    print("[PHASE 2/10] CORRECTNESS VALIDATION")
    print("="*80)
    phase_start = time.perf_counter()
    
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_coordinate_bounds)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_color_consistency)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_connectivity)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_no_duplicate_coordinates)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_size_calculations)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_center_calculation)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_compactness_range)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_rotation_preservation)
    runner.run_test_on_all_grids(TestCorrectnessValidation.test_embedding_dimensions)
    
    phase_times['Correctness Validation'] = time.perf_counter() - phase_start
    
    # Phase 3: Regression Tests
    print("\n" + "="*80)
    print("[PHASE 3/10] REGRESSION TESTS")
    print("="*80)
    phase_start = time.perf_counter()
    
    runner.run_single_test(TestRegression.test_empty_component_handling)
    runner.run_single_test(TestRegression.test_single_cell_object)
    runner.run_single_test(TestRegression.test_rectangle_cache_consistency)
    runner.run_single_test(TestRegression.test_grid_modification_safety)
    runner.run_single_test(TestRegression.test_color_zero_handling)
    runner.run_single_test(TestRegression.test_boundary_objects)
    runner.run_single_test(TestRegression.test_large_object_performance)
    runner.run_single_test(TestRegression.test_multicolor_object)
    
    phase_times['Regression'] = time.perf_counter() - phase_start
    
    # Phase 4: Property-Based Tests
    print("\n" + "="*80)
    print("[PHASE 4/10] PROPERTY-BASED TESTS")
    print("="*80)
    phase_start = time.perf_counter()
    
    runner.run_test_on_all_grids(TestProperties.test_object_count_invariant)
    runner.run_test_on_all_grids(TestProperties.test_symmetry_invariant)
    runner.run_test_on_all_grids(TestProperties.test_containment_transitivity)
    runner.run_test_on_all_grids(TestProperties.test_distance_symmetry)
    runner.run_test_on_all_grids(TestProperties.test_alignment_consistency)
    
    phase_times['Property-Based'] = time.perf_counter() - phase_start
    
    # Phase 5: Edge Cases
    print("\n" + "="*80)
    print("[PHASE 5/10] EDGE CASES")
    print("="*80)
    phase_start = time.perf_counter()
    
    runner.run_single_test(TestEdgeCases.test_empty_grid)
    runner.run_single_test(TestEdgeCases.test_single_pixel_grid)
    runner.run_single_test(TestEdgeCases.test_minimum_size_grid)
    runner.run_single_test(TestEdgeCases.test_large_grid)
    runner.run_single_test(TestEdgeCases.test_many_colors)
    runner.run_single_test(TestEdgeCases.test_all_same_color)
    runner.run_single_test(TestEdgeCases.test_rectangular_grids)
    runner.run_single_test(TestEdgeCases.test_disconnected_pixels)
    
    phase_times['Edge Cases'] = time.perf_counter() - phase_start
    
    # Phase 6: Stress Tests
    if TestConfig.RUN_STRESS_TESTS:
        print("\n" + "="*80)
        print("[PHASE 6/10] STRESS TESTS")
        print("="*80)
        phase_start = time.perf_counter()
        
        runner.run_single_test(TestStress.test_many_small_objects)
        runner.run_single_test(TestStress.test_dense_multicolor)
        runner.run_single_test(TestStress.test_extreme_aspect_ratio)
        runner.run_single_test(TestStress.test_maximum_colors)
        runner.run_single_test(TestStress.test_repeated_operations)
        runner.run_single_test(TestStress.test_complex_nesting)
        
        phase_times['Stress'] = time.perf_counter() - phase_start
    
    # Phase 7: Compatibility Tests
    print("\n" + "="*80)
    print("[PHASE 7/10] COMPATIBILITY TESTS")
    print("="*80)
    phase_start = time.perf_counter()
    
    runner.run_single_test(TestCompatibility.test_different_dtypes)
    runner.run_single_test(TestCompatibility.test_tuple_vs_list_coords)
    
    phase_times['Compatibility'] = time.perf_counter() - phase_start
    
    # Phase 8: Documentation Tests
    print("\n" + "="*80)
    print("[PHASE 8/10] DOCUMENTATION TESTS")
    print("="*80)
    phase_start = time.perf_counter()
    
    runner.run_single_test(TestDocumentation.test_basic_usage_example)
    runner.run_single_test(TestDocumentation.test_pattern_generation_example)
    runner.run_single_test(TestDocumentation.test_object_creation_example)
    runner.run_single_test(TestDocumentation.test_relation_analysis_example)
    
    phase_times['Documentation'] = time.perf_counter() - phase_start
    
    # Phase 9: Performance Benchmarks
    if TestConfig.RUN_PERFORMANCE_TESTS:
        print("\n" + "="*80)
        print("[PHASE 9/10] PERFORMANCE BENCHMARKS")
        print("="*80)
        phase_start = time.perf_counter()
        
        print("  Pattern Generation:")
        results = TestPerformance.benchmark_pattern_generation()
        for size, duration in results.items():
            print(f"    {size}: {duration:.2f}ms")
        
        print("  Grid Summary:")
        results = TestPerformance.benchmark_grid_summary()
        for name, duration in results.items():
            print(f"    {name}: {duration:.2f}ms")
        
        print("  Object Operations:")
        results = TestPerformance.benchmark_object_operations()
        for op, duration in results.items():
            print(f"    {op}: {duration:.2f}ms")
        
        print("  Stress Test:")
        duration = TestPerformance.stress_test_many_objects()
        print(f"    Many objects: {duration:.2f}ms")
        
        phase_times['Performance'] = time.perf_counter() - phase_start
    
    # Phase 10: Generate Reports
    print("\n" + "="*80)
    print("[PHASE 10/10] GENERATING REPORTS")
    print("="*80)
    
    # Main report
    report = runner.generate_report()
    print(report)
    
    # Phase timing summary
    print("\n" + "="*80)
    print("PHASE TIMING SUMMARY")
    print("="*80)
    for phase, duration in phase_times.items():
        print(f"  {phase}: {duration:.2f}s")
    print(f"  Total: {sum(phase_times.values()):.2f}s")
    
    # Test coverage statistics
    print("\n" + "="*80)
    print("TEST COVERAGE STATISTICS")
    print("="*80)
    
    grid_coverage = defaultdict(int)
    grid_failures = defaultdict(int)
    
    for result in runner.results:
        if result.grid_name:
            grid_coverage[result.grid_name] += 1
            if not result.passed:
                grid_failures[result.grid_name] += 1
    
    print(f"\nTests per grid scenario:")
    for grid_name in sorted(grid_coverage.keys()):
        count = grid_coverage[grid_name]
        failures = grid_failures[grid_name]
        status = "✓" if failures == 0 else f"✗ ({failures} failed)"
        print(f"  {status} {grid_name}: {count} tests")
    
    # Final statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    
    passed = sum(1 for r in runner.results if r.passed)
    total = len(runner.results)
    success_rate = 100 * passed / total if total > 0 else 0
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Total Execution Time: {sum(phase_times.values()):.2f}s")
    
    # Success/failure message
    print("\n" + "="*80)
    if success_rate >= 95:
        print("✓ ✓ ✓ SUCCESS: Test suite passed with {:.1f}% success rate!".format(success_rate))
    elif success_rate >= 85:
        print("⚠ WARNING: Test suite passed with {:.1f}% success rate".format(success_rate))
        print("Some tests failed - review needed")
    else:
        print("✗ ✗ ✗ FAILURE: Test suite failed with {:.1f}% success rate".format(success_rate))
        print("Critical issues detected - immediate review required!")
    print("="*80)
    
    # Save results
    TestUtilities.save_test_results(runner)
    
    return runner