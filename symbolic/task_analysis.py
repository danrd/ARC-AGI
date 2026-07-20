from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict, Counter
import numpy as np
from symbolic.summaries import GridSummary, calculate_shape_similarity

colors_mapping = {
        0: 'black', 1: 'blue', 2: 'red', 3: 'green', 4: 'yellow',
        5: 'gray', 6: 'magenta', 7: 'orange', 8: 'sky', 9: 'brown'
        }

@dataclass(frozen=True)
class CellChange:
    """Represents a single cell modification between input and output."""
    coord: Tuple[int, int]
    old_color: int
    new_color: int
    
@dataclass(frozen=True)
class GridDiff:
    """Captures differences between input and output grids."""
    same_shape: bool
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    changed_cells: Tuple[CellChange, ...] = field(default_factory=tuple)
    added_cells: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    removed_cells: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    num_changes: int = 0
    change_ratio: float = 0.0  # Proportion of cells changed
    
    @property
    def has_size_change(self) -> bool:
        return not self.same_shape

@dataclass(frozen=True)
class ObjectChange:
    """Tracks changes to a specific object between input and output."""
    object_label: str
    change_type: str  # 'modified', 'added', 'deleted', 'unchanged'
    input_obj: Optional[Any] = None  # GridObject or None
    output_obj: Optional[Any] = None  # GridObject or None
    property_changes: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    similarity_score: float = 0.0
    
@dataclass(frozen=True)
class TransformationPattern:
    """Describes a consistent pattern observed in the transformation."""
    pattern_type: str
    description: str
    confidence: float  # 0.0 to 1.0
    supporting_evidence: Tuple[str, ...] = field(default_factory=tuple)
    parameters: Dict[str, Any] = field(default_factory=dict)


class SubtaskAnalysis:
    """Analyzes a single input-output example pair."""
    
    def __init__(self, subtask, font_color: int = 0, levels: List[int] = [2]):
        """
        Initialize analysis for a single ARCSubtask.
        
        Args:
            subtask: ARCSubtask instance with train_inp and train_out
            font_color: Background color to ignore (default 0)
            levels: Representation levels to analyze
        """
        
        self.subtask = subtask
        self.example_id = subtask.label
        self.input_grid = subtask.train_inp
        self.output_grid = subtask.train_out
        self.font_color = font_color
        self.levels = levels
        self.primary_level = 2
        
        # Create grid summaries
        self.input_summary = GridSummary(
            self.input_grid, self.input_grid.shape, font_color, levels
        )
        self.output_summary = GridSummary(
            self.output_grid, self.output_grid.shape, font_color, levels
        )
        
        # Analyze differences
        self.grid_diff = self._compute_grid_diff()
        self.object_changes = {}  # level -> List[ObjectChange]
        self.transformation_patterns = []
        
        # Perform analysis
        self._analyze_transformations()
    
    def _compute_grid_diff(self) -> GridDiff:
        """Compute cell-level differences between input and output."""
        same_shape = self.input_grid.shape == self.output_grid.shape
        
        if same_shape:
            # Find changed cells
            changes = []
            for i in range(self.input_grid.shape[0]):
                for j in range(self.input_grid.shape[1]):
                    old_color = self.input_grid[i, j]
                    new_color = self.output_grid[i, j]
                    if old_color != new_color:
                        changes.append(CellChange((i, j), old_color, new_color))
            
            total_cells = self.input_grid.shape[0] * self.input_grid.shape[1]
            change_ratio = len(changes) / total_cells if total_cells > 0 else 0.0
            
            return GridDiff(
                same_shape=True,
                input_shape=self.input_grid.shape,
                output_shape=self.output_grid.shape,
                changed_cells=tuple(changes),
                num_changes=len(changes),
                change_ratio=change_ratio
            )
        else:
            # Different sizes - identify added/removed regions
            # Try to find input grid within output or vice versa
            smaller = self.input_grid if np.prod(self.input_grid.shape) < np.prod(self.output_grid.shape) else self.output_grid
            larger = self.output_grid if np.prod(self.input_grid.shape) < np.prod(self.output_grid.shape) else self.input_grid
            
            return GridDiff(
                same_shape=False,
                input_shape=self.input_grid.shape,
                output_shape=self.output_grid.shape,
                num_changes=abs(np.prod(self.input_grid.shape) - np.prod(self.output_grid.shape))
            )

    def _analyze_transformations(self):
        """Analyze transformations at each representation level."""
        for level in self.levels:
            self.object_changes[level] = self._analyze_objects_at_level(level)
            patterns = self._detect_patterns_at_level(level)
            self.transformation_patterns.extend(patterns)
    
    def _analyze_objects_at_level(self, level: int) -> List[ObjectChange]:
        """Compare objects between input and output at a specific level."""
        input_objs = self.input_summary.repr_levels[level].objects
        output_objs = self.output_summary.repr_levels[level].objects
        
        changes = []
        matched_input = {}
        matched_output = {}
        
        # Match objects based on similarity
        for in_obj in input_objs:
            best_match = None
            best_score = 0.0
            
            for out_obj in output_objs:
                if out_obj.label in matched_output.keys():
                    continue
                score = self._calculate_object_similarity(in_obj, out_obj)
                if score > best_score and score > 0.5:  # Threshold for matching
                    best_score = score
                    best_match = out_obj
            
            if best_match:
                matched_input[in_obj.label] = in_obj
                matched_output[best_match.label] = best_match
                
                # Check for property changes
                prop_changes = self._find_property_changes(in_obj, best_match)
                change_type = 'modified' if prop_changes else 'unchanged'
                
                changes.append(ObjectChange(
                    object_label=in_obj.label,
                    change_type=change_type,
                    input_obj=in_obj,
                    output_obj=best_match,
                    property_changes=prop_changes,
                    similarity_score=best_score
                ))
            else:
                # Object deleted
                changes.append(ObjectChange(
                    object_label=in_obj.label,
                    change_type='deleted',
                    input_obj=in_obj,
                    output_obj=None,
                    similarity_score=0.0
                ))
        
        # Find added objects
        for out_obj in output_objs:
            if out_obj.label not in matched_output.keys():
                changes.append(ObjectChange(
                    object_label=out_obj.label,
                    change_type='added',
                    input_obj=None,
                    output_obj=out_obj,
                    similarity_score=0.0
                ))
        
        return changes

    def _calculate_object_similarity(self, obj1, obj2) -> float:
        """Calculate similarity score between two objects."""
        score = 0.0
        weights_sum = 0.0

        # Exact shape and color match 
        if hasattr(obj1, 'color_structure') and hasattr(obj2, 'color_structure'):
            if (obj1.color_structure.shape == obj2.color_structure.shape and 
                np.array_equal(obj1.color_structure, obj2.color_structure)):
                    return 1
        
        # Exact shape match 
        if hasattr(obj1, 'obj_mask') and hasattr(obj2, 'obj_mask'):
            if (obj1.obj_mask.shape == obj2.obj_mask.shape and 
                np.array_equal(obj1.obj_mask, obj2.obj_mask)):
                    return 1
                
        # High shape similarity case
        if calculate_shape_similarity(obj1, obj2) > 0.8:
            return 1
        
        # Shape type similarity
        if hasattr(obj1, 'shape') and hasattr(obj2, 'shape'):
            if obj1.shape != 'complex' and obj2.shape != 'complex' and obj1.shape == obj2.shape:
                score += 0.3
            weights_sum += 0.3
        
        # Size similarity
        if hasattr(obj1, 'size') and hasattr(obj2, 'size'):
            if obj1.size > 0 and obj2.size > 0:
                size_ratio = min(obj1.size, obj2.size) / max(obj1.size, obj2.size)
                score += 0.2 * size_ratio
            weights_sum += 0.2
        
        # Color similarity
        if hasattr(obj1, 'colors') and hasattr(obj2, 'colors'):
            if obj1.colors == obj2.colors:
                score += 0.3
            weights_sum += 0.3
        
        # Position similarity (lower weight for different grid sizes)
        if hasattr(obj1, 'center') and hasattr(obj2, 'center') and self.grid_diff.same_shape:
            max_dist = max(self.input_grid.shape)
            if max_dist > 0:
                dist = np.linalg.norm(np.array(obj1.center) - np.array(obj2.center))
                position_sim = 1.0 - min(dist / max_dist, 1.0)
                score += 0.2 * position_sim
            weights_sum += 0.2
        
        # Normalize by total weights
        return score / weights_sum if weights_sum > 0 else 0.0
    
    def _find_property_changes(self, in_obj, out_obj) -> Dict[str, Tuple[Any, Any]]:
        """Identify changed properties between two objects."""
        changes = {}
        
        properties = ['size', 'colors', 'color_numbers', 'shape', 'center', 'symmetry', 
                      'hor_size', 'vert_size', 'positioning', 'compactness']
        
        for prop in properties:
            if hasattr(in_obj, prop) and hasattr(out_obj, prop):
                in_val = getattr(in_obj, prop)
                out_val = getattr(out_obj, prop)
                
                # Handle different types appropriately
                if isinstance(in_val, np.ndarray) and isinstance(out_val, np.ndarray):
                    if not np.array_equal(in_val, out_val):
                        changes[prop] = (in_val.tolist() if in_val.ndim > 0 else in_val, 
                                       out_val.tolist() if out_val.ndim > 0 else out_val)
                elif in_val != out_val:
                    changes[prop] = (in_val, out_val)
        
        return changes
    
    def _analyze_object_context(self, obj, grid_summary, obj_type: str = 'input') -> Dict[str, Any]:
        """Analyze contextual properties and relations of an object."""
        context = {
            'relations': defaultdict(list),
            'properties': {},
            'nearby_objects': []
        }
        
        # Get representation level objects and triples
        level_data = grid_summary.repr_levels[self.levels[0]]
        all_objects = level_data.objects
        triples = level_data.triples
        
        # Extract object properties
        if hasattr(obj, 'size'):
            context['properties']['size'] = obj.size
        if hasattr(obj, 'colors'):
            context['properties']['colors'] = obj.colors
        if hasattr(obj, 'shape'):
            context['properties']['shape'] = obj.shape
        if hasattr(obj, 'center'):
            context['properties']['center'] = obj.center
        if hasattr(obj, 'symmetry'):
            context['properties']['symmetry'] = obj.symmetry
        if hasattr(obj, 'inner_holes'):
            context['properties']['inner_holes_count'] = len(obj.inner_holes)
        if hasattr(obj, 'outer_holes'):
            context['properties']['outer_holes_count'] = len(obj.outer_holes)
        
        # Get relations involving this object
        obj_triples = triples.get_triples_for_object(obj.label)
        for triple in obj_triples:
            head, relation, tail = triple
            context['relations'][relation].append(tail)
        
        # Find nearby objects (within certain distance threshold)
        distances = level_data.distances
        for other_obj in all_objects:
            if other_obj.label == obj.label:
                continue
            dist = distances.get_distance(obj.label, other_obj.label)
            if dist <= 3:  # Threshold for "nearby"
                context['nearby_objects'].append({
                    'label': other_obj.label,
                    'distance': dist,
                    'shape': other_obj.shape if hasattr(other_obj, 'shape') else None,
                    'colors': other_obj.colors if hasattr(other_obj, 'colors') else None
                })
        
        return context
    
    def _find_addition_patterns(self, added_changes: List[ObjectChange]) -> List[TransformationPattern]:
        """Analyze why objects were added - find common patterns."""
        patterns = []
        
        if not added_changes:
            return patterns
        
        # Analyze each added object's context
        added_contexts = []
        for change in added_changes:
            added_obj = change.output_obj
            # Check if it relates to existing input objects
            input_objs = self.input_summary.repr_levels[self.levels[0]].objects
            
            # Find spatial relationships
            best_alignment = None
            best_alignment_score = 0
            shape_matches = []
            color_matches = []
            
            for inp_obj in input_objs:
                # Check alignment
                if hasattr(added_obj, 'center') and hasattr(inp_obj, 'center'):
                    if added_obj.center[0] == inp_obj.center[0]:  # x-aligned
                        alignment_score = 0.5
                        if added_obj.center[1] == inp_obj.center[1]:  # Same position
                            alignment_score = 1.0
                        if alignment_score > best_alignment_score:
                            best_alignment_score = alignment_score
                            best_alignment = ('x_aligned', inp_obj.label)
                    elif added_obj.center[1] == inp_obj.center[1]:  # y-aligned
                        best_alignment = ('y_aligned', inp_obj.label)
                        best_alignment_score = 0.5
                
                # Check shape similarity
                if hasattr(added_obj, 'shape') and hasattr(inp_obj, 'shape'):
                    if added_obj.shape == inp_obj.shape:
                        shape_matches.append(inp_obj.label)
                
                # Check color similarity
                if hasattr(added_obj, 'colors') and hasattr(inp_obj, 'colors'):
                    if added_obj.colors == inp_obj.colors:
                        color_matches.append(inp_obj.label)
            
            added_contexts.append({
                'added_obj': added_obj,
                'alignment': best_alignment,
                'alignment_score': best_alignment_score,
                'shape_matches': shape_matches,
                'color_matches': color_matches
            })
        
        # Find common patterns across added objects
        all_alignments = [ctx['alignment'] for ctx in added_contexts if ctx['alignment']]
        if len(all_alignments) == len(added_changes):
            alignment_types = [a[0] for a in all_alignments]
            if len(set(alignment_types)) == 1:
                patterns.append(TransformationPattern(
                    pattern_type='aligned_addition',
                    description=f'All {len(added_changes)} added objects are {alignment_types[0]} with existing objects',
                    confidence=0.9,
                    parameters={'alignment_type': alignment_types[0], 'count': len(added_changes)}
                ))
        
        # Check if added objects duplicate existing shapes
        shape_duplication_count = sum(1 for ctx in added_contexts if ctx['shape_matches'])
        if shape_duplication_count == len(added_changes):
            patterns.append(TransformationPattern(
                pattern_type='shape_duplication',
                description=f'All {len(added_changes)} added objects duplicate existing shapes',
                confidence=0.9,
                parameters={'count': len(added_changes)}
            ))
        
        return patterns
    
    def _find_deletion_patterns(self, deleted_changes: List[ObjectChange]) -> List[TransformationPattern]:
        """Analyze why objects were deleted - find common patterns."""
        patterns = []
        
        if not deleted_changes:
            return patterns
        
        # Analyze properties of deleted objects
        deleted_properties = defaultdict(list)
        for change in deleted_changes:
            deleted_obj = change.input_obj
            
            if hasattr(deleted_obj, 'colors'):
                deleted_properties['colors'].append(deleted_obj.colors)
            if hasattr(deleted_obj, 'shape'):
                deleted_properties['shape'].append(deleted_obj.shape)
            if hasattr(deleted_obj, 'size'):
                deleted_properties['size'].append(deleted_obj.size)
            if hasattr(deleted_obj, 'positioning'):
                for pos in deleted_obj.positioning:
                    deleted_properties['positioning'].append(pos)
        
        # Find common deletion criteria
        # Same color deleted
        if 'colors' in deleted_properties:
            unique_colors = set([str(c) for c in deleted_properties['colors']])
            if len(unique_colors) == 1:
                patterns.append(TransformationPattern(
                    pattern_type='color_based_deletion',
                    description=f'All deleted objects have color {deleted_properties["colors"][0]}',
                    confidence=0.9,
                    parameters={'color': deleted_properties['colors'][0], 'count': len(deleted_changes)}
                ))
        
        # Same shape deleted
        if 'shape' in deleted_properties:
            unique_shapes = set(deleted_properties['shape'])
            if len(unique_shapes) == 1:
                patterns.append(TransformationPattern(
                    pattern_type='shape_based_deletion',
                    description=f'All deleted objects are {deleted_properties["shape"][0]}',
                    confidence=0.9,
                    parameters={'shape': deleted_properties['shape'][0], 'count': len(deleted_changes)}
                ))
        
        # Position-based deletion
        if 'positioning' in deleted_properties:
            position_counter = Counter(deleted_properties['positioning'])
            common_positions = [pos for pos, count in position_counter.items() if count == len(deleted_changes)]
            if common_positions:
                patterns.append(TransformationPattern(
                    pattern_type='position_based_deletion',
                    description=f'All deleted objects share positioning: {common_positions}',
                    confidence=0.85,
                    parameters={'positions': common_positions, 'count': len(deleted_changes)}
                ))
        
        return patterns
    
    def _find_modification_causal_patterns(self, modified_changes: List[ObjectChange]) -> List[TransformationPattern]:
        """Find causal relationships in object modifications (e.g., shift by number of holes)."""
        patterns = []
        
        if not modified_changes:
            return patterns
        
        # True shifts: center changed AND size/shape unchanged
        actual_shifts = []
        
        for change in modified_changes:
            if 'center' not in change.property_changes:
                continue
            
            # Only count as shift if size and shape stayed the same
            size_changed = 'size' in change.property_changes
            shape_changed = 'shape' in change.property_changes
            
            if size_changed or shape_changed:
                continue  # Not a pure shift
            
            old_center, new_center = change.property_changes['center']
            shift = (new_center[0] - old_center[0], new_center[1] - old_center[1])
            shift_magnitude = abs(shift[0]) + abs(shift[1])
            
            if shift_magnitude == 0:
                continue
            
            inp_obj = change.input_obj
            
            # Check various property relationships
            relations = {}
            
            if hasattr(inp_obj, 'inner_holes'):
                holes_count = len(inp_obj.inner_holes)
                if holes_count > 0 and shift_magnitude == holes_count:
                    relations['shift_equals_inner_holes'] = True
            
            if hasattr(inp_obj, 'outer_holes'):
                outer_holes_count = len(inp_obj.outer_holes)
                if outer_holes_count > 0 and shift_magnitude == outer_holes_count:
                    relations['shift_equals_outer_holes'] = True
            
            if hasattr(inp_obj, 'size'):
                if shift_magnitude == inp_obj.size:
                    relations['shift_equals_size'] = True
            
            if hasattr(inp_obj, 'hor_size'):
                if abs(shift[1]) == inp_obj.hor_size:
                    relations['horizontal_shift_equals_hor_size'] = True
            
            if hasattr(inp_obj, 'vert_size'):
                if abs(shift[0]) == inp_obj.vert_size:
                    relations['vertical_shift_equals_vert_size'] = True
            
            # Check distance changes with other objects
            if hasattr(change, 'input_obj') and hasattr(change, 'output_obj'):
                input_objs = self.input_summary.repr_levels[self.primary_level].objects
                output_objs = self.output_summary.repr_levels[self.primary_level].objects
                
                # Find distance changes
                for other_inp_obj in input_objs:
                    if other_inp_obj.label == inp_obj.label:
                        continue
                    
                    # Try to find corresponding output object
                    for other_out_obj in output_objs:
                        if other_out_obj.label == other_inp_obj.label or \
                           self._calculate_object_similarity(other_inp_obj, other_out_obj) > 0.8:
                            
                            # Calculate distance changes
                            inp_dist = self.input_summary.repr_levels[self.primary_level].distances.get_distance(
                                inp_obj.label, other_inp_obj.label
                            )
                            out_dist = self.output_summary.repr_levels[self.primary_level].distances.get_distance(
                                change.output_obj.label, other_out_obj.label
                            )
                            
                            if inp_dist > 0 and out_dist < inp_dist:
                                distance_decrease = inp_dist - out_dist
                                if shift_magnitude == distance_decrease:
                                    relations[f'shift_decreases_distance_to_{other_inp_obj.shape}'] = True
                            break
            
            actual_shifts.append({
                'object': inp_obj.label,
                'shift': shift,
                'shift_magnitude': shift_magnitude,
                'relations': relations
            })
        
        # Find common causal patterns
        if actual_shifts:
            # Check if all shifts follow the same rule
            all_relations = [set(spr['relations'].keys()) for spr in actual_shifts if spr['relations']]
            
            if all_relations:
                common_relations = set.intersection(*all_relations)
                
                for relation in common_relations:
                    # Create more readable descriptions
                    readable_rule = relation.replace('_', ' ')
                    
                    patterns.append(TransformationPattern(
                        pattern_type='causal_shift',
                        description=f'Objects shifted based on: {readable_rule}',
                        confidence=1.0,
                        parameters={
                            'rule': relation,
                            'count': len([spr for spr in actual_shifts if relation in spr['relations']]),
                            'shifts': [spr['shift'] for spr in actual_shifts]
                        }
                    ))
        
        # Check for uniform shifts
        all_shifts = [spr['shift'] for spr in actual_shifts]
        if all_shifts and len(set(all_shifts)) == 1:
            patterns.append(TransformationPattern(
                pattern_type='uniform_translation',
                description=f'All objects shifted uniformly by {all_shifts[0]}',
                confidence=1.0,
                parameters={'shift': all_shifts[0], 'count': len(all_shifts)}
            ))
        
        return patterns
    
    def _detect_patterns_at_level(self, level: int) -> List[TransformationPattern]:
        """Detect transformation patterns at a specific level."""
        patterns = []
        changes = self.object_changes.get(level, [])
        
        if not changes:
            return patterns
        
        # Pattern 1: Color changes
        color_changes = [c for c in changes if 'colors' in c.property_changes or 'color_numbers' in c.property_changes]
        if color_changes:
            color_map = defaultdict(list)
            for change in color_changes:
                if 'colors' in change.property_changes:
                    old_colors = change.property_changes['colors'][0]
                    new_colors = change.property_changes['colors'][1]
                    # Handle single vs multiple colors
                    old_key = tuple(old_colors) if isinstance(old_colors, (list, tuple)) else (old_colors,)
                    new_val = tuple(new_colors) if isinstance(new_colors, (list, tuple)) else (new_colors,)
                    color_map[old_key].append(new_val)
            
            # Check if consistent color mapping
            if color_map:
                consistent = all(len(set(v)) == 1 for v in color_map.values())
                confidence = 1.0 if consistent else 0.5
                
                patterns.append(TransformationPattern(
                    pattern_type='color_mapping',
                    description=f'Color mapping detected: {len(color_changes)} objects changed color',
                    confidence=confidence,
                    parameters={'color_map': dict(color_map), 'consistent': consistent}
                ))
        
        # Pattern 2: Object addition/deletion with deep analysis
        added = [c for c in changes if c.change_type == 'added']
        deleted = [c for c in changes if c.change_type == 'deleted']
        unchanged = [c for c in changes if c.change_type == 'unchanged']
        modified = [c for c in changes if c.change_type == 'modified']
        
        # Analyze additions in context
        if added:
            addition_patterns = self._find_addition_patterns(added)
            patterns.extend(addition_patterns)
            
            # Generic addition pattern if no specific pattern found
            if not addition_patterns:
                patterns.append(TransformationPattern(
                    pattern_type='object_addition',
                    description=f'Added {len(added)} objects',
                    confidence=0.6,
                    parameters={'count': len(added), 'added_objects': [c.object_label for c in added]}
                ))
        
        # Analyze deletions in context
        if deleted:
            deletion_patterns = self._find_deletion_patterns(deleted)
            patterns.extend(deletion_patterns)
            
            # Generic deletion pattern if no specific pattern found
            if not deletion_patterns:
                patterns.append(TransformationPattern(
                    pattern_type='object_deletion',
                    description=f'Deleted {len(deleted)} objects',
                    confidence=0.6,
                    parameters={'count': len(deleted), 'deleted_objects': [c.object_label for c in deleted]}
                ))
        
        # Analyze modifications for causal patterns
        if modified:
            modification_patterns = self._find_modification_causal_patterns(modified)
            patterns.extend(modification_patterns)
        
        # Pattern 3: Size scaling
        size_changes = [c for c in changes if 'size' in c.property_changes]
        if size_changes:
            ratios = []
            for c in size_changes:
                old_size, new_size = c.property_changes['size']
                if old_size > 0:
                    ratios.append(new_size / old_size)
            
            if ratios:
                unique_ratios = set(ratios)
                if len(unique_ratios) == 1:
                    patterns.append(TransformationPattern(
                        pattern_type='size_scaling',
                        description=f'All objects scaled by factor {ratios[0]:.2f}',
                        confidence=1.0,
                        parameters={'scale_factor': ratios[0]}
                    ))
                elif ratios:
                    patterns.append(TransformationPattern(
                        pattern_type='size_scaling',
                        description=f'Objects scaled by various factors (mean: {np.mean(ratios):.2f})',
                        confidence=0.5,
                        parameters={'scale_factors': ratios, 'mean_factor': np.mean(ratios)}
                    ))
        
        # Pattern 4: Position changes (after causal analysis)
        position_changes = [c for c in changes if 'center' in c.property_changes]
        if position_changes and self.grid_diff.same_shape:
            # Only add generic position pattern if no causal pattern was found
            has_causal_pattern = any(p.pattern_type == 'causal_shift' for p in patterns)
            if not has_causal_pattern:
                offsets = []
                for c in position_changes:
                    old_center, new_center = c.property_changes['center']
                    offset = (new_center[0] - old_center[0], new_center[1] - old_center[1])
                    offsets.append(offset)
                
                if offsets:
                    unique_offsets = set(offsets)
                    if len(unique_offsets) == 1:
                        patterns.append(TransformationPattern(
                            pattern_type='translation',
                            description=f'All objects translated by {offsets[0]}',
                            confidence=1.0,
                            parameters={'offset': offsets[0]}
                        ))
        
        # Pattern 5: Symmetry changes
        symmetry_changes = [c for c in changes if 'symmetry' in c.property_changes]
        if symmetry_changes:
            patterns.append(TransformationPattern(
                pattern_type='symmetry_change',
                description=f'Symmetry changed in {len(symmetry_changes)} objects',
                confidence=0.7,
                parameters={'count': len(symmetry_changes)}
            ))
        
        return patterns
    
    def get_summary(self) -> str:
        """Generate a human-readable summary of the transformation."""
        summary = [f"Example {self.example_id} Analysis:"]
        summary.append(f"Grid shape: {self.input_grid.shape} -> {self.output_grid.shape}")
        summary.append(f"Changed cells: {self.grid_diff.num_changes} ({self.grid_diff.change_ratio:.1%})")
        
        for level in self.levels:
            changes = self.object_changes.get(level, [])
            added = sum(1 for c in changes if c.change_type == 'added')
            deleted = sum(1 for c in changes if c.change_type == 'deleted')
            modified = sum(1 for c in changes if c.change_type == 'modified')
            unchanged = sum(1 for c in changes if c.change_type == 'unchanged')
            
            summary.append(f"\nLevel {level} ({len(changes)} objects):")
            summary.append(f"  Added: {added}, Deleted: {deleted}, Modified: {modified}, Unchanged: {unchanged}")
        
        if self.transformation_patterns:
            summary.append("\nDetected Patterns:")
            for pattern in self.transformation_patterns:
                summary.append(f"  - {pattern.description} (confidence: {pattern.confidence:.2f})")
        else:
            summary.append("\nNo clear patterns detected")
        
        return "\n".join(summary)
    
    def debug_object_change(self, example_id: int, object_label: str) -> str:
        """Get detailed debug information about a specific object change."""
        if example_id >= len(self.subtasks_analyses):
            return f"Example {example_id} not found"
        
        analysis = self.subtasks_analyses[example_id]
        
        # Find the object change
        object_change = None
        for level in analysis.object_changes:
            for change in analysis.object_changes[level]:
                if object_label in change.object_label:
                    object_change = change
                    break
            if object_change:
                break
        
        if not object_change:
            return f"Object {object_label} not found in example {example_id}"
        
        debug_info = [f"Debug Info for {object_label} in Example {example_id}:"]
        debug_info.append(f"Change Type: {object_change.change_type}")
        debug_info.append(f"Similarity Score: {object_change.similarity_score:.3f}")
        
        if object_change.input_obj:
            debug_info.append("\nInput Object Properties:")
            debug_info.append(f"  Shape: {object_change.input_obj.shape}")
            debug_info.append(f"  Size: {object_change.input_obj.size}")
            debug_info.append(f"  Colors: {object_change.input_obj.colors}")
            if hasattr(object_change.input_obj, 'center'):
                debug_info.append(f"  Center: {object_change.input_obj.center}")
            if hasattr(object_change.input_obj, 'symmetry'):
                debug_info.append(f"  Symmetry: {object_change.input_obj.symmetry}")
            if hasattr(object_change.input_obj, 'inner_holes'):
                debug_info.append(f"  Inner Holes: {len(object_change.input_obj.inner_holes)}")
        
        if object_change.output_obj:
            debug_info.append("\nOutput Object Properties:")
            debug_info.append(f"  Shape: {object_change.output_obj.shape}")
            debug_info.append(f"  Size: {object_change.output_obj.size}")
            debug_info.append(f"  Colors: {object_change.output_obj.colors}")
            if hasattr(object_change.output_obj, 'center'):
                debug_info.append(f"  Center: {object_change.output_obj.center}")
            if hasattr(object_change.output_obj, 'symmetry'):
                debug_info.append(f"  Symmetry: {object_change.output_obj.symmetry}")
            if hasattr(object_change.output_obj, 'inner_holes'):
                debug_info.append(f"  Inner Holes: {len(object_change.output_obj.inner_holes)}")
        
        if object_change.property_changes:
            debug_info.append("\nProperty Changes:")
            for prop, (old_val, new_val) in object_change.property_changes.items():
                debug_info.append(f"  {prop}: {old_val} → {new_val}")
        
        return "\n".join(debug_info)


class TaskAnalysis:
    """Analyzes a complete ARC task with multiple training examples and test cases."""
    
    def __init__(self, task, font_color: int = 0, levels: List[int] = [2]):
        """
        Initialize task analysis.
        
        Args:
            task: ARCTask instance with subtasks
            font_color: Background color to ignore (default 0)
            levels: Representation levels to analyze
        """
        self.task = task
        self.task_id = task.label
        self.font_color = font_color
        self.levels = levels
        self.train_subtasks = task.subtasks
        self.subtasks_analyses = []
        
        # Analyze each subtask
        for subtask in self.train_subtasks:
            analysis = SubtaskAnalysis(subtask, font_color, levels)
            self.subtasks_analyses.append(analysis)
        
        # Infer consistent patterns across training examples
        self.consistent_patterns = self._infer_consistent_patterns()
        self.transformation_rules = self._synthesize_transformation_rules()
    
    def _infer_consistent_patterns(self) -> List[TransformationPattern]:
        """Find patterns that appear consistently across training examples."""
        if not self.subtasks_analyses:
            return []
        
        # Collect all patterns from all subtasks
        pattern_groups = defaultdict(list)
        
        for subtask_analysis in self.subtasks_analyses:
            for pattern in subtask_analysis.transformation_patterns:
                pattern_groups[pattern.pattern_type].append(pattern)
        
        # Keep patterns that appear in all or most examples
        threshold = max(1, len(self.subtasks_analyses) * 0.5)  # At least 50% of examples
        
        consistent = []
        for pattern_type, patterns in pattern_groups.items():
            if len(patterns) >= threshold:
                # Merge patterns of same type
                avg_confidence = np.mean([p.confidence for p in patterns])
                
                # Merge parameters
                merged_params = {}
                for pattern in patterns:
                    for k, v in pattern.parameters.items():
                        if k not in merged_params:
                            merged_params[k] = []
                        merged_params[k].append(v)
                
                # Check parameter consistency
                param_consistency = {}
                for k, v in merged_params.items():
                    unique_values = len(set([str(x) for x in v]))  # Convert to string for comparison
                    param_consistency[k] = unique_values == 1
                
                overall_consistency = all(param_consistency.values()) if param_consistency else False
                adjusted_confidence = avg_confidence * (1.0 if overall_consistency else 0.7)
                
                consistent.append(TransformationPattern(
                    pattern_type=pattern_type,
                    description=f'{pattern_type} (appears in {min(len(patterns),len(self.subtasks_analyses))}/{len(self.subtasks_analyses)} examples)',
                    confidence=adjusted_confidence,
                    parameters={'values': merged_params, 'consistency': param_consistency}
                ))
        
        return sorted(consistent, key=lambda x: x.confidence, reverse=True)
    
    def _synthesize_transformation_rules(self) -> List[str]:
        """Synthesize high-level transformation rules from consistent patterns."""
        rules = []
        
        for pattern in self.consistent_patterns:
            if pattern.confidence < 0.5:
                continue
                
            if pattern.pattern_type == 'color_mapping':
                rules.append("Apply consistent color mapping")
            elif pattern.pattern_type == 'size_scaling':
                rules.append("Scale objects (pattern detected in multiple examples)")
            elif pattern.pattern_type == 'object_addition':
                rules.append("Add new objects to the output")
            elif pattern.pattern_type == 'object_deletion':
                rules.append("Remove objects from the input")
            elif pattern.pattern_type == 'translation':
                rules.append("Translate objects by consistent offset")
        
        # Grid shape rules
        if self.subtasks_analyses:
            shape_changes = [s.grid_diff.has_size_change for s in self.subtasks_analyses]
            if all(shape_changes):
                rules.append("Output grid has different size than input (all examples)")
            elif not any(shape_changes):
                rules.append("Output grid preserves input dimensions (all examples)")
            else:
                rules.append("Output grid size varies (inconsistent across examples)")
        
        return rules
    
    def get_task_summary(self) -> str:
        """Generate comprehensive task summary."""
        summary = [f"Task {self.task_id} Analysis:"]
        summary.append(f"Training examples: {len(self.subtasks_analyses)}")
        
        summary.append("\n" + "="*60)
        summary.append("Per-Example Analysis:")
        summary.append("="*60)
        for subtask_analysis in self.subtasks_analyses:
            summary.append("\n" + subtask_analysis.get_summary())
        
        summary.append("\n" + "="*60)
        summary.append("Consistent Patterns Across Examples:")
        summary.append("="*60)
        if self.consistent_patterns:
            for pattern in self.consistent_patterns:
                summary.append(f"  [{pattern.confidence:.2f}] {pattern.description}")
        else:
            summary.append("  No consistent patterns detected")
        
        summary.append("\n" + "="*60)
        summary.append("Inferred Transformation Rules:")
        summary.append("="*60)
        if self.transformation_rules:
            for i, rule in enumerate(self.transformation_rules, 1):
                summary.append(f"  {i}. {rule}")
        else:
            summary.append("  No clear transformation rules identified")
        
        return "\n".join(summary)
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistical summary of detected patterns."""
        stats = {
            'total_examples': len(self.subtasks_analyses),
            'pattern_types': defaultdict(int),
            'avg_confidence_by_type': {},
            'grid_size_changes': [],
            'avg_cell_changes': []
        }
        
        for analysis in self.subtasks_analyses:
            stats['grid_size_changes'].append(analysis.grid_diff.has_size_change)
            stats['avg_cell_changes'].append(analysis.grid_diff.change_ratio)
            
            for pattern in analysis.transformation_patterns:
                stats['pattern_types'][pattern.pattern_type] += 1
        
        # Calculate average confidence by pattern type
        pattern_confidences = defaultdict(list)
        for analysis in self.subtasks_analyses:
            for pattern in analysis.transformation_patterns:
                pattern_confidences[pattern.pattern_type].append(pattern.confidence)
        
        for ptype, confidences in pattern_confidences.items():
            stats['avg_confidence_by_type'][ptype] = np.mean(confidences)
        
        return stats
    
    def get_transformation_hypothesis(self) -> str:
        """Generate a human-readable hypothesis about the transformation."""
        hypothesis_parts = []
        
        # Analyze consistent patterns to build hypothesis
        high_confidence_patterns = [p for p in self.consistent_patterns if p.confidence >= 0.8]
        medium_confidence_patterns = [p for p in self.consistent_patterns if 0.5 <= p.confidence < 0.8]
        
        if high_confidence_patterns:
            hypothesis_parts.append("HIGH CONFIDENCE RULES:")
            for pattern in high_confidence_patterns:
                if pattern.pattern_type == 'causal_shift':
                    rule = pattern.parameters.get('common_values', {}).get('rule', 'unknown')
                    hypothesis_parts.append(f"  • Objects are shifted based on: {rule.replace('_', ' ')}")
                
                elif pattern.pattern_type == 'aligned_addition':
                    alignment = pattern.parameters.get('common_values', {}).get('alignment_type', 'unknown')
                    hypothesis_parts.append(f"  • New objects are added {alignment} with existing objects")
                
                elif pattern.pattern_type == 'shape_duplication':
                    hypothesis_parts.append("  • Shapes are duplicated from input to create new objects")
                
                elif pattern.pattern_type == 'color_based_deletion':
                    color = pattern.parameters.get('common_values', {}).get('color', 'unknown')
                    hypothesis_parts.append(f"  • Objects with color {color} are removed")
                
                elif pattern.pattern_type == 'shape_based_deletion':
                    shape = pattern.parameters.get('common_values', {}).get('shape', 'unknown')
                    hypothesis_parts.append(f"  • All {shape} objects are removed")
                
                elif pattern.pattern_type == 'uniform_translation':
                    shift = pattern.parameters.get('common_values', {}).get('shift', (0, 0))
                    hypothesis_parts.append(f"  • All objects are translated by {shift}")
                
                elif pattern.pattern_type == 'color_mapping':
                    hypothesis_parts.append("  • Colors are mapped according to consistent rules")
                
                elif pattern.pattern_type == 'size_scaling':
                    factor = pattern.parameters.get('common_values', {}).get('scale_factor', 1.0)
                    if isinstance(factor, (int, float)):
                        hypothesis_parts.append("  • Objects are scaled by factor {factor:.2f}")
                    else:
                        hypothesis_parts.append("  • Objects are scaled (factor varies)")
        
        if medium_confidence_patterns:
            hypothesis_parts.append("\nMEDIUM CONFIDENCE OBSERVATIONS:")
            for pattern in medium_confidence_patterns:
                hypothesis_parts.append(f"  • {pattern.description}")
        
        # Add grid size observations
        size_changes = [a.grid_diff.has_size_change for a in self.subtasks_analyses]
        if all(size_changes):
            hypothesis_parts.append("\nGRID OBSERVATIONS:")
            hypothesis_parts.append("  • Output grid size always differs from input")
        elif not any(size_changes):
            hypothesis_parts.append("\nGRID OBSERVATIONS:")
            hypothesis_parts.append("  • Output grid size always matches input")
        
        if not hypothesis_parts:
            return "No clear transformation hypothesis could be established. The transformation may be highly variable or complex."
        
        return "\n".join(hypothesis_parts)
    
    def get_actionable_insights(self) -> List[str]:
        """Extract actionable transformation steps that could be programmed."""
        insights = []
        
        for pattern in self.consistent_patterns:
            if pattern.confidence < 0.7:
                continue
            
            if pattern.pattern_type == 'causal_shift':
                rule = pattern.parameters.get('common_values', {}).get('rule', '')
                if 'inner_holes' in rule:
                    insights.append("For each object: shift_amount = count(inner_holes)")
                elif 'size' in rule:
                    insights.append("For each object: shift_amount = object.size")
                elif 'hor_size' in rule:
                    insights.append("For each object: horizontal_shift = object.hor_size")
                elif 'vert_size' in rule:
                    insights.append("For each object: vertical_shift = object.vert_size")
            
            elif pattern.pattern_type == 'aligned_addition':
                alignment = pattern.parameters.get('common_values', {}).get('alignment_type', '')
                if 'x_aligned' in alignment:
                    insights.append("Create new objects x-aligned with existing objects")
                elif 'y_aligned' in alignment:
                    insights.append("Create new objects y-aligned with existing objects")
            
            elif pattern.pattern_type == 'shape_duplication':
                insights.append("Duplicate shapes from input (possibly with transformations)")
            
            elif pattern.pattern_type == 'color_based_deletion':
                color = pattern.parameters.get('common_values', {}).get('color', '')
                insights.append(f"Delete all objects with color: {color}")
            
            elif pattern.pattern_type == 'shape_based_deletion':
                shape = pattern.parameters.get('common_values', {}).get('shape', '')
                insights.append(f"Delete all objects with shape: {shape}")
            
            elif pattern.pattern_type == 'uniform_translation':
                shift = pattern.parameters.get('common_values', {}).get('shift', (0, 0))
                insights.append(f"Translate all objects by offset: {shift}")
            
            elif pattern.pattern_type == 'color_mapping':
                insights.append("Apply color mapping transformation to all objects")
            
            elif pattern.pattern_type == 'size_scaling':
                factor = pattern.parameters.get('common_values', {}).get('scale_factor', '')
                if isinstance(factor, (int, float)):
                    insights.append(f"Scale all objects by factor: {factor:.2f}")
        
        return insights