"""
Symbolic solvers for ARC tasks, wrapped as classes with one uniform
contract: every `.solve(task)` call returns a `SolveResult` — either a
solved grid, or a debug string explaining why not. No solver raises for a
"no answer" case, and none silently produces a blank/wrong grid when its
internal checks fail — the reason is captured as an actual message instead.

Three solvers, aggregated (not orchestrated — no dispatch logic, just
attribute access) under SymbolicModule:
    SymbolicModule().mixer.solve(task)
    SymbolicModule().upscale_or_covering.solve(task)
    SymbolicModule().color_restore.solve(task)
upscale_or_covering merges UpscaleSolver and PatternPlantingSolver (aka
"Covering") into a single dispatch step — different underlying logic each,
but per-agent module wiring only needs one cheap "did symbolic solve it"
call rather than a choice between the two.
"""
from __future__ import annotations

import numpy as np
from copy import copy, deepcopy
from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations, product
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SolveResult:
    """Every solver returns this: either a solved grid, or a debug string
    explaining why it couldn't produce one."""
    success: bool
    grid: Optional[np.ndarray] = None
    debug: str = ""

    @classmethod
    def ok(cls, grid: np.ndarray, debug: str = "solved") -> "SolveResult":
        return cls(success=True, grid=grid, debug=debug)

    @classmethod
    def fail(cls, debug: str) -> "SolveResult":
        return cls(success=False, grid=None, debug=debug)


def invert_pattern(pattern: np.ndarray, font_color) -> np.ndarray:
    """Swap background/foreground: shared by upscale and pattern planting."""
    inverted = pattern.copy()
    fg_colors = np.unique(pattern[pattern != font_color])
    if len(fg_colors) == 0:
        return inverted
    fg_color = fg_colors[0]
    inverted[pattern == font_color] = fg_color
    inverted[pattern != font_color] = font_color
    return inverted


# ============================================================================
# UPSCALE
# ============================================================================

class UpscaleSolver:
    """Detects a per-color (or whole-pattern) upscaling rule from training
    examples and applies it to the test input."""

    def __init__(self, font_color: int = 0):
        self.font_color = font_color

    def solve(self, task) -> SolveResult:
        try:
            subtasks = task.subtasks
            test_subtask = task.test_subtask
            scales = self._define_examples_scales(subtasks)

            pattern_candidates = self._extract_pattern_from_output(subtasks, scales)
            all_mappings = self._get_mappings(subtasks, scales, pattern_candidates)
            if all_mappings is None:
                return SolveResult.fail("no training examples to derive a color mapping from")

            replacement_map = self._build_replacement_map(subtasks, test_subtask, all_mappings)
            if not replacement_map:
                return SolveResult.fail(
                    "could not determine a consistent color->pattern replacement map "
                    "from the training examples"
                )

            answer = self._apply_upscale(test_subtask.train_inp, subtasks, scales,
                                          pattern_candidates, replacement_map)
            return SolveResult.ok(answer)

        except Exception as e:  # last resort: never propagate, always explain
            return SolveResult.fail(f"upscale solver raised {type(e).__name__}: {e}")

    # -- training-example analysis -----------------------------------------

    def _define_examples_scales(self, subtasks) -> Dict[int, Tuple[float, float]]:
        scales = {}
        for idx, subtask in enumerate(subtasks):
            inp, out = subtask.train_inp, subtask.train_out
            scales[idx] = (out.shape[0] / inp.shape[0], out.shape[1] / inp.shape[1])
        return scales

    def _extract_pattern_from_output(self, subtasks, scales) -> Dict[int, np.ndarray]:
        """Top-left scale x scale block that differs from the input cell it
        replaces, taken as the repeating pattern — if a subtask has two
        conflicting candidate blocks, the whole extraction is abandoned."""
        candidates: Dict[int, np.ndarray] = {}
        for idx, subtask in enumerate(subtasks):
            inp, out = subtask.train_inp, subtask.train_out
            scale_i, scale_j = int(scales[idx][0]), int(scales[idx][1])
            if scale_i == 0 or scale_j == 0:
                continue  # non-integer scaling — handled separately in apply_upscale
            for j in range(inp.shape[1]):
                for i in range(inp.shape[0]):
                    cell_val = inp[i, j]
                    i_out, j_out = i * scale_i, j * scale_j
                    block = out[i_out:i_out + scale_i, j_out:j_out + scale_j]
                    if (block != self.font_color).any() and (block != cell_val).all() \
                            and inp.shape != block.shape:
                        if idx in candidates and (candidates[idx] != block).any():
                            return {}
                        candidates.setdefault(idx, block)
        return candidates

    def _get_mappings(self, subtasks, scales, pattern_candidates) -> Optional[List[dict]]:
        use_pattern = len(pattern_candidates) == len(subtasks)
        all_mappings = []
        for idx, subtask in enumerate(subtasks):
            inp, out = subtask.train_inp, subtask.train_out
            scale_i, scale_j = int(scales[idx][0]), int(scales[idx][1])
            pattern = pattern_candidates.get(idx, inp) if use_pattern else inp
            mapping = {}
            for color in np.unique(inp):
                positions = np.argwhere(inp == color)
                blocks = []
                for i, j in positions:
                    out_i, out_j = i * scale_i, j * scale_j
                    blocks.append(out[out_i:out_i + scale_i, out_j:out_j + scale_j])
                if not blocks:
                    continue
                first_block = blocks[0]
                if not all(np.array_equal(b, first_block) for b in blocks):
                    continue  # inconsistent block for this color within this example
                if np.all(first_block == self.font_color):
                    mapping[color] = 'font'
                elif np.array_equal(first_block, inp):
                    mapping[color] = 'original'
                elif np.array_equal(first_block, pattern):
                    mapping[color] = 'output_pattern'
                elif np.array_equal(first_block, invert_pattern(inp, self.font_color)) or \
                        np.array_equal(first_block, invert_pattern(pattern, self.font_color)):
                    mapping[color] = 'inverted'
                elif np.unique(first_block).size and np.unique(first_block)[0] == color:
                    mapping[color] = 'color_upscale'
                else:
                    mapping[color] = None
            all_mappings.append(mapping)
        return all_mappings or None

    def _build_replacement_map(self, subtasks, test_subtask, all_mappings) -> Optional[dict]:
        replacement_map: Dict[Any, Any] = {}
        all_colors = set()
        for mapping in all_mappings:
            all_colors |= set(mapping.keys())

        test_unique_colors = np.unique(test_subtask.train_inp)

        if all_mappings and all('color_upscale' in mapping.values() for mapping in all_mappings):
            for color in test_unique_colors:
                replacement_map[color] = "color_upscale"
        elif self._need_ranking(all_mappings):
            replacement_map = self._ranking_mapping(subtasks, test_subtask.train_inp, all_mappings)
        else:
            for color in all_colors:
                values = [m[color] for m in all_mappings if color in m]
                if values:
                    replacement_map[color] = values[0]

        return replacement_map or None

    def _need_ranking(self, mappings: List[dict]) -> bool:
        key_lens = [len(m) for m in mappings]

        if len(set(key_lens)) > 1:
            max_len = max(key_lens)
            smaller = [m for m, kl in zip(mappings, key_lens) if kl < max_len]
            larger = [m for m, kl in zip(mappings, key_lens) if kl == max_len]
            if all(set(s.keys()) <= set(larger_map.keys()) for s in smaller for larger_map in larger):
                return False

        elif len(set(key_lens)) == 1:
            non_font_ranks = [k for m in mappings for k, v in m.items() if v != 'font']
            return key_lens[0] == 2 and len(set(non_font_ranks)) > 1

        unique_keys = set().union(*mappings) if mappings else set()
        for key in unique_keys:
            if not all(key in m for m in mappings):
                return True
            current_val = None
            for m in mappings:
                if key in m:
                    if current_val is None:
                        current_val = m[key]
                    elif current_val != m[key]:
                        return True
        return False

    def _create_ranking_dict(self, array: np.ndarray) -> Dict[Any, Any]:
        unique_elements, counts = np.unique(array, return_counts=True)
        order = np.argsort(counts)[::-1]
        sorted_elements = list(unique_elements[order])
        if self.font_color in sorted_elements:
            sorted_elements.remove(self.font_color)
        ranking = {}
        for idx, el in enumerate(sorted_elements):
            value = idx + 1
            if value == len(sorted_elements):  # fixed: was len(unique_elements)
                value = "rarest"
            ranking[el] = value
        return ranking

    def _ranking_mapping(self, subtasks, test_inp, all_mappings) -> Optional[dict]:
        replacement_map = {}
        test_ranking = self._create_ranking_dict(test_inp)
        test_unique_colors = np.unique(test_inp)
        colors_rankings = [self._create_ranking_dict(subtask.train_inp) for subtask in subtasks]

        used_ranks, all_non_font_replacements = [], []
        for idx, mapping in enumerate(all_mappings):
            used_ranks.extend(colors_rankings[idx][k] for k, v in mapping.items()
                               if v != 'font' and k != self.font_color)
            all_non_font_replacements.extend(v for v in mapping.values() if v != 'font')

        unique_ranks = set(used_ranks)
        unique_non_font = set(all_non_font_replacements)
        if len(unique_ranks) == 1 and len(unique_non_font) == 1:
            target_rank = unique_ranks.pop()
            inverted_ranking = {v: k for k, v in test_ranking.items()}
            target_color = inverted_ranking.get(target_rank)
            for color in test_unique_colors:
                replacement_map[color] = unique_non_font.copy().pop() if color == target_color else 'font'
        return replacement_map or None

    # -- applying the detected rule ------------------------------------------

    def _apply_upscale(self, inp, subtasks, scales, pattern_candidates, replacement_map) -> np.ndarray:
        unique_colors = np.unique(inp)

        if 'output_pattern' in replacement_map.values():
            pattern = next(iter(pattern_candidates.values()))
            scale_i, scale_j = pattern.shape
        elif 'color_upscale' in replacement_map.values():
            pattern = copy(inp)
            unique_scalings = list(set(scales.values()))
            if not float(scales[0][0]).is_integer():
                return self._non_int_scaling(inp, subtasks)
            elif len(unique_scalings) == 1:
                scale_i, scale_j = unique_scalings[0]
            elif all((len(np.unique(s.train_inp)) - 1) == scales[i][0] for i, s in enumerate(subtasks)):
                scale_i = scale_j = len(unique_colors) - 1
            elif all(len(np.unique(s.train_inp)) == scales[i][0] for i, s in enumerate(subtasks)):
                scale_i = scale_j = len(unique_colors)
            else:
                scale_i, scale_j = unique_scalings[0]
            scale_i, scale_j = int(scale_i), int(scale_j)
        else:
            pattern = copy(inp)
            scale_i, scale_j = pattern.shape

        h, w = inp.shape
        output = np.full((h * scale_i, w * scale_j), self.font_color, dtype=inp.dtype)

        for i in range(h):
            for j in range(w):
                color = inp[i, j]
                out_i, out_j = i * scale_i, j * scale_j
                if color not in replacement_map:
                    continue
                mapping = replacement_map[color]
                if mapping == 'original':
                    output[out_i:out_i + scale_i, out_j:out_j + scale_j] = inp
                elif mapping == 'output_pattern':
                    output[out_i:out_i + scale_i, out_j:out_j + scale_j] = pattern
                elif mapping == 'inverted':
                    output[out_i:out_i + scale_i, out_j:out_j + scale_j] = invert_pattern(pattern, self.font_color)
                elif mapping == 'font':
                    output[out_i:out_i + scale_i, out_j:out_j + scale_j] = self.font_color
                elif mapping == 'color_upscale':
                    output[out_i:out_i + scale_i, out_j:out_j + scale_j] = color

        return output

    def _non_int_scaling(self, inp, subtasks) -> np.ndarray:
        """Non-integer / uneven scaling, with the remainder distributed from
        the center outward, using the per-example step distribution computed
        by `_non_eq_scales`."""
        row_steps = self._non_eq_scales(subtasks, dim=0)
        col_steps = self._non_eq_scales(subtasks, dim=1)
        if row_steps is None or col_steps is None:
            raise ValueError("training examples don't agree on a single non-integer scaling")

        h, w = inp.shape
        out_h, out_w = sum(row_steps), sum(col_steps)
        output = np.zeros((out_h, out_w), dtype=inp.dtype)

        base_i = 0
        for i in range(h):
            base_j = 0
            for j in range(w):
                output[base_i:base_i + row_steps[i], base_j:base_j + col_steps[j]] = inp[i, j]
                base_j += col_steps[j]
            base_i += row_steps[i]
        return output

    def _non_eq_scales(self, subtasks, dim: int) -> Optional[List[int]]:
        """Per-example step distribution along one dimension, agreeing
        across all training examples, or None if they disagree."""
        all_steps = []
        for subtask in subtasks:
            p = subtask.train_inp.shape[dim]
            n = subtask.train_out.shape[dim]
            base, remainder = divmod(n, p)
            result = [base] * p
            for i in range(remainder):
                if i % 2 == 0:
                    result[i // 2] += 1
                else:
                    result[p - 1 - i // 2] += 1
            all_steps.append(result)
        if all(steps == all_steps[0] for steps in all_steps):
            return all_steps[0]
        return None


# ============================================================================
# PATTERN PLANTING (aka "Covering")
# ============================================================================

class PatternPlantingSolver:
    """Detects a placement strategy (which rotation/flip of the input goes
    in each output tile) and a tiling/scaling rule, then applies both to the
    test input."""

    VARIANT_NAMES = ('original', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', 'inverted')

    def __init__(self, font_color: int = 0):
        self.font_color = font_color

    def solve(self, task) -> SolveResult:
        try:
            subtasks = task.subtasks
            test_subtask = task.test_subtask

            strategy, scaling_rule, debug = self._analyze_planting_strategy(subtasks)
            if strategy is None:
                return SolveResult.fail(debug)

            answer = self._apply_planting_strategy(test_subtask.train_inp, strategy, scaling_rule)
            return SolveResult.ok(answer)

        except Exception as e:
            return SolveResult.fail(f"pattern planting solver raised {type(e).__name__}: {e}")

    def _generate_pattern_variants(self, pattern: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            'original': pattern.copy(),
            'rot90': np.rot90(pattern, 1),
            'rot180': np.rot90(pattern, 2),
            'rot270': np.rot90(pattern, 3),
            'flip_h': np.fliplr(pattern),
            'flip_v': np.flipud(pattern),
            'inverted': invert_pattern(pattern, self.font_color),
        }

    def _analyze_planting_strategy(self, subtasks):
        all_possible_strategies = []
        scaling_rules = []

        for subtask in subtasks:
            inp, out = subtask.train_inp, subtask.train_out
            if inp.shape[0] == 0 or out.shape[0] % inp.shape[0] or out.shape[1] % inp.shape[1]:
                return None, None, (
                    "training output shape isn't a whole multiple of the input shape "
                    "for at least one example"
                )
            i_steps = out.shape[0] // inp.shape[0]
            j_steps = out.shape[1] // inp.shape[1]
            scaling_rules.append({
                'input_shape': inp.shape, 'output_shape': out.shape,
                'i_steps': i_steps, 'j_steps': j_steps,
                'num_colors': len(np.unique(inp)),
            })

            variants = self._generate_pattern_variants(inp)
            possible_per_position = []
            for i in range(i_steps):
                for j in range(j_steps):
                    block = out[i * inp.shape[0]:(i + 1) * inp.shape[0],
                                j * inp.shape[1]:(j + 1) * inp.shape[1]]
                    matches = [name for name, var in variants.items() if np.array_equal(block, var)]
                    possible_per_position.append(matches or ['unknown'])
            all_possible_strategies.append(possible_per_position)

        # All training examples must agree on the number of output tiles —
        # a mismatch is a real inconsistency, not something to truncate to
        # the first example's count.
        position_counts = {len(s) for s in all_possible_strategies}
        if len(position_counts) > 1:
            return None, None, (
                f"training examples disagree on the number of output tiles "
                f"({sorted(position_counts)}) — can't infer one per-position strategy"
            )

        inferred_scaling = self._infer_scaling_rule(scaling_rules)
        if not all_possible_strategies:
            return [], inferred_scaling, ""

        num_positions = position_counts.pop()
        consistent_strategy = []
        for pos_idx in range(num_positions):
            possible_at_pos = [set(example[pos_idx]) for example in all_possible_strategies]
            consistent_mods = possible_at_pos[0]
            for pos_set in possible_at_pos[1:]:
                consistent_mods &= pos_set
            if not consistent_mods:
                return None, None, f"no transformation is consistent for output tile #{pos_idx}"
            if 'original' in consistent_mods:
                consistent_strategy.append('original')
            elif consistent_mods == {'unknown'}:
                # An unrecognized tile is a genuine failure, not something to
                # silently guess 'original' for.
                return None, None, (
                    f"output tile #{pos_idx} doesn't match any known rotation/flip/inversion "
                    f"of the input pattern"
                )
            else:
                consistent_strategy.append(sorted(consistent_mods)[0])

        return consistent_strategy, inferred_scaling, ""

    def _infer_scaling_rule(self, scaling_rules: List[dict]) -> dict:
        if not scaling_rules:
            return {'type': 'fixed', 'i_steps': 1, 'j_steps': 1}

        first = scaling_rules[0]
        if all(r['i_steps'] == first['i_steps'] and r['j_steps'] == first['j_steps'] for r in scaling_rules):
            return {'type': 'fixed', 'i_steps': first['i_steps'], 'j_steps': first['j_steps']}

        if all(r['i_steps'] * r['j_steps'] == r['num_colors'] for r in scaling_rules):
            return {'type': 'color_based_total'}
        if all(r['i_steps'] == r['num_colors'] and r['j_steps'] == r['num_colors'] for r in scaling_rules):
            return {'type': 'color_based_per_dim'}
        if all(r['i_steps'] == r['input_shape'][0] and r['j_steps'] == r['input_shape'][1] for r in scaling_rules):
            return {'type': 'dimension_based'}
        if all(r['i_steps'] * r['j_steps'] == r['input_shape'][0] * r['input_shape'][1] for r in scaling_rules):
            return {'type': 'area_based'}

        return {'type': 'fixed', 'i_steps': first['i_steps'], 'j_steps': first['j_steps']}

    def _calculate_output_dimensions(self, input_shape, scaling_rule, num_colors=None) -> Tuple[int, int]:
        h_in, w_in = input_shape
        rule_type = scaling_rule['type']

        if rule_type == 'fixed':
            return scaling_rule['i_steps'], scaling_rule['j_steps']

        if rule_type in ('color_based_total', 'area_based'):
            total = num_colors if rule_type == 'color_based_total' else h_in * w_in
            if not total:
                return 1, 1
            j_steps = int(np.sqrt(total))
            while total % j_steps != 0 and j_steps > 1:
                j_steps -= 1
            return total // j_steps, j_steps

        if rule_type == 'color_based_per_dim':
            return (num_colors, num_colors) if num_colors else (1, 1)

        if rule_type == 'dimension_based':
            return h_in, w_in

        return 1, 1

    def _apply_planting_strategy(self, inp: np.ndarray, strategy: List[str], scaling_rule: dict) -> np.ndarray:
        h_in, w_in = inp.shape
        num_colors = len(np.unique(inp))
        i_steps, j_steps = self._calculate_output_dimensions(inp.shape, scaling_rule, num_colors)
        output = np.zeros((h_in * i_steps, w_in * j_steps), dtype=inp.dtype)
        variants = self._generate_pattern_variants(inp)

        idx = 0
        for i in range(i_steps):
            for j in range(j_steps):
                i_out, j_out = i * h_in, j * w_in
                transform = strategy[idx % len(strategy)] if strategy else 'original'
                output[i_out:i_out + h_in, j_out:j_out + w_in] = variants.get(transform, variants['original'])
                idx += 1

        return output


# ============================================================================
# UPSCALE + COVERING, MERGED — one dispatch step for two different solvers
# ============================================================================

class UpscaleOrCoveringSolver:
    """Tries UpscaleSolver, then PatternPlantingSolver ("Covering"), in that
    order — different underlying logic each, merged into a single call so
    SymbolicModule exposes 3 dispatch steps instead of 4. Order is a rough
    "probably cheaper first" guess, not a benchmarked choice; swap it if it
    turns out to matter."""

    def __init__(self, font_color: int = 0):
        self.upscale = UpscaleSolver(font_color=font_color)
        self.covering = PatternPlantingSolver(font_color=font_color)

    def solve(self, task) -> SolveResult:
        upscale_result = self.upscale.solve(task)
        if upscale_result.success:
            return upscale_result
        covering_result = self.covering.solve(task)
        if covering_result.success:
            return covering_result
        return SolveResult.fail(f"upscale: {upscale_result.debug}; covering: {covering_result.debug}")


# ============================================================================
# MIXER (grid-segment interaction: logical ops / color mixing / conjunction)
# ============================================================================

def _np_logical_both_not(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.logical_and(np.logical_not(a), np.logical_not(b))


def _np_logical_fill(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xor = np.logical_xor(a, b)
    return np.logical_and(a, b) if xor.all() else a


LOGIC_FUNCS = {
    "AND": np.logical_and, "OR": np.logical_or, "XOR": np.logical_xor,
    "BNOT": _np_logical_both_not, "FILL": _np_logical_fill,
}

AUGS = {
    "ID": lambda x: x, "LR": np.fliplr, "UD": np.flipud,
    "90": lambda x: np.rot90(x, k=1, axes=(0, 1)),
    "180": lambda x: np.rot90(x, k=2, axes=(0, 1)),
    "270": lambda x: np.rot90(x, k=3, axes=(0, 1)),
}


class MixerSolver:
    """Solves tasks where the grid is split into segments (by a detected
    markup/partition pattern, or a homogeneous-tiling heuristic otherwise)
    that interact via a logical operation, color layering, or a per-segment
    conjunction rule."""

    def __init__(self, font_val: int = 0, pad_val: int = 10):
        self.font_val = font_val
        self.pad_val = pad_val

    def solve(self, task) -> SolveResult:
        try:
            from symbolic.patterns import retrieve_shapes
        except ImportError as e:
            return SolveResult.fail(f"retrieve_shapes unavailable: {e}")

        try:
            test_input = task.test_subtask.train_inp
            solution: List = []
            colors_mapper: Dict[Any, Any] = {}
            transf_type: Optional[str] = None
            skipped: List[str] = []

            for idx, subtask in enumerate(task.subtasks):
                grid = subtask.train_inp
                grid_shape = subtask.train_inp_shape
                patterns = retrieve_shapes(grid, grid_shape, ('markup', 'partition_lines'), self.font_val)

                if idx == 0:
                    transf_type = self._color_analysis(task, patterns)

                segments = self._get_segments(grid, patterns)
                if not segments:
                    skipped.append(f"example {idx}: no segments found")
                    continue

                if transf_type == 'logical_ops':
                    segment_color = self._main_color(segments[0])
                    target_color = self._main_color(subtask.train_out)
                    colors_mapper[segment_color] = target_color

                try:
                    pos_solution = self._solver(segments, transf_type, solution, subtask.train_out)
                except _WrongCheck as e:
                    return SolveResult.fail(
                        f"strategy from earlier example contradicts example {idx}: {e.message}"
                    )
                if pos_solution:
                    solution = copy(pos_solution)

            if not solution:
                reason = "; ".join(skipped) if skipped else "no consistent transformation found across examples"
                return SolveResult.fail(f"mixer: {reason}")

            grid_shape = test_input.shape
            patterns = retrieve_shapes(test_input, grid_shape, ('markup', 'partition_lines'), self.font_val)
            segments = self._get_segments(test_input, patterns)
            if not segments:
                return SolveResult.fail("mixer: could not segment the test input")

            answer = self._infer_grid(segments, transf_type, solution, colors_mapper)
            return SolveResult.ok(answer)

        except Exception as e:
            return SolveResult.fail(f"mixer solver raised {type(e).__name__}: {e}")

    # -- shared helpers -------------------------------------------------------

    def _main_color(self, grid: np.ndarray):
        i, j = np.where((grid != self.pad_val) * (grid != self.font_val))
        return grid[i[0], j[0]]

    def _arr_diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a != b)  # fixed: was an O(n^2) manual coordinate-membership loop

    def _color_analysis(self, task, patterns) -> str:
        flattened_inp, flattened_out = [], []
        for subtask in task.subtasks:
            flattened_inp.extend(subtask.train_inp.flatten().tolist())
            flattened_out.extend(subtask.train_out.flatten().tolist())
        uniq_inp = len(set(flattened_inp))
        uniq_out = len(set(flattened_out))
        if patterns:  # fixed: equivalent to (but clearer than) comparing to an empty defaultdict
            uniq_inp -= 1
        if self.font_val in flattened_inp:
            uniq_inp -= 1
        if self.font_val in flattened_out:
            uniq_out -= 1
        return 'color_mix' if uniq_inp == uniq_out and uniq_out > 2 else 'logical_ops'

    # -- segmentation ---------------------------------------------------------

    def _homog_colored(self, segment: np.ndarray) -> bool:
        return len(set(segment.flatten().tolist())) == 2

    def _get_segments(self, grid: np.ndarray, markups: Dict[str, list]) -> List[np.ndarray]:
        shape = grid.shape

        if markups.get('markup'):
            return self._segments_from_markup(grid, markups['markup'])
        if markups.get('partition_lines'):
            return self._segments_from_partition_lines(grid, markups['partition_lines'])
        return self._segments_from_heuristic(grid, shape)

    def _segments_from_markup(self, grid, markup) -> List[np.ndarray]:
        from symbolic.utils import find_upper_left_corner, coords_transform
        shape = grid.shape
        ul = find_upper_left_corner(shape)
        markup_i_coords, _ = coords_transform(markup[0])
        n_lines = sum(np.array(markup_i_coords) == ul[0])
        n_segments = n_lines + 1
        step_i, step_j = shape[0] // n_segments, shape[1] // n_segments

        segments = []
        i_offset = -1
        for i in range(n_segments):
            i_offset += 1
            j_offset = 0
            for j in range(n_segments):
                segments.append(grid[i * step_i + i_offset:(i + 1) * step_i + i_offset,
                                      j * step_j + j_offset:(j + 1) * step_j + j_offset])
                j_offset += 1
        return segments

    def _segments_from_partition_lines(self, grid, partition_lines) -> List[np.ndarray]:
        # Each divider keeps its own axis (row vs. column), so a markup that
        # genuinely mixes horizontal and vertical dividers is reported as
        # such rather than being split along whichever axis happens to sort
        # last.
        row_coords, col_coords = [], []
        for markup in partition_lines:
            if markup[0][0] == markup[1][0]:
                row_coords.append(markup[0][0])
            else:
                col_coords.append(markup[0][1])

        if row_coords and col_coords:
            raise ValueError(
                "partition markup mixes horizontal and vertical dividers — "
                "not a single-axis split"
            )

        segments = []
        cur = 0
        if row_coords:
            for coord in sorted(row_coords):
                segments.append(grid[cur:coord, :])
                cur = coord + 1
            segments.append(grid[cur:, :])
        else:
            for coord in sorted(col_coords):
                segments.append(grid[:, cur:coord])
                cur = coord + 1
            segments.append(grid[:, cur:])
        return segments

    def _segments_from_heuristic(self, grid: np.ndarray, shape: Tuple[int, int]) -> List[np.ndarray]:
        """No markup detected: try splitting into an NxN grid of equally
        homogeneous-colored square tiles (for square grids), else into equal
        strips along whichever dimension is larger. Stops at the first
        successful tiling for the square case; for strips, keeps the finest
        (largest n_segments) successful split found."""
        if shape[0] == shape[1] and shape[0] >= 4:
            for n_segments in range(2, shape[0] // 2 + 1):
                if shape[0] % n_segments != 0:
                    continue
                step = shape[0] // n_segments
                tiles = []
                ok = True
                for i in range(n_segments):
                    for j in range(n_segments):
                        segment = grid[i * step:(i + 1) * step, j * step:(j + 1) * step]
                        if not self._homog_colored(segment):
                            ok = False
                            break
                        tiles.append(segment)
                    if not ok:
                        break
                if ok:
                    return tiles
            return []

        dim = 0 if shape[0] > shape[1] else 1
        best: List[np.ndarray] = []
        for n_segments in range(2, shape[dim] // 2 + 1):
            if shape[dim] % n_segments != 0:
                continue
            step = shape[dim] // n_segments
            cache = []
            for i in range(n_segments):
                segment = grid[i * step:(i + 1) * step, :] if dim == 0 else grid[:, i * step:(i + 1) * step]
                if not self._homog_colored(segment):
                    cache = []
                    break
                cache.append(segment)
            if len(cache) == n_segments:
                # prefer the finest (largest n_segments) successful split
                if len(cache) > len(best):
                    best = cache
        return best

    # -- transformation search --------------------------------------------

    def _solver(self, segments, transf_type, solution, target):
        if transf_type == "logical_ops":
            return self._logical_transf_search(segments, target, solution)
        if transf_type == "color_mix":
            return self._color_mix_search(segments, target, solution)
        if transf_type == "conjunction":
            return self._conjunction_search(segments, target, solution)
        raise ValueError(f'Unsupported transformation type: {transf_type!r}')

    def _logical_transf_search(self, segments, target, solution):
        target_color = self._main_color(target)
        masks = [g != self.font_val for g in segments]

        if solution:
            res_mask = masks[0]
            aug_name, func_name = solution
            aug, func = AUGS[aug_name], LOGIC_FUNCS[func_name]
            for m in masks[1:]:
                res_mask = func(res_mask, aug(m))
            if np.equal(res_mask * target_color, target).all():
                return (aug_name, func_name)
            raise _WrongCheck(f"previously found ({aug_name}, {func_name}) no longer matches")

        for func_name, func in LOGIC_FUNCS.items():
            for aug_name, aug in AUGS.items():
                if aug(masks[0]).shape != target.shape:
                    continue
                res_mask = masks[0]
                for m in masks[1:]:
                    res_mask = func(res_mask, aug(m))
                if np.equal(res_mask * target_color, target).all():
                    return (aug_name, func_name)
        return False

    def _color_mix_search(self, segments, target, solution):
        masks = [g != self.font_val for g in segments]
        shape = segments[0].shape

        def build(perm, aug_name):
            aug = AUGS[aug_name]
            answer = np.zeros(shape)
            res_mask_prev = masks[perm[0]]
            answer += res_mask_prev * segments[perm[0]]
            for idx in perm[1:]:
                res_mask = np.logical_or(res_mask_prev, aug(masks[idx]))
                answer += self._arr_diff(res_mask, res_mask_prev) * aug(segments[idx])
                res_mask_prev = copy(res_mask)
            return answer

        if solution:
            aug_name, perm = solution
            if AUGS[aug_name](masks[perm[0]]).shape == target.shape:
                answer = build(perm, aug_name)
                if np.equal(answer, target).all():
                    return (aug_name, perm)
            raise _WrongCheck(f"previously found ({aug_name}, {perm}) no longer matches")

        for perm in permutations(range(len(segments))):
            perm = list(perm)
            for aug_name in AUGS:
                if AUGS[aug_name](masks[perm[0]]).shape != target.shape:
                    continue
                if np.equal(build(perm, aug_name), target).all():
                    return (aug_name, perm)
        return False

    def _conjunction_search(self, segments, target, solution):
        masks = [g != self.font_val for g in segments]
        n = len(masks)

        def build(aug_name, segments_colors):
            aug = AUGS[aug_name]
            aug_masks = [aug(m) for m in masks]
            answer = aug(segments[0])
            for i in range(n):
                segment = aug(segments[i])
                j = (i + 1) % n
                unique_mask = aug_masks[i] != aug_masks[j]
                for coord in zip(*np.where(unique_mask)):
                    if segment[coord] != self.font_val:
                        answer[coord] = segments_colors[i]
            return answer

        if solution:
            aug_name, segments_colors = solution
            answer = build(aug_name, segments_colors)
            if np.equal(answer, target).all():
                return (aug_name, segments_colors)
            raise _WrongCheck(f"previously found ({aug_name}, {segments_colors}) no longer matches")

        aug_name = "ID"
        if AUGS[aug_name](segments[0]).shape != target.shape:
            return False

        aug_masks = [AUGS[aug_name](m) for m in masks]
        segments_colors = []
        unique_coords_list = []
        for i in range(n):
            j = (i + 1) % n
            unique_mask = aug_masks[i] != aug_masks[j]
            unique_coords = list(zip(*np.where(unique_mask)))
            found_colors = {
                AUGS[aug_name](target)[coord] for coord in unique_coords
                if AUGS[aug_name](target)[coord] != self.font_val
                and AUGS[aug_name](segments[i])[coord] != self.font_val
            }
            if len(found_colors) != 1:
                return False  # no single consistent color for this segment
            segments_colors.append(found_colors.pop())
            unique_coords_list.append(unique_coords)

        answer = AUGS[aug_name](segments[0])
        for i in range(n):
            segment = AUGS[aug_name](segments[i])
            for coord in unique_coords_list[i]:
                if segment[coord] != self.font_val:
                    answer[coord] = segments_colors[i]
        if np.equal(answer, target).all():
            return (aug_name, segments_colors)
        return False

    # -- applying the found transformation to the test input ----------------

    def _infer_grid(self, segments, transf_type, solution, colors_mapper) -> np.ndarray:
        if transf_type == "logical_ops":
            masks = [g != self.font_val for g in segments]
            res_mask = masks[0]
            segment_color = self._main_color(segments[0])
            aug_name, func_name = solution
            aug, func = AUGS[aug_name], LOGIC_FUNCS[func_name]
            for m in masks[1:]:
                res_mask = func(res_mask, aug(m))
            target_color = colors_mapper[segment_color]
            return (res_mask * target_color).astype(int)

        if transf_type == "color_mix":
            masks = [g != self.font_val for g in segments]
            shape = segments[0].shape
            aug_name, perm = solution
            aug = AUGS[aug_name]
            answer = np.zeros(shape)
            res_mask_prev = masks[perm[0]]
            answer += res_mask_prev * segments[perm[0]]
            for idx in perm[1:]:
                res_mask = np.logical_or(res_mask_prev, aug(masks[idx]))
                answer += self._arr_diff(res_mask, res_mask_prev) * aug(segments[idx])
                res_mask_prev = copy(res_mask)
            return answer.astype(int)

        if transf_type == "conjunction":
            aug_name, segments_colors = solution
            aug = AUGS[aug_name]
            aug_masks = [aug(g != self.font_val) for g in segments]
            answer = segments[0].copy()
            n = len(segments)
            for i in range(n):
                segment = segments[i]
                j = (i + 1) % n
                unique_mask = aug_masks[i] != aug_masks[j]
                for coord in zip(*np.where(unique_mask)):
                    if segment[coord] != self.font_val:
                        answer[coord] = segments_colors[i]
            return answer.astype(int)

        raise ValueError(f'Unsupported transformation type: {transf_type!r}')


class _WrongCheck(Exception):
    """Internal-only: a previously found solution no longer matches the
    current example. Caught in MixerSolver.solve() and turned into a
    SolveResult.fail(...) with the message preserved."""
    def __init__(self, message="Contradiction in answer searching"):
        self.message = message
        super().__init__(message)


# ============================================================================
# COLOR RESTORE (symmetry-based patch filling)
# ============================================================================

class ColorRestoreSolver:
    """Fills font-colored (missing) cells using whatever left-right/up-down
    symmetry the grid exhibits — either globally, or within a detected
    symmetric sub-region when the whole grid isn't symmetric."""

    def __init__(self, font_val: int = 0, pad_val: int = 10, max_iterations: int = 10):
        self.font_val = font_val
        self.pad_val = pad_val
        self.max_iterations = max_iterations

    def solve(self, task) -> SolveResult:
        try:
            train_subtask = deepcopy(task.subtasks[-1])
            test_subtask = deepcopy(task.test_subtask)
            train_inp, train_out = train_subtask.train_inp, train_subtask.train_out
            test_inp = test_subtask.train_inp

            shape_correspondence = train_inp.shape == train_out.shape
            if not shape_correspondence:
                try:
                    from symbolic.patterns import find_connected_components_with_color
                except ImportError as e:
                    return SolveResult.fail(f"find_connected_components_with_color unavailable: {e}")
                patch = find_connected_components_with_color(train_inp, self.font_val)
                if not patch:
                    return SolveResult.fail("no font-colored patch found in the training input")
                i_1, i_2, j_1, j_2 = self._segment2slice(patch[0])
                train_inp[i_1:i_2, j_1:j_2] = train_out
                train_out = train_inp

            restored_grid = copy(test_inp)
            symmetry = self._check_symmetry(train_out)

            if symmetry:
                restored_grid, ok = self._restore_with_symmetry(test_inp, symmetry)
                if not ok:
                    return SolveResult.fail(
                        f"detected '{symmetry}' symmetry but couldn't fill every font cell"
                    )
            else:
                symmetry_shape = self._find_symmetry_shape(train_out)
                if not symmetry_shape:
                    return SolveResult.fail("no symmetric region found in the training output")
                i_1, i_2, j_1, j_2 = symmetry_shape
                symmetry_type = self._check_symmetry(train_out[i_1:i_2, j_1:j_2])
                if symmetry_type == "lr&ud":
                    restored_grid = self._restore_with_edges(restored_grid, symmetry_shape)
                restored_patch, ok = self._restore_with_symmetry(test_inp[i_1:i_2, j_1:j_2], symmetry_type)
                if not ok:
                    return SolveResult.fail(
                        f"detected '{symmetry_type}' symmetry in a sub-region but couldn't fill it"
                    )
                restored_grid[i_1:i_2, j_1:j_2] = restored_patch

            if not shape_correspondence:
                test_patch = find_connected_components_with_color(test_inp, self.font_val)
                if not test_patch:
                    return SolveResult.fail("no font-colored patch found in the test input")
                i_1, i_2, j_1, j_2 = self._segment2slice(test_patch[0])
                restored_grid = restored_grid[i_1:i_2, j_1:j_2]

            return SolveResult.ok(restored_grid)

        except Exception as e:
            return SolveResult.fail(f"color restore solver raised {type(e).__name__}: {e}")

    # -- symmetry detection -------------------------------------------------

    def _check_symmetry(self, grid: np.ndarray):
        shape = grid.shape
        if shape[0] % 2 != 0 or shape[1] % 2 != 0:
            return False
        mid_i, mid_j = shape[0] // 2, shape[1] // 2
        lr = np.equal(np.fliplr(grid[:, :mid_j]), grid[:, mid_j:]).all()
        ud = np.equal(np.flipud(grid[:mid_i, :]), grid[mid_i:, :]).all()
        if lr and ud:
            return "lr&ud"
        if lr:
            return "lr"
        if ud:
            return "ud"
        return False

    def _find_symmetry_shape(self, grid: np.ndarray, max_slice: int = 4):
        max_i, max_j = grid.shape
        increments = list(range(max_slice))
        neg_increments = [-i for i in increments]

        candidates = (
            [(i, j, max_i, max_j) for i, j in list(product(increments, increments))[1:]]
            + [(i, 0, max_i, max_j + j) for i, j in list(product(increments, neg_increments))[1:]]
            + [(0, j, max_i - i, max_j) for i, j in list(product(neg_increments, increments))[1:]]
            + [(0, 0, max_i - i, max_j - j) for i, j in list(product(neg_increments, neg_increments))[1:]]
        )
        for i_1, j_1, i_2, j_2 in candidates:
            if self._check_symmetry(grid[i_1:i_2, j_1:j_2]) == "lr&ud":
                return (i_1, i_2, j_1, j_2)
        return False

    def _segment2slice(self, coords: List[tuple]) -> Tuple[int, int, int, int]:
        i_coords = [c[0] for c in coords]
        j_coords = [c[1] for c in coords]
        return min(i_coords), max(i_coords) + 1, min(j_coords), max(j_coords) + 1

    # -- restoration ----------------------------------------------------------

    def _restore_with_symmetry(self, grid: np.ndarray, symmetry_type: str) -> Tuple[np.ndarray, bool]:
        """Iteratively fill font cells from whichever fully-colored
        half/quarter is available, re-expanding outward (restore_with_slices)
        between rounds when no new fully-colored section has appeared.

        Rewritten from a recursive version whose recursive calls had the
        wrong number of arguments, referenced an undefined `symmetric_shape`
        variable, and never captured/returned the recursive result — any
        grid that needed more than one restoration pass crashed instead of
        actually restoring."""
        grid = copy(grid)
        prev_full_sects = -1

        for _ in range(self.max_iterations):
            if self.font_val not in grid:
                return grid, True

            shape = grid.shape
            mid_i, mid_j, max_i, max_j = shape[0] // 2, shape[1] // 2, shape[0], shape[1]
            halves = {0: (0, max_i, 0, mid_j), 1: (0, mid_i, 0, max_j),
                      2: (0, max_i, mid_j, max_j), 3: (mid_i, max_i, 0, max_j)}
            # Quarter 2 (bottom-right) starts its column range at mid_j, not
            # mid_i — the two coincide on square grids but not on
            # rectangular ones.
            quarters = {0: (0, mid_i, 0, mid_j), 1: (0, mid_i, mid_j, max_j),
                        2: (mid_i, max_i, mid_j, max_j), 3: (mid_i, max_i, 0, mid_j)}

            h_idxs = [k for k, c in halves.items() if self.font_val not in grid[c[0]:c[1], c[2]:c[3]]]
            q_idxs = [k for k, c in quarters.items() if self.font_val not in grid[c[0]:c[1], c[2]:c[3]]]
            n_full_sects = len(h_idxs) + len(q_idxs)

            if n_full_sects == prev_full_sects:
                expanded = self._restore_with_slices(grid, symmetry_type)
                if (expanded == grid).all():
                    return grid, False  # stuck: no further progress possible
                grid = expanded
                prev_full_sects = -1
                continue
            prev_full_sects = n_full_sects

            grid = self._fill_from_symmetry(grid, symmetry_type, halves, h_idxs, quarters, q_idxs)

        return grid, self.font_val not in grid

    def _fill_from_symmetry(self, grid, symmetry_type, halves, h_idxs, quarters, q_idxs) -> np.ndarray:
        """Fill whichever sections are derivable from a known quarter/half
        under the detected symmetry, using the actual mirror relationship
        (fliplr / flipud / both) between sections — not rotation, which is a
        different (stronger) property than lr/ud mirror symmetry and, on a
        rectangular grid, swaps the two axis lengths so the fill wouldn't
        even fit its target slot."""
        grid = copy(grid)
        if q_idxs:
            self._fill_from_quarter(grid, quarters, q_idxs[0], symmetry_type)
        if h_idxs:
            self._fill_from_half(grid, halves, h_idxs[0], symmetry_type)
        return grid

    def _fill_from_quarter(self, grid, quarters, src_idx, symmetry_type) -> None:
        can_lr = symmetry_type in ("lr", "lr&ud")
        can_ud = symmetry_type in ("ud", "lr&ud")
        src_coords = quarters[src_idx]
        source = grid[src_coords[0]:src_coords[1], src_coords[2]:src_coords[3]].copy()
        src_top, src_left = src_idx in (0, 1), src_idx in (0, 3)

        for idx, coords in quarters.items():
            if idx == src_idx:
                continue
            is_top, is_left = idx in (0, 1), idx in (0, 3)
            same_row, same_col = is_top == src_top, is_left == src_left
            if same_row and not same_col and can_lr:
                grid[coords[0]:coords[1], coords[2]:coords[3]] = np.fliplr(source)
            elif same_col and not same_row and can_ud:
                grid[coords[0]:coords[1], coords[2]:coords[3]] = np.flipud(source)
            elif not same_row and not same_col and can_lr and can_ud:
                grid[coords[0]:coords[1], coords[2]:coords[3]] = np.flipud(np.fliplr(source))

    def _fill_from_half(self, grid, halves, src_idx, symmetry_type) -> None:
        can_lr = symmetry_type in ("lr", "lr&ud")
        can_ud = symmetry_type in ("ud", "lr&ud")
        src_coords = halves[src_idx]
        source = grid[src_coords[0]:src_coords[1], src_coords[2]:src_coords[3]].copy()

        if can_lr and src_idx in (0, 2):
            target = 2 if src_idx == 0 else 0
            c = halves[target]
            grid[c[0]:c[1], c[2]:c[3]] = np.fliplr(source)
        if can_ud and src_idx in (1, 3):
            target = 3 if src_idx == 1 else 1
            c = halves[target]
            grid[c[0]:c[1], c[2]:c[3]] = np.flipud(source)

    def _restore_with_slices(self, grid: np.ndarray, symmetry_type: str) -> np.ndarray:
        shape = grid.shape
        mid_i, mid_j, max_i, max_j = shape[0] // 2, shape[1] // 2, shape[0], shape[1]
        restored_grid = copy(grid)

        for _ in range(5):
            for j in range(mid_j):
                left = restored_grid[:, :mid_j - j] if mid_j - j > 0 else None
                right = restored_grid[:, mid_j + j:]
                left_ok = left is not None and self.font_val not in left
                right_ok = self.font_val not in right

                if left_ok and not right_ok:
                    if symmetry_type == "lr&ud":
                        restored_grid[:, mid_j + j:] = np.fliplr(left)
                        restored_grid[:mid_j - j, :] = np.rot90(left, k=1, axes=(1, 0))
                        restored_grid[mid_j + j:, :] = np.rot90(left, k=1, axes=(0, 1))
                    elif symmetry_type == "lr":
                        restored_grid[:, mid_j + j:] = np.fliplr(left)
                elif right_ok and not left_ok:
                    if symmetry_type == "lr&ud":
                        restored_grid[:, :mid_j - j] = np.fliplr(right)
                        restored_grid[:mid_j - j, :] = np.rot90(right, k=1, axes=(0, 1))
                        restored_grid[mid_j + j:, :] = np.rot90(right, k=1, axes=(1, 0))
                    elif symmetry_type == "lr":
                        restored_grid[:, :mid_j - j] = np.fliplr(right)

            for i in range(mid_i):
                top = restored_grid[:mid_i - i, :] if mid_i - i > 0 else None
                bottom = restored_grid[mid_i + i:, :]
                top_ok = top is not None and self.font_val not in top
                bottom_ok = self.font_val not in bottom

                if top_ok and not bottom_ok:
                    if symmetry_type == "lr&ud":
                        restored_grid[mid_i + i:, :] = np.flipud(top)
                        restored_grid[:, mid_i + i:] = np.rot90(top, k=1, axes=(1, 0))
                        restored_grid[:, :mid_i - i] = np.rot90(top, k=1, axes=(0, 1))
                    elif symmetry_type == "ud":
                        restored_grid[mid_i + i:, :] = np.flipud(top)
                elif bottom_ok and not top_ok:
                    if symmetry_type == "lr&ud":
                        restored_grid[:mid_i - i, :] = np.flipud(bottom)
                        restored_grid[:, mid_i + i:] = np.rot90(bottom, k=1, axes=(0, 1))
                        restored_grid[:, :mid_i - i] = np.rot90(bottom, k=1, axes=(1, 0))
                    elif symmetry_type == "ud":
                        restored_grid[:mid_i - i, :] = np.flipud(bottom)

        return restored_grid

    def _restore_with_edges(self, restored_grid: np.ndarray, symmetry_shape: tuple) -> np.ndarray:
        max_i, max_j = restored_grid.shape
        patch_min_i, patch_max_i, patch_min_j, patch_max_j = symmetry_shape
        offset = (patch_min_i, max_i - patch_max_i, patch_min_j, max_j - patch_max_j)

        if offset[0] > 0 and offset[2] > 0:
            top = restored_grid[0:patch_min_i, :]
            left = restored_grid[:, 0:patch_min_j]
            if self.font_val not in top:
                restored_grid[:, 0:patch_min_j] = np.fliplr(np.rot90(top, k=1, axes=(1, 0)))
            elif self.font_val not in left:
                restored_grid[0:patch_min_i, :] = np.fliplr(np.rot90(left, k=1, axes=(1, 0)))
        elif offset[0] > 0 and offset[3] > 0:
            top = restored_grid[0:patch_min_i, :]
            right = restored_grid[:, patch_max_j:]
            if self.font_val not in top:
                restored_grid[:, 0:patch_min_j] = np.rot90(top, k=1, axes=(1, 0))
            elif self.font_val not in right:
                restored_grid[0:patch_min_i, :] = np.rot90(right, k=1, axes=(0, 1))
        elif offset[1] > 0 and offset[2] > 0:
            bottom = restored_grid[patch_max_i:, :]
            left = restored_grid[:, 0:patch_min_j]
            if self.font_val not in bottom:
                restored_grid[:, 0:patch_min_j] = np.rot90(bottom, k=1, axes=(1, 0))
            elif self.font_val not in left:
                restored_grid[0:patch_min_i, :] = np.rot90(left, k=1, axes=(0, 1))
        elif offset[1] > 0 and offset[3] > 0:
            bottom = restored_grid[patch_max_i:, :]
            right = restored_grid[:, patch_max_j:]
            if self.font_val not in bottom:
                restored_grid[:, 0:patch_min_j] = np.fliplr(np.rot90(bottom, k=1, axes=(1, 0)))
            elif self.font_val not in right:
                restored_grid[0:patch_min_i, :] = np.fliplr(np.rot90(right, k=1, axes=(1, 0)))

        return restored_grid


# ============================================================================
# AGGREGATOR — plain attribute access, no shared logic or dispatch
# ============================================================================

class SymbolicModule:
    """Groups the three solvers for convenience only:
        SymbolicModule().mixer.solve(task)
        SymbolicModule().upscale_or_covering.solve(task)
        SymbolicModule().color_restore.solve(task)
    No shared state, no orchestration/dispatch logic between them.
    """

    def __init__(self, font_val: int = 0, pad_val: int = 10):
        self.mixer = MixerSolver(font_val=font_val, pad_val=pad_val)
        self.upscale_or_covering = UpscaleOrCoveringSolver(font_color=font_val)
        self.color_restore = ColorRestoreSolver(font_val=font_val, pad_val=pad_val)
