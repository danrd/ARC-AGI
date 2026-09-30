"""Symbolic solvers for ARC tasks, wrapped as classes with one uniform
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
from dataclasses import dataclass
from itertools import permutations, product
from math import factorial
from typing import Any, Dict, List, Optional, Tuple


#: What a colour-mix search will spend before it gives up.
#:
#: That search tries every ordering of the segments against every
#: augmentation, so its cost is `len(segments)! * len(AUGS)`. Measured at
#: 12-15 thousand candidates a second: six segments take 0.3 s, eight take
#: 20 s, nine about three minutes, and the two training tasks that hand it
#: sixteen segments would need some thirty thousand years. That is the
#: "hang" every run of this module has worked around with an alarm - it was
#: never a deadlock, just a factorial.
#:
#: The bound costs no answer that was ever produced. A census of the
#: training split: 38 tasks reach this search, 32 of them with five
#: segments or fewer, and the six above that (8, 9, 9, 16, 16 and 36
#: segments) are all refused by the solver anyway - including the one at
#: eight, which finishes its search in 7 s and finds nothing.
#:
#: Counted rather than timed on purpose. A wall-clock cutoff makes the
#: answer depend on how loaded the machine was, and this pipeline
#: reproduces run to run - the same seed gives the same number in a
#: different process on a different day, which is a property worth keeping.
MAX_MIX_CANDIDATES = 50_000


def affordable_orderings(segment_count: int) -> bool:
    """Whether a colour-mix search over this many segments is one this
    module will spend.

    A function rather than an expression inlined at the one call site,
    because it is the decision itself and a test that re-derives
    `factorial(n) * len(AUGS)` on its own side proves only that two copies
    of the same arithmetic agree - it passes just as happily when the
    module counts something else entirely.
    """
    return factorial(segment_count) * len(AUGS) <= MAX_MIX_CANDIDATES


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


def _holding_out(task, index: int):
    """`task` with training pair `index` moved into the test slot.

    Built from `type(task)` rather than by importing ARCTask: rl imports
    symbolic and not the other way round, and this needs no more of the
    class than the constructor every task here already has.
    """
    kept = [pair for position, pair in enumerate(task.subtasks)
            if position != index]
    held = task.subtasks[index]
    return type(task)(label=f"{task.label}-holding-out-{index}",
                      subtasks=kept,
                      test_inp=held.train_inp, test_out=held.train_out)


def checked_solve(solver, task) -> SolveResult:
    """`solver.solve(task)`, with the claim held to the solver's own examples.

    A solver that thinks it succeeded has nothing to explain, which is the
    one case where these modules are silent and the one case where they are
    usually wrong: upscale_or_covering claims 350 of 400 training tasks and
    is right on 13, color_restore claims 31 and is right on 1. Overall the
    three solvers make 392 claims and 25 are correct - a claim on its own
    carries almost no information.

    A rule inferred from the examples can be applied back to them. Hold one
    training pair out, give the solver the rest, and ask it for the held-out
    pair, whose answer is known. A rule that cannot reproduce an example it
    was shown is not a rule. Nothing here reads the test answer, so the
    check is available at inference.

    Measured over both splits, keeping a claim only when every held-out
    example came back exactly right:

        training    392 claims, 25 correct ->  21 kept, 21 correct
                    precision   6.4% -> 100.0%
        evaluation  375 claims, 28 correct ->  28 kept, 26 correct
                    precision   7.5% ->  92.9%

    The strict reading is what ships. Forgiving a declined example - a
    solver refusing when it has one example fewer is arguably saying it
    needs that example rather than that its rule is false - keeps one more
    claim on training and it is a wrong one (100% -> 95.5%), and changes
    nothing at all on evaluation. Same answers retained either way, so the
    stricter rule is free.

    Not folded into `solve` itself, and not a third kind of result. It
    cannot live in `solve` because it calls `solve`, and a solver whose own
    examples contradict it is simply wrong about the task - so it comes
    back as `fail` with the reason in `debug`, like every other way these
    solvers decline.

    Costs one solve per training pair on top of the first: three or four
    for most ARC tasks.
    """
    result = solver.solve(task)
    if not result.success:
        return result

    pairs = list(getattr(task, "subtasks", ()))
    if len(pairs) < 2:
        # Defensive: holding one pair out of one leaves nothing to infer
        # from. No ARC task is like this - the fewest either split has is
        # two - but an unverifiable claim is not a verified one.
        return SolveResult.fail(
            f"{len(pairs)} training example(s) is too few to check the claim "
            f"against")

    for index, held in enumerate(pairs):
        replay = solver.solve(_holding_out(task, index))
        if not replay.success:
            return SolveResult.fail(
                f"claimed an answer, then declined example {index} when it "
                f"was held out: {replay.debug}")
        if not np.array_equal(np.asarray(replay.grid),
                              np.asarray(held.train_out)):
            return SolveResult.fail(
                f"claimed an answer, then got example {index} wrong when it "
                f"was held out - the rule does not reproduce an example it "
                f"was shown")
    return result


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

    #: How many times the search may be asked for another answer for the first
    #: example when the one it gave contradicts a later one.
    MAX_RETRIES = 50

    def solve(self, task) -> SolveResult:
        try:
            from symbolic.patterns import retrieve_shapes
        except ImportError as e:
            return SolveResult.fail(f"retrieve_shapes unavailable: {e}")

        try:
            first = task.subtasks[0]
            first_patterns = retrieve_shapes(first.train_inp, first.train_inp_shape,
                                             ('markup', 'partition_lines'), self.font_val)
            primary = self._color_analysis(task, first_patterns)
            # The kind the colours point to first; "fit" - the pieces are laid
            # over one another when they do not collide, and the first piece
            # stands alone when they do - is tried where that finds nothing.
            reasons = []
            for transf_type in dict.fromkeys([primary, 'fit']):
                outcome = self._fit_examples(task, transf_type, retrieve_shapes)
                if outcome["solution"]:
                    return self._answer(task, transf_type, outcome, retrieve_shapes)
                reasons.append(outcome["reason"])
            return SolveResult.fail(reasons[0])

        except Exception as e:
            return SolveResult.fail(f"mixer solver raised {type(e).__name__}: {e}")

    def _fit_examples(self, task, transf_type, retrieve_shapes) -> Dict[str, Any]:
        """Search the examples for one way of combining the pieces of type
        `transf_type`.

        The search settles on the first way that gives the first example, and
        checks it against the others. Where several ways give the first
        example - pieces that never overlap in it, so any order of layering
        does - the first of them was kept and a later example that said
        otherwise ended the search. Now the way that failed is set aside and
        the search asked again."""
        rejected: List = []
        last_wrong = ""
        for _ in range(self.MAX_RETRIES + 1):
            solution: List = []
            first_found: Optional[List] = None
            colors_mapper: Dict[Any, Any] = {}
            skipped: List[str] = []
            #: How many equal strips the examples were cut into where nothing
            #: marked a split, read off the output's size - see
            #: _segments_from_heuristic.
            strip_counts = set()
            wrong: Optional[str] = None

            for idx, subtask in enumerate(task.subtasks):
                grid = subtask.train_inp
                patterns = retrieve_shapes(grid, subtask.train_inp_shape,
                                           ('markup', 'partition_lines'), self.font_val)
                segments = self._get_segments(grid, patterns, out_shape=subtask.train_out.shape)
                if not segments:
                    skipped.append(f"example {idx}: no segments found")
                    continue
                if not patterns:
                    strip_counts.add(len(segments))

                if transf_type == 'logical_ops':
                    segment_color = self._main_color(segments[0])
                    target_color = self._main_color(subtask.train_out)
                    colors_mapper[segment_color] = target_color

                try:
                    pos_solution = self._solver(segments, transf_type, solution, subtask.train_out, rejected)
                except _WrongCheck as e:
                    wrong = f"strategy from earlier example contradicts example {idx}: {e.message}"
                    break
                except _TooManyOrderings as e:
                    # Not "raised", which is what the catch-all below would
                    # call it: this is the solver declining a task it
                    # cannot afford, and it reads as a refusal.
                    return {"solution": None, "reason": f"mixer declines example {idx}: {e.message}"}
                if pos_solution:
                    if not solution:
                        first_found = copy(pos_solution)
                    solution = copy(pos_solution)

            if wrong is not None:
                last_wrong = wrong
                if first_found is None:
                    break
                rejected.append(self._key(first_found))
                continue
            if not solution:
                reason = "; ".join(skipped) if skipped else "no consistent transformation found across examples"
                return {"solution": None, "reason": f"mixer: {reason}"}
            return {"solution": solution, "colors_mapper": colors_mapper,
                    "strip_counts": strip_counts, "reason": ""}
        return {"solution": None, "reason": last_wrong}

    def _answer(self, task, transf_type, outcome, retrieve_shapes) -> SolveResult:
        test_input = task.test_subtask.train_inp
        patterns = retrieve_shapes(test_input, test_input.shape, ('markup', 'partition_lines'), self.font_val)
        counts = outcome["strip_counts"]
        segments = self._get_segments(test_input, patterns, count=next(iter(counts)) if len(counts) == 1 else None)
        if not segments:
            return SolveResult.fail("mixer: could not segment the test input")
        answer = self._infer_grid(segments, transf_type, outcome["solution"], outcome["colors_mapper"])
        return SolveResult.ok(answer)

    @staticmethod
    def _key(solution):
        """A found way of combining as something a set can hold."""
        return tuple(tuple(part) if isinstance(part, list) else part for part in solution)

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
        if uniq_inp == uniq_out and uniq_out > 2:
            return 'color_mix'
        # Colours the inputs do not have, more of them than the pieces have
        # between them: the cells one piece has and its neighbour lacks are
        # marked in a colour of their own - the conjunction rule, which
        # _color_analysis was never routing to (the search and the way of
        # applying it were both here, and nothing chose them).
        if uniq_out > uniq_inp:
            return 'conjunction'
        return 'logical_ops'

    # -- segmentation ---------------------------------------------------------

    def _homog_colored(self, segment: np.ndarray) -> bool:
        return len(set(segment.flatten().tolist())) == 2

    def _get_segments(self, grid: np.ndarray, markups: Dict[str, list],
                      out_shape=None, count=None) -> List[np.ndarray]:
        shape = grid.shape

        if markups.get('markup'):
            return self._segments_from_markup(grid, markups['markup'])
        if markups.get('partition_lines'):
            return self._segments_from_partition_lines(grid, markups['partition_lines'])
        return self._segments_from_heuristic(grid, shape, out_shape=out_shape, count=count)

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

    def _segments_from_heuristic(self, grid: np.ndarray, shape: Tuple[int, int],
                                 out_shape=None, count=None) -> List[np.ndarray]:
        """No markup detected: try splitting into an NxN grid of equally
        homogeneous-colored square tiles (for square grids), else into equal
        strips along whichever dimension is larger. Stops at the first
        successful tiling for the square case; for strips, keeps the finest
        (largest n_segments) successful split found.

        The finest split is a guess, and a wrong one where the answer is a
        few large pieces: a 12x4 input that is two 6x4 pieces one over the
        other is also six 2x4 strips of two colours each, and the finer one
        won. Where the output is known and is one piece of the input, the
        strips are that piece's size; where it is not (the test input), the
        number of strips the examples were cut into."""
        if shape[0] != shape[1] and (out_shape is not None or count is not None):
            dim = 0 if shape[0] > shape[1] else 1
            n_segments = count
            if n_segments is None and out_shape[1 - dim] == shape[1 - dim] \
                    and 0 < out_shape[dim] < shape[dim] and shape[dim] % out_shape[dim] == 0:
                n_segments = shape[dim] // out_shape[dim]
            if n_segments and n_segments > 1 and shape[dim] % n_segments == 0:
                step = shape[dim] // n_segments
                strips = [grid[i * step:(i + 1) * step, :] if dim == 0 else grid[:, i * step:(i + 1) * step]
                          for i in range(n_segments)]
                if all(self._homog_colored(strip) for strip in strips):
                    return strips
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

    def _solver(self, segments, transf_type, solution, target, rejected=()):
        if transf_type == "logical_ops":
            return self._logical_transf_search(segments, target, solution, rejected)
        if transf_type == "color_mix":
            return self._color_mix_search(segments, target, solution, rejected)
        if transf_type == "conjunction":
            return self._conjunction_search(segments, target, solution)
        if transf_type == "fit":
            return self._fit_search(segments, target, solution)
        raise ValueError(f'Unsupported transformation type: {transf_type!r}')

    def _logical_transf_search(self, segments, target, solution, rejected=()):
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
                if aug(masks[0]).shape != target.shape or (aug_name, func_name) in rejected:
                    continue
                res_mask = masks[0]
                for m in masks[1:]:
                    res_mask = func(res_mask, aug(m))
                if np.equal(res_mask * target_color, target).all():
                    return (aug_name, func_name)
        return False

    def _color_mix_search(self, segments, target, solution, rejected=()):
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

        # Before the loop rather than inside it: the point is to decline a
        # search that cannot finish, and a check that only fires after the
        # first million orderings has already spent the time it was meant
        # to save. Checked after the `solution` branch above, which costs
        # one build and is worth doing at any segment count.
        if not affordable_orderings(len(segments)):
            raise _TooManyOrderings(
                f"{len(segments)} segments would take "
                f"{factorial(len(segments)) * len(AUGS):,} orderings to try, "
                f"past the {MAX_MIX_CANDIDATES:,} this search spends")

        for perm in permutations(range(len(segments))):
            perm = list(perm)
            for aug_name in AUGS:
                if AUGS[aug_name](masks[perm[0]]).shape != target.shape \
                        or (aug_name, tuple(perm)) in rejected:
                    continue
                if np.equal(build(perm, aug_name), target).all():
                    return (aug_name, perm)
        return False

    def _fit_build(self, segments):
        """The pieces laid over one another, the first on top, when no two
        have a cell in the same place; the first piece alone when two do."""
        masks = [g != self.font_val for g in segments]
        if any((masks[a] & masks[b]).any() for a in range(len(masks)) for b in range(a + 1, len(masks))):
            return segments[0].copy()
        answer = segments[0].copy()
        for piece, mask in zip(segments[1:], masks[1:]):
            answer = np.where(mask, piece, answer)
        return answer

    def _fit_search(self, segments, target, solution):
        if len({segment.shape for segment in segments}) != 1 or segments[0].shape != target.shape:
            if solution:
                raise _WrongCheck("previously found fit rule does not apply to this example's shape")
            return False
        matches = np.array_equal(self._fit_build(segments), target)
        if solution:
            if matches:
                return solution
            raise _WrongCheck("previously found fit rule no longer matches")
        return ("ID", "fit") if matches else False

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

        if transf_type == "fit":
            return self._fit_build(segments).astype(int)

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


class _TooManyOrderings(Exception):
    """Internal-only: the colour-mix search was handed more segments than
    MAX_MIX_CANDIDATES allows orderings for. Caught in MixerSolver.solve()
    and turned into a SolveResult.fail(...) - a solver that declines is
    wrong about the task, not broken, and the message says which it is."""
    def __init__(self, message):
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

            inferred = self._restore_by_inferred_symmetries(test_inp)
            if inferred is not None:
                if shape_correspondence:
                    return SolveResult.ok(inferred)
                test_patch = find_connected_components_with_color(test_inp, self.font_val)
                if test_patch:
                    i_1, i_2, j_1, j_2 = self._segment2slice(test_patch[0])
                    return SolveResult.ok(inferred[i_1:i_2, j_1:j_2])

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

            # A restoration fills what is missing and touches nothing else.
            # The region-based restoration above reflects whole strips, and
            # where the picture is not symmetric all the way out it rewrites
            # cells that were right: measured, 20 of them on one evaluation
            # task, and that answer passed the held-out check.
            # (Where the answer is only the patch, what is done outside it
            # does not reach the answer.)
            known = test_inp != self.font_val
            if shape_correspondence and (restored_grid[known] != test_inp[known]).any():
                return SolveResult.fail("the restoration changed cells that were not missing")

            if not shape_correspondence:
                test_patch = find_connected_components_with_color(test_inp, self.font_val)
                if not test_patch:
                    return SolveResult.fail("no font-colored patch found in the test input")
                i_1, i_2, j_1, j_2 = self._segment2slice(test_patch[0])
                restored_grid = restored_grid[i_1:i_2, j_1:j_2]

            return SolveResult.ok(restored_grid)

        except Exception as e:
            return SolveResult.fail(f"color restore solver raised {type(e).__name__}: {e}")

    # -- symmetries read off the grid itself --------------------------------

    #: How much of the visible picture a mirror has to overlap to count, and
    #: how much of that overlap has to agree.
    MIN_OVERLAP = 0.3

    def _mirrors(self, height: int, width: int):
        """(name, function of index arrays) for every mirror the grid could
        have: left-right and up-down about any line, and the two diagonals
        through any offset - the symmetries a restoration task is built on."""
        found = []
        for total in range(width // 2, 2 * width - width // 2):
            found.append(("lr", total, lambda i, j, t=total: (i, t - j)))
        for total in range(height // 2, 2 * height - height // 2):
            found.append(("ud", total, lambda i, j, t=total: (t - i, j)))
        for offset in range(-(width // 2), height // 2 + 1):
            found.append(("diag", offset, lambda i, j, d=offset: (j + d, i - d)))
        for total in range(min(height, width) // 2, height + width - min(height, width) // 2):
            found.append(("anti", total, lambda i, j, t=total: (t - j, t - i)))
        return found

    def _agreement(self, grid, visible, mirror):
        """(cells compared, share that agree) for a mirror, over the cells
        both it and its image leave visible."""
        height, width = grid.shape
        rows, cols = np.indices((height, width))
        image_rows, image_cols = mirror(rows, cols)
        inside = (image_rows >= 0) & (image_rows < height) & (image_cols >= 0) & (image_cols < width)
        image_rows = np.clip(image_rows, 0, height - 1)
        image_cols = np.clip(image_cols, 0, width - 1)
        both = inside & visible & visible[image_rows, image_cols]
        compared = int(both.sum())
        if not compared:
            return 0, 0.0
        return compared, float((grid[both] == grid[image_rows, image_cols][both]).mean())

    def _restore_by_inferred_symmetries(self, grid: np.ndarray):
        """The grid with every font cell filled from a mirror image of it, or
        None when the grid has none to fill or its mirrors do not reach them.

        The mirrors are found in the grid in hand, from the cells that are
        not font: any left-right, up-down or diagonal mirror under which
        those cells agree wherever a cell and its image are both there.
        Nothing is carried over from the training pictures, so a symmetry
        axis that sits somewhere else in each example - the case a fixed
        region of the last training output cannot express - is found again
        each time. A cell is filled from the image the best-supported mirror
        gives it, and the result has to keep every mirror it was filled by
        exact; a picture whose mirrors contradict each other is refused."""
        grid = np.asarray(grid)
        visible = (grid != self.font_val) & (grid != self.pad_val)
        if visible.all() or not visible.any():
            return None
        mirrors, mirrors_named = [], []
        for name, key, function in self._mirrors(*grid.shape):
            compared, share = self._agreement(grid, visible, function)
            if compared >= self.MIN_OVERLAP * visible.sum() and share == 1.0:
                mirrors.append((compared, function))
                mirrors_named.append((name, key))
        if not mirrors:
            return None
        mirrors.sort(key=lambda item: -item[0])
        filled = grid.copy()
        missing = grid == self.font_val
        for _ in range(self.max_iterations):
            if not missing.any():
                break
            progressed = False
            rows, cols = np.where(missing)
            for _support, function in mirrors:
                image_rows, image_cols = function(rows, cols)
                inside = (image_rows >= 0) & (image_rows < grid.shape[0]) \
                    & (image_cols >= 0) & (image_cols < grid.shape[1])
                for k in np.where(inside)[0]:
                    r, c = rows[k], cols[k]
                    if missing[r, c] and not missing[image_rows[k], image_cols[k]]:
                        filled[r, c] = filled[image_rows[k], image_cols[k]]
                        missing[r, c] = False
                        progressed = True
            if not progressed:
                break
        if missing.any():
            filled, missing = self._restore_by_local_mirrors(filled, missing, self._quarter_turns(grid, mirrors_named))
            if missing.any():
                return None
        everything = np.ones(grid.shape, dtype=bool)
        for _support, function in mirrors:
            _n, share = self._agreement(filled, everything, function)
            if share != 1.0:
                return None
        return filled

    #: The pairs along a cell's row, or along its column, a mirror has to
    #: agree on, none against, for it to be trusted at a cell the
    #: picture-wide mirrors do not reach.
    MIN_LOCAL_PAIRS = 8

    def _quarter_turns(self, grid, mirrors_named):
        """Quarter turns about the centre the picture-wide left-right and
        up-down mirrors share: the frame round a symmetric picture is often
        the picture's own edge turned, top strip to side strip."""
        totals_j = [key for name, key in mirrors_named if name == "lr"]
        totals_i = [key for name, key in mirrors_named if name == "ud"]
        turns = []
        for a in totals_j[:1]:
            for b in totals_i[:1]:
                if (a - b) % 2:
                    continue
                shift, both = (b - a) // 2, (a + b) // 2
                turns.append(lambda i, j, u=shift, v=both: (j + u, v - i))
                turns.append(lambda i, j, u=shift, v=both: (v - j, i - u))
        return turns

    def _restore_by_local_mirrors(self, grid, missing, extra=()):
        """Fill what the picture-wide mirrors could not reach - a frame
        around the symmetric part, whose top strip is the transpose of its
        left one - from a mirror that is exact along the cell's own row or
        along its own column. Returns the grid and what is still missing."""
        filled, missing = grid.copy(), missing.copy()
        height, width = grid.shape
        visible = ~missing & (grid != self.pad_val)
        rows_all, cols_all = np.indices((height, width))
        table = []
        for function in [f for _n, _k, f in self._mirrors(height, width)] + list(extra):
            image_rows, image_cols = function(rows_all, cols_all)
            inside = (image_rows >= 0) & (image_rows < height) & (image_cols >= 0) & (image_cols < width)
            clipped_rows, clipped_cols = np.clip(image_rows, 0, height - 1), np.clip(image_cols, 0, width - 1)
            both = inside & visible & visible[clipped_rows, clipped_cols]
            agree = both & (grid == grid[clipped_rows, clipped_cols])
            against = both & ~agree
            table.append((function, agree.sum(axis=1), agree.sum(axis=0),
                          against.sum(axis=1), against.sum(axis=0)))
        for _ in range(self.max_iterations):
            progressed = False
            for r, c in zip(*np.where(missing)):
                best, best_support = None, 0
                for function, agree_rows, agree_cols, against_rows, against_cols in table:
                    support = max(int(agree_rows[r]) if not against_rows[r] else 0,
                                  int(agree_cols[c]) if not against_cols[c] else 0)
                    if support < self.MIN_LOCAL_PAIRS or support <= best_support:
                        continue
                    image_row, image_col = function(r, c)
                    if 0 <= image_row < height and 0 <= image_col < width and not missing[image_row, image_col]:
                        best, best_support = (image_row, image_col), support
                if best is not None:
                    filled[r, c] = filled[best]
                    missing[r, c] = False
                    progressed = True
            if not progressed or not missing.any():
                break
        return filled, missing

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
        mid_i, mid_j = shape[0] // 2, shape[1] // 2
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

def font_value_candidates(task, limit: int = 3) -> List[int]:
    """The colours a task's "font" - its missing or background cells - may be,
    most likely first.

    Every solver here is built around one such colour: ColorRestoreSolver
    fills the cells of it, the mixer treats it as empty, the upscale solvers
    as the ground. It was left at 0, which is right where the picture sits on
    black and wrong wherever the missing part is painted some other colour -
    a symmetric picture with a patch of 8 laid over it, say. Measured on the
    evaluation tasks the symbolic modules were meant to solve: a fixed 0
    solved 14 of 30, and eight of the other sixteen were solved by
    ColorRestoreSolver once it was told the patch was 8, 6, 3, 2, 0, 7, 3.

    Read off the training pairs, which is available at inference:

      same-size pairs   the colour of the input cells that change, by how
                        many change (the patch a restoration removes)
      other pairs       a colour every input has and no output has (the
                        patch a crop cuts out)

    then 0, then the most common colour of the inputs - and no more than
    `limit`, since each candidate is a further round of every solver.
    """
    pairs = [(np.asarray(s.train_inp), np.asarray(s.train_out)) for s in task.subtasks]
    ranked: List[int] = []
    if pairs and all(inp.shape == out.shape for inp, out in pairs):
        changed: Dict[int, int] = {}
        for inp, out in pairs:
            for colour in inp[inp != out].tolist():
                changed[colour] = changed.get(colour, 0) + 1
        ranked += [colour for colour, _ in sorted(changed.items(), key=lambda item: -item[1])][:2]
    elif pairs:
        in_every_input = set.intersection(*(set(np.unique(inp).tolist()) for inp, _ in pairs))
        in_an_output = set().union(*(set(np.unique(out).tolist()) for _, out in pairs))
        ranked += sorted(in_every_input - in_an_output)[:2]
    ranked.append(0)
    if pairs:
        counts = np.bincount(np.concatenate([inp.ravel() for inp, _ in pairs]))
        ranked.append(int(counts.argmax()))
    unique = list(dict.fromkeys(int(colour) for colour in ranked))
    return unique[:limit]


class SymbolicModule:
    """Groups the three solvers for convenience only:
        SymbolicModule().mixer.solve(task)
        SymbolicModule().upscale_or_covering.solve(task)
        SymbolicModule().color_restore.solve(task)
    No shared state, no orchestration/dispatch logic between them.

    Built with one font colour. For a task, `for_task` builds one module per
    colour the task's training pairs suggest - see font_value_candidates.
    """

    def __init__(self, font_val: int = 0, pad_val: int = 10):
        self.font_val = font_val
        self.mixer = MixerSolver(font_val=font_val, pad_val=pad_val)
        self.upscale_or_covering = UpscaleOrCoveringSolver(font_color=font_val)
        self.color_restore = ColorRestoreSolver(font_val=font_val, pad_val=pad_val)

    @classmethod
    def for_task(cls, task, pad_val: int = 10, limit: int = 3) -> List["SymbolicModule"]:
        return [cls(font_val=colour, pad_val=pad_val)
                for colour in font_value_candidates(task, limit=limit)]
