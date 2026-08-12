"""ARC-specific evaluator + result plotter for subsymbolic.llm_run: scores
one LLM generation against a task's known test output, and (separately)
renders that same comparison as a figure for wandb's per-task result plot.
"""
from __future__ import annotations

import numpy as np

from subsymbolic.arc_grid_formatting import format_grid
from subsymbolic.llm_run import EvalResult
from subsymbolic.utils import lev_sim, parse_llm_output


def arc_grid_evaluator(task, generated_text: str, repr_type: str = "concise",
                        colors_str: bool = False, expected_prefix: str = "") -> EvalResult:
    """Exact grid match plus Levenshtein similarity between the raw
    generated text and the ground truth rendered the same way it's shown in
    the prompt. primary_score is whichever signal is higher; "solved" means
    either one hit a perfect match.

    `expected_prefix`: if the prompt pre-seeds part of the answer (e.g. a
    known target-shape header via PromptingConfig.assistant_prefix, so the
    model only needs to generate the grid body), pass that same prefix here
    so parsing sees the same text the model was actually completing.
    """
    target_grid = task.test_subtask.train_out
    target_text = format_grid(target_grid, repr_type=repr_type)

    parsed = parse_llm_output(expected_prefix + generated_text, colors_str=colors_str)
    exact_match = 0.0
    if isinstance(parsed, np.ndarray) and parsed.shape == target_grid.shape:
        exact_match = float(np.array_equal(parsed, target_grid))

    similarity = lev_sim(generated_text, target_text)
    primary_score = max(exact_match, similarity)

    return EvalResult(
        metrics={"exact_match": exact_match, "lev_sim": similarity},
        primary_score=primary_score,
        solved=(primary_score == 1.0),
    )


def arc_result_plotter(task, generated_text: str, eval_result: EvalResult,
                        colors_str: bool = False, expected_prefix: str = ""):
    """The wandb-loggable counterpart to arc_grid_evaluator: parses
    generated_text the same way (same colors_str/expected_prefix, so the
    picture matches what was actually scored) and renders it as a
    side-by-side grid comparison instead of a score. Wired in as
    run_llm_over_tasks' result_plotter - only called when
    WandbLogConfig.log_result_plot is on.
    """
    from utils.plotting import plot_grid, plot_grids_comparison  # lazy: matplotlib shouldn't be required just to score

    test_input = np.array(task.test_subtask.train_inp)
    target_grid = np.array(task.test_subtask.train_out)
    parsed = parse_llm_output(expected_prefix + generated_text, colors_str=colors_str)

    if isinstance(parsed, np.ndarray) and parsed.ndim == 2:
        fig = plot_grids_comparison(test_input, parsed, target_grid=target_grid)
    else:
        # generated_text didn't parse into a grid at all - nothing to
        # compare cell-by-cell, so just show what the correct answer
        # should have been.
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plot_grid(target_grid)
        plt.title("Prediction did not parse into a grid - showing target only")

    fig.suptitle(f"score={eval_result.primary_score:.2f} solved={eval_result.solved}")
    return fig
