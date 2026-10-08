"""The measurement of the invariants: the wrong answers it makes must be wrong in the way it says."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_invariants as measuring  # noqa: E402


def test_a_nudged_answer_differs_in_at_most_three_cells_and_keeps_to_its_own_palette():
    rng = np.random.default_rng(0)
    answer = np.array([[1, 2, 1], [2, 1, 2]])
    for _ in range(50):
        out = measuring.nudged(answer, rng)
        assert (out != answer).sum() <= 3 and set(np.unique(out)) <= {1, 2}


def test_a_one_colour_answer_cannot_be_nudged_inside_its_palette():
    out = measuring.nudged(np.zeros((2, 2), int), np.random.default_rng(0))
    assert (out == 0).all()


def test_the_true_answer_is_refused_rarely_and_a_task_of_another_kind_almost_always():
    counts = measuring.measure("training", seed=0)
    n, refused, _ = counts["true"]
    assert refused / n < 0.03
    n, refused, _ = counts["other"]
    assert refused / n > 0.9
