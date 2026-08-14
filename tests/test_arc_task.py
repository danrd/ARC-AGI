"""Tests for rl/arc_task.py."""
from __future__ import annotations

import numpy as np

from rl.arc_task import ARCTask


def test_id_aliases_label():
    """subsymbolic.llm_run.run_llm_over_tasks reads task.id as its
    generic, JSON-serializable checkpoint key - ARCTask's own identity is
    `label`, so `id` has to mirror it rather than introduce a second,
    possibly-diverging name."""
    task = ARCTask(label="007bbfb7", subtasks=[],
                    test_inp=np.zeros((2, 2), dtype=int), test_out=np.zeros((2, 2), dtype=int))

    assert task.id == "007bbfb7" == task.label
