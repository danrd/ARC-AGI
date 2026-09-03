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


def test_index_defaults_to_none():
    """Unset unless a caller (typically ARCDataset, after building its
    full task list) explicitly stamps one on - it only means something
    relative to a specific dataset ordering, not to the task itself."""
    task = ARCTask(label="007bbfb7", subtasks=[],
                    test_inp=np.zeros((2, 2), dtype=int), test_out=np.zeros((2, 2), dtype=int))

    assert task.index is None


def test_index_can_be_set_at_construction():
    task = ARCTask(label="007bbfb7", subtasks=[],
                    test_inp=np.zeros((2, 2), dtype=int), test_out=np.zeros((2, 2), dtype=int),
                    index=37)

    assert task.index == 37
