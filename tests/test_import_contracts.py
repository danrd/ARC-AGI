"""The layering in ARCHITECTURE.md, checked rather than asserted.

`lint-imports` reads the contracts from pyproject.toml, so this is the same
check CI would run; it lives in the suite as well because a layering
violation is cheapest to see at the moment it is written, and because a
contract nobody runs is a contract nobody keeps.

The known debts are exemptions listed in pyproject.toml with the reason
attached. Deleting a line there must mean the import is gone, never that the
run needed to pass.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_the_import_contracts_hold():
    """import-linter is a dev dependency, so an environment without it
    skips rather than fails - the check belongs to CI, not to whether the
    tests can run at all."""
    pytest.importorskip("importlinter",
                        reason="pip install import-linter to run this")
    from importlinter import configuration
    from importlinter.application import use_cases

    configuration.configure()
    previous = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        kept = use_cases.lint_imports(
            config_filename=str(REPO_ROOT / "pyproject.toml"), no_logo=True)
    finally:
        os.chdir(previous)

    assert kept, ("an import crosses a layer ARCHITECTURE.md says it cannot - "
                  "run `lint-imports` for the offending import")
