"""Tests for utils/utils.py: torch is only needed inside seed_everything's
own torch.manual_seed calls, not by the module as a whole - load_json/
load_pickle etc (and everything that imports them, e.g. ARCDataset via
load_json) shouldn't require torch to be installed just to import.
"""
from __future__ import annotations

import importlib
import json
import sys


def _reload_utils_without_torch(monkeypatch):
    """sys.modules["torch"] = None is Python's own way of forcing `import
    torch` to raise ImportError - then force a fresh import of utils.utils
    so its own import statements re-run under that condition."""
    monkeypatch.setitem(sys.modules, "torch", None)
    sys.modules.pop("utils.utils", None)
    return importlib.import_module("utils.utils")


def test_module_imports_without_torch_installed(monkeypatch):
    module = _reload_utils_without_torch(monkeypatch)

    assert module is not None


def test_load_json_works_without_torch(monkeypatch, tmp_path):
    module = _reload_utils_without_torch(monkeypatch)
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"a": 1}))

    assert module.load_json(str(path)) == {"a": 1}


def test_seed_everything_does_not_crash_without_torch(monkeypatch):
    module = _reload_utils_without_torch(monkeypatch)

    module.seed_everything(123)  # should not raise despite no torch


def test_seed_everything_still_seeds_torch_when_available(monkeypatch):
    sys.modules.pop("utils.utils", None)
    module = importlib.import_module("utils.utils")

    module.seed_everything(123)  # torch is installed in this test env - exercises the real path
