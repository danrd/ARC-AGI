"""Tests for scripts/module_map.py.

The point of generating the architecture map is that it cannot quietly
stop being true, so what is pinned here is the part that would make it
lie: edges read from the imports, cycles named with the import that
causes them, and --check noticing when the document has drifted from the
code. A generator that silently produces a stale-but-plausible diagram is
worse than no diagram.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "module_map", REPO_ROOT / "scripts" / "module_map.py")
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


@pytest.fixture
def tree(tmp_path):
    """Two packages, one importing the other both ways round."""
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / "alpha" / "one.py").write_text(
        "from beta.two import thing\nimport beta.three\n")
    (tmp_path / "alpha" / "two.py").write_text("from alpha.one import x\n")
    (tmp_path / "beta" / "two.py").write_text("import os\n")
    (tmp_path / "beta" / "three.py").write_text("from alpha.one import x\n")
    return tmp_path


class TestScanning:
    def test_every_module_is_found(self, tree):
        modules, _, packages = script.scan(tree)

        assert set(modules) == {"alpha.one", "alpha.two", "beta.two", "beta.three"}
        assert packages == ["alpha", "beta"]

    def test_imports_inside_the_tree_become_edges(self, tree):
        _, edges, _ = script.scan(tree)

        assert ("alpha.one", "beta.two") in edges
        assert ("alpha.one", "beta.three") in edges

    def test_imports_of_the_outside_world_do_not(self, tree):
        _, edges, _ = script.scan(tree)

        assert all(not target.startswith("os") for _, target in edges)

    def test_a_file_that_does_not_parse_is_skipped_not_fatal(self, tree):
        """A tree with one broken file still has a map; refusing to draw
        anything because of it helps nobody."""
        (tree / "alpha" / "broken.py").write_text("def (:\n")

        modules, edges, _ = script.scan(tree)

        assert "alpha.broken" in modules
        assert ("alpha.one", "beta.two") in edges


class TestPackageEdges:
    def test_edges_within_one_package_are_not_between_packages(self, tree):
        _, edges, _ = script.scan(tree)

        counted = script.package_edges(edges)

        assert ("alpha", "alpha") not in counted
        assert counted[("alpha", "beta")] == 2

    def test_a_cycle_names_the_import_that_causes_it(self, tree):
        _, edges, _ = script.scan(tree)

        cycles = script.back_edges(edges)

        assert len(cycles) == 1
        (source, target), few, many, causes = cycles[0]
        assert (source, target) == ("beta", "alpha")
        assert (few, many) == (1, 2)
        assert causes == ["beta.three imports alpha.one"]

    def test_packages_that_only_point_one_way_are_not_reported(self, tree):
        (tree / "beta" / "three.py").write_text("x = 1\n")
        _, edges, _ = script.scan(tree)

        assert script.back_edges(edges) == []


class TestSplicing:
    def test_the_hand_written_half_survives(self):
        document = f"prose\n\n{script.START}\nold\n{script.END}\ntail\n"

        out = script.splice(document, f"{script.START}\nnew\n{script.END}")

        assert out == f"prose\n\n{script.START}\nnew\n{script.END}\ntail\n"

    def test_a_document_without_markers_gets_the_section_appended(self):
        out = script.splice("prose\n", f"{script.START}\nnew\n{script.END}")

        assert out.startswith("prose\n")
        assert script.START in out


class TestCheck:
    def test_the_repository_map_is_current(self):
        """The check CI would run: ARCHITECTURE.md still describes this
        code. It fails on the commit that adds a module and forgets to
        regenerate."""
        generated = script.render(REPO_ROOT)
        document = (REPO_ROOT / "ARCHITECTURE.md").read_text()

        assert script.splice(document, generated) == document
