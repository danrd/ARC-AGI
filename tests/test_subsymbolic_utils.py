"""Tests for subsymbolic/utils.py: parse_llm_output's actual accepted
format (documented here because it's narrower than its own docstring
suggests) and build_grid_grammar, the GBNF grammar meant to keep a
grammar-constrained generation inside that same accepted format.
"""
from __future__ import annotations

import numpy as np

from subsymbolic.utils import build_grid_grammar, parse_llm_output


def test_parse_llm_output_accepts_concatenated_digit_rows():
    """The only row format that actually parses for n_cols > 1: one
    concatenated digit string per row, no spaces between cells."""
    result = parse_llm_output("2,3:\n1 012\n2 345")

    assert np.array_equal(result, np.array([[0, 1, 2], [3, 4, 5]]))


def test_parse_llm_output_rejects_space_separated_multi_column_rows():
    """Regression/documentation test: line.split() on a space-separated row
    ("1 0 1 2") produces more than 2 parts for any n_cols > 1, which hits
    `if len(parts) > 2: return ""` - despite the docstring's "1 x_1 ... x_m"
    notation reading like space-separated values are supported, they never
    parse. build_grid_grammar's row rule deliberately only allows the
    concatenated form for exactly this reason."""
    result = parse_llm_output("2,3:\n1 0 1 2\n2 3 4 5")

    assert result == ""


def test_build_grid_grammar_compiles_as_valid_gbnf():
    """A real llama.cpp GBNF syntax check, not just "it's a string" - a
    malformed grammar would otherwise only surface at actual inference
    time."""
    from llama_cpp import LlamaGrammar

    LlamaGrammar.from_string(build_grid_grammar())


def test_build_grid_grammar_colors_str_variant_compiles_too():
    from llama_cpp import LlamaGrammar

    LlamaGrammar.from_string(build_grid_grammar(colors_str=True))


def test_build_grid_grammar_default_cells_are_digits():
    grammar = build_grid_grammar()

    assert "[0-9]" in grammar
    assert "bBRGYgMOSW" not in grammar


def test_build_grid_grammar_colors_str_cells_are_letters():
    grammar = build_grid_grammar(colors_str=True)

    assert "[bBRGYgMOSW]" in grammar
