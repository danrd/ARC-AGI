"""Tests for subsymbolic/analyst.py - the ranked shortlist of agents that
fills the "agent relevance scores" coordinator_instruction/v1.j2 asks for.

The parsing tests carry most of the weight. What comes back is model output,
so every way it can be wrong has to end somewhere legible: an empty
shortlist with a reason, never a guessed agent. An empty shortlist is a
usable answer - the coordinator's instruction reads relevance as "if given" -
while a wrong one sends it after the wrong agent while looking like a result.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from subsymbolic.analyst import (
    ANALYST_BLOCKS,
    DEFAULT_SHORTLIST_SIZE,
    AgentShortlist,
    parse_shortlist,
    rank_agents,
    render_agent_roster,
)

ROSTER = [
    {"index": 1, "name": "Constructor", "purpose": "Create new objects.",
     "modules": ["Symbolic", "Subsymbolic"]},
    {"index": 2, "name": "Highlighter", "purpose": "Extract key elements.",
     "modules": ["Subsymbolic"]},
    {"index": 3, "name": "Shifter", "purpose": "Relocate objects.", "modules": ["Interactive"]},
]
DETAILS = {"Constructor": "Role: colour background cells\nFocus: placement"}


def _module(response, prompt="analyst prompt"):
    """SubsymbolicModule's shape, as much of it as rank_agents touches."""
    seen = {}

    def build(task, context=None):
        seen["context"] = context
        return prompt

    return SimpleNamespace(
        builder=SimpleNamespace(build=build, config=SimpleNamespace(blocks=ANALYST_BLOCKS)),
        runner=SimpleNamespace(generate=lambda p: response),
        seen=seen,
    )


# -- the roster the analyst reads --------------------------------------------

class TestRoster:
    @staticmethod
    def test_both_descriptions_reach_the_analyst():
        """The one-line purpose distinguishes agents poorly on its own -
        "establish correspondences between objects" against "transform object
        coloration" - so the longer role text goes in beside it."""
        text = render_agent_roster(ROSTER, DETAILS)

        assert "Create new objects." in text
        assert "Role: colour background cells" in text

    @staticmethod
    def test_every_agent_appears_under_its_own_name():
        text = render_agent_roster(ROSTER, DETAILS)

        for agent in ROSTER:
            assert f"### {agent['name']}" in text

    @staticmethod
    def test_an_agent_without_a_role_text_still_appears():
        """The two descriptions live in different places and can disagree
        about who exists; the roster is the registry's, not the role texts'."""
        text = render_agent_roster(ROSTER, role_instructions={})

        assert "### Shifter" in text
        assert "Relocate objects." in text


# -- reading the model's answer ----------------------------------------------

class TestParsing:
    @staticmethod
    def test_a_clean_answer_keeps_the_models_order():
        """Order is the entire content of the answer - it decides which agent
        runs first and which is cut."""
        result = parse_shortlist(
            '{"shortlist": ["Shifter", "Constructor"], "reasoning": "objects move"}', ROSTER)

        assert result.agents == ("Shifter", "Constructor")
        assert result.reasoning == "objects move"
        assert result.error is None

    @staticmethod
    def test_json_wrapped_in_prose_is_still_read():
        """Instructed not to, models still explain themselves first."""
        text = 'Sure! Here you go:\n```json\n{"shortlist": ["Constructor"]}\n```\nHope that helps.'

        assert parse_shortlist(text, ROSTER).agents == ("Constructor",)

    @staticmethod
    def test_casing_is_not_a_different_agent():
        assert parse_shortlist('{"shortlist": ["constructor"]}', ROSTER).agents == ("Constructor",)

    @staticmethod
    def test_an_invented_agent_is_dropped_and_recorded():
        """Not corrected to the nearest real name: a near-miss is not
        evidence about which agent was meant. Recorded, because a model that
        keeps inventing names is a prompt problem and dropping them quietly
        is how that stays invisible."""
        result = parse_shortlist('{"shortlist": ["Rotator", "Shifter"]}', ROSTER)

        assert result.agents == ("Shifter",)
        assert result.unknown == ("Rotator",)

    @staticmethod
    def test_a_repeated_agent_is_one_vote():
        result = parse_shortlist('{"shortlist": ["Shifter", "Shifter", "Constructor"]}', ROSTER)

        assert result.agents == ("Shifter", "Constructor")

    @staticmethod
    def test_the_shortlist_is_cut_to_size():
        result = parse_shortlist(
            '{"shortlist": ["Shifter", "Constructor", "Highlighter"]}', ROSTER, shortlist_size=2)

        assert result.agents == ("Shifter", "Constructor")

    @staticmethod
    @pytest.mark.parametrize("text,reason", [
        ("", "nothing"),
        ("   ", "nothing"),
        ("I could not decide.", "no JSON"),
        ('{"shortlist": ["Shifter"', "truncated"),
        ('{"shortlist": [oops]}', "not valid JSON"),
        ('["Shifter"]', "not an object"),
        ('{"agents": ["Shifter"]}', "no 'shortlist' list"),
        ('{"shortlist": ["Rotator", "Painter"]}', "no agent on the roster"),
    ])
    def test_every_bad_answer_ends_up_empty_with_a_reason(text, reason):
        result = parse_shortlist(text, ROSTER)

        assert result.is_empty
        assert result.error is not None and reason in result.error

    @staticmethod
    def test_a_single_name_instead_of_a_list_is_accepted():
        assert parse_shortlist('{"shortlist": "Shifter"}', ROSTER).agents == ("Shifter",)


# -- what the coordinator ends up seeing --------------------------------------

class TestAuxiliaryInfo:
    @staticmethod
    def test_the_order_survives_into_the_rendered_block():
        info = AgentShortlist(agents=("Shifter", "Constructor"), reasoning="why").as_auxiliary_info()
        (block,) = info.values()

        assert block.index("1. Shifter") < block.index("2. Constructor")
        assert "why" in block

    @staticmethod
    def test_nothing_established_renders_no_section_at_all():
        """Rather than an empty heading: the coordinator's instruction reads
        relevance as "if given", and a present-but-empty section is a claim
        that nothing was found worth saying."""
        assert AgentShortlist(error="the model returned nothing").as_auxiliary_info() == {}


# -- end to end against a stand-in model --------------------------------------

class TestRankAgents:
    @staticmethod
    def test_the_roster_reaches_the_prompt_context():
        module = _module('{"shortlist": ["Constructor"]}')

        rank_agents(task=object(), module=module, registry=ROSTER, role_instructions=DETAILS)

        context = module.seen["context"]
        assert "### Constructor" in context["agents_info_text"]
        assert context["shortlist_size"] == DEFAULT_SHORTLIST_SIZE

    @staticmethod
    def test_a_prompt_that_does_not_fit_is_reported_not_guessed():
        module = _module('{"shortlist": ["Constructor"]}', prompt=None)

        result = rank_agents(task=object(), module=module, registry=ROSTER,
                              role_instructions=DETAILS)

        assert result.is_empty
        assert "token_limit" in result.error

    @staticmethod
    def test_the_configured_blocks_render_against_the_real_builder(arc_task):
        """Every test above stands a fake in for PromptBuilder, which means
        none of them opens a .j2 file. This one builds the analyst's prompt
        the way SubsymbolicModule would - a missing template, an undeclared
        filter or a context key no one supplies fails here and nowhere else.
        """
        from subsymbolic.prompt_builder import ApproxTokenizer, PromptBuilder, PromptingConfig
        from subsymbolic.registry import FILTER_REGISTRY, RESOLVER_REGISTRY
        from data.configs.agents_config import AGENTS_REGISTRY

        config = PromptingConfig(blocks=ANALYST_BLOCKS, resolvers=["examples", "summary"],
                                  filters=["grid"], token_limit=8000)
        builder = PromptBuilder(config, ApproxTokenizer(),
                                 resolver_registry=RESOLVER_REGISTRY,
                                 filter_registry=FILTER_REGISTRY)
        echoed = {}

        def echo(prompt):
            echoed["prompt"] = prompt
            return prompt

        module = SimpleNamespace(builder=builder, runner=SimpleNamespace(generate=echo))

        result = rank_agents(task=arc_task, module=module,
                              context={"grid_repr_type": "concise"})

        built = echoed["prompt"]
        for agent in AGENTS_REGISTRY:
            assert agent["name"] in built, f"{agent['name']} missing from the roster block"
        assert str(DEFAULT_SHORTLIST_SIZE) in built
        # The runner echoed the prompt back, which is not JSON - so reaching a
        # parse failure is the proof the prompt was built and sent at all.
        assert "token_limit" not in (result.error or "")

    @staticmethod
    def test_caller_context_reaches_the_templates():
        """A block the caller added needs its own context, and adding one is
        how the analyst gets new evidence - MCTS rollouts, once the RL branch
        runs - rather than growing an argument here."""
        module = _module('{"shortlist": ["Shifter"]}')

        rank_agents(task=object(), module=module, registry=ROSTER, role_instructions=DETAILS,
                     context={"promising_actions": "rotate90, shift_object"})

        assert module.seen["context"]["promising_actions"] == "rotate90, shift_object"
