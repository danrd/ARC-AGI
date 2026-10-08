"""The registry's agents as the graph runs them, and a run without the interactive module."""
from data.configs.agents_config import AGENTS_REGISTRY
from orchestration.assemble import assemble
from orchestration.configs import AgentRunConfig, OrchestrationOptions, SystemRunConfig
from orchestration.graph import AgentInvConfig, ModuleInvConfig, solve_task
from orchestration.trace import Tracer
from orchestration.roster import agent_factory, order_by_analyst, without_module
from subsymbolic.analyst import AgentShortlist

from tests.test_orchestration_assemble import GOOD, WRONG_SHAPE, judge, make_task, module


class TestTheRegistry:
    def test_a_module_is_taken_from_every_agent_and_an_agent_left_with_nothing_goes(self):
        registry = without_module(AGENTS_REGISTRY, "Interactive")
        assert all("Interactive" not in a["modules"] for a in registry)
        assert all(m["name"] != "Interactive" for a in registry for m in a["available_modules"])
        dropped = {a["name"] for a in AGENTS_REGISTRY} - {a["name"] for a in registry}
        assert "Connector" in dropped and "Constructor" not in dropped
        constructor = next(a for a in registry if a["name"] == "Constructor")
        assert constructor["modules"] == ["Symbolic", "Subsymbolic"]
        assert any("Interactive" in a["modules"] for a in AGENTS_REGISTRY)           # the original is untouched

    def test_the_case_of_the_name_does_not_matter(self):
        assert without_module(AGENTS_REGISTRY, "interactive") == without_module(AGENTS_REGISTRY, "Interactive")

    def test_an_entry_becomes_a_strict_agent_starting_with_its_symbolic_module(self):
        make = agent_factory(AGENTS_REGISTRY)
        constructor = make({"index": 4, "name": "Constructor"})
        assert constructor.strict and constructor.initial_module.module_name == "Symbolic"
        assert [m["name"] for m in constructor.available_modules] == ["Symbolic", "Subsymbolic", "Interactive"]
        mapper = make({"index": 3, "name": "Mapper"})
        assert mapper.initial_module.module_name == "Subsymbolic"

    def test_the_analysts_agents_come_first_and_an_empty_shortlist_changes_nothing(self):
        ordered = order_by_analyst(AGENTS_REGISTRY, AgentShortlist(agents=("Shifter", "Mapper")))
        assert [a["name"] for a in ordered[:2]] == ["Shifter", "Mapper"] and len(ordered) == len(AGENTS_REGISTRY)
        assert [a["name"] for a in order_by_analyst(AGENTS_REGISTRY, AgentShortlist())] == \
            [a["name"] for a in AGENTS_REGISTRY]


class TestAStrictAgent:
    def run(self, modules, tiny_tokenizer, replies, rl):
        started = []

        def rl_start(task):
            started.append(1)
            return None

        run = assemble(OrchestrationOptions(), module(tiny_tokenizer, replies), judge, rl_start_fn=rl_start,
                       tracer=Tracer())
        agent = AgentInvConfig(0, "a", ModuleInvConfig(0, "symbolic"),
                               [{"index": i, "name": n} for i, n in enumerate(modules)], strict=True)
        context = {"grid_repr_type": "concise", "test_input_grid": make_task().test_subtask.train_inp}
        result = solve_task(make_task(), initial_agent=agent, available_agents=[{"index": 0, "name": "a"}],
                            auxiliary_info=context, **run.kwargs())
        return result, started, run

    def test_without_the_interactive_module_no_rl_job_is_started(self, tiny_tokenizer):
        _, started, run = self.run(["symbolic", "subsymbolic"], tiny_tokenizer, [GOOD], rl=False)
        assert started == [] and "symbolic" in run.tracer.summary()
        _, started, _ = self.run(["symbolic", "subsymbolic", "interactive"], tiny_tokenizer, [GOOD], rl=True)
        assert started == [1]

    def test_without_a_symbolic_module_the_models_answer_is_checked_and_not_taken_as_a_solvers(self, tiny_tokenizer):
        result, _, run = self.run(["subsymbolic"], tiny_tokenizer, [WRONG_SHAPE], rl=False)
        assert result["accepted_source"] != "llm" and result["accepted_source"] != "symbolic"
        assert run.module.runner.prompts          # the model was asked, with no symbolic gate before it
        assert "symbolic" not in run.tracer.summary()

    def test_without_a_subsymbolic_module_the_model_is_never_asked(self, tiny_tokenizer):
        m = module(tiny_tokenizer, [GOOD])
        run = assemble(OrchestrationOptions(), m, judge)
        agent = AgentInvConfig(0, "a", ModuleInvConfig(0, "symbolic"), [{"index": 0, "name": "symbolic"}], strict=True)
        solve_task(make_task(), initial_agent=agent, available_agents=[{"index": 0, "name": "a"}], **run.kwargs())
        assert m.runner.prompts == []

    def test_an_agent_with_nothing_to_run_ends_at_once(self, tiny_tokenizer):
        m = module(tiny_tokenizer, [GOOD])
        run = assemble(OrchestrationOptions(), m, judge)
        agent = AgentInvConfig(0, "a", ModuleInvConfig(0, "none"), [], strict=True)
        result = solve_task(make_task(), initial_agent=agent, available_agents=[{"index": 0, "name": "a"}], **run.kwargs())
        assert m.runner.prompts == [] and result.get("accepted_source") is None

    def test_a_loose_agent_still_falls_back_to_its_initial_module(self, tiny_tokenizer):
        m = module(tiny_tokenizer, [GOOD])
        run = assemble(OrchestrationOptions(), m, judge, rl_start_fn=lambda task: None)
        agent = AgentInvConfig(0, "a", ModuleInvConfig(0, "symbolic"), [{"index": 0, "name": "symbolic"}])
        solve_task(make_task(), initial_agent=agent, available_agents=[{"index": 0, "name": "a"}], **run.kwargs())
        assert m.runner.prompts == []        # the fallback is the old behaviour: the one module it has runs in both steps


class TestTheRoleOfTheAgent:
    AGENTS = [{"index": 8, "name": "Modifier"}, {"index": 6, "name": "Highlighter"}]

    def run(self, tiny_tokenizer, agents=None):
        from data.configs.agents_config import AGENTS_REGISTRY, ROLE_INSTRUCTIONS
        m = module(tiny_tokenizer, [WRONG_SHAPE])
        run = assemble(OrchestrationOptions(rl=False), m, judge, rl_start_fn=lambda task: None,
                       agent_factory=agent_factory(AGENTS_REGISTRY))
        from orchestration.assemble import solve_with_orchestration
        by_name = {a["name"]: a for a in AGENTS_REGISTRY}
        entries = [by_name["Modifier"], by_name["Highlighter"]] if agents is None else agents
        context = {"grid_repr_type": "concise", "test_input_grid": make_task().test_subtask.train_inp}
        solve_with_orchestration(make_task(), run, agents=entries, auxiliary_info=context,
                                 system_run_config=SystemRunConfig(max_system_iterations=3,
                                                                   agent_run_config=AgentRunConfig(max_agent_iterations=1)))
        return m.runner.prompts, ROLE_INSTRUCTIONS

    def test_each_agent_is_asked_with_its_own_role(self, tiny_tokenizer):
        prompts, roles = self.run(tiny_tokenizer)
        assert len(prompts) == 2 and prompts[0] != prompts[1]
        assert roles["Modifier"].splitlines()[0] in prompts[0] and roles["Highlighter"].splitlines()[0] in prompts[1]
        assert roles["Highlighter"].splitlines()[0] not in prompts[0]

    def test_an_agent_with_no_role_text_is_asked_without_one(self, tiny_tokenizer, monkeypatch):
        from data.configs.agents_config import AGENTS_REGISTRY, ROLE_INSTRUCTIONS
        monkeypatch.delitem(ROLE_INSTRUCTIONS, "Generalizer")
        prompts, _ = self.run(tiny_tokenizer, agents=[AGENTS_REGISTRY[0]])
        assert prompts and all("Role:" not in p for p in prompts)


class TestEveryOptionOverTheRoster:
    @staticmethod
    def solve(tiny_tokenizer, options, replies):
        from data.configs.agents_config import AGENTS_REGISTRY
        from orchestration.assemble import solve_with_orchestration
        m = module(tiny_tokenizer, replies)
        run = assemble(options, m, judge, rl_start_fn=lambda task: None, agent_factory=agent_factory(AGENTS_REGISTRY))
        by_name = {a["name"]: a for a in AGENTS_REGISTRY}
        context = {"grid_repr_type": "concise", "test_input_grid": make_task().test_subtask.train_inp}
        solve_with_orchestration(make_task(), run, agents=[by_name["Modifier"], by_name["Highlighter"]],
                                 auxiliary_info=context,
                                 system_run_config=SystemRunConfig(max_system_iterations=2,
                                                                   agent_run_config=AgentRunConfig(max_agent_iterations=1)))
        return m.runner.prompts

    def test_the_refinement_loop_and_the_roles_work_together(self, tiny_tokenizer):
        prompts = self.solve(tiny_tokenizer, OrchestrationOptions(rl=False, refine_rounds=2, feedback=True),
                             ["1,3:\n1 001"])
        assert any("Role:" in p for p in prompts) and any("Attempt 1" in p for p in prompts)

    def test_the_tools_and_the_roles_work_together(self, tiny_tokenizer):
        prompts = self.solve(tiny_tokenizer, OrchestrationOptions(rl=False, info_tools=True), ["1,3:\n1 001"])
        assert any("Role:" in p and "REQUEST: <tool>" in p for p in prompts)

    def test_the_model_made_decisions_and_the_roles_work_together(self, tiny_tokenizer):
        options = OrchestrationOptions(rl=False, decision="llm", coordinator="llm")
        prompts = self.solve(tiny_tokenizer, options, ["1,3:\n1 001"])
        assert prompts and any("Role:" in p for p in prompts)
