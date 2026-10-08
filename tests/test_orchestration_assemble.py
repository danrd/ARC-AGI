"""The options, run through the real graph with the model, the second model and RL canned."""
import numpy as np

from orchestration.assemble import Orchestration, assemble, solve_with_orchestration
from orchestration.configs import AgentRunConfig, ExperimentConfig, OrchestrationOptions, SystemRunConfig
from orchestration.trace import Tracer
from rl.arc_task import ARCSubtask, ARCTask
from subsymbolic.prompt_builder import PromptingConfig
from subsymbolic.subsymbolic_module import SubsymbolicModule

GOOD = "2,3:\n1 001\n2 002"
WRONG_SHAPE = "1,3:\n1 001"
SAME_SHAPE_WRONG = "2,3:\n1 001\n2 020"


def make_task():
    pairs = [(np.array([[0, 1, 0], [0, 0, 0]]), np.array([[0, 1, 0], [0, 2, 0]])),
             (np.array([[1, 0, 0], [0, 0, 0]]), np.array([[1, 0, 0], [0, 2, 0]]))]
    subtasks = [ARCSubtask(f"t_{i}", a, b) for i, (a, b) in enumerate(pairs)]
    return ARCTask(label="t", subtasks=subtasks, test_inp=np.array([[0, 0, 1], [0, 0, 0]]),
                   test_out=np.array([[0, 0, 1], [0, 0, 2]]))


class Runner:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]

    def close(self):
        pass


def module(tokenizer, replies):
    config = ExperimentConfig(prompt=PromptingConfig(blocks=["general_instruction", "examples", "output_format"],
                                                     resolvers=["examples"], filters=["grid"], token_limit=20000))
    return SubsymbolicModule(config, tokenizer, runner=Runner(replies))


def judge(task, grid):
    """A second model that is right: it accepts the task's own answer and nothing else."""
    return bool(np.array_equal(np.asarray(grid), task.test_subtask.train_out))


def verifier(*verdicts):
    calls = []

    def verify(task, grid):
        calls.append(np.asarray(grid).tolist())
        return verdicts[min(len(calls) - 1, len(verdicts) - 1)]

    verify.calls = calls
    return verify


def solve(task, run, rounds=3):
    context = {"grid_repr_type": "concise", "test_input_grid": task.test_subtask.train_inp}
    return solve_with_orchestration(task, run, auxiliary_info=context, system_run_config=SystemRunConfig(
        agent_run_config=AgentRunConfig(max_agent_iterations=rounds)))


class NoRL:
    """An agent that has no RL job: the graph asks for a handle and gets none."""


def no_rl(task):
    return None


class TestTheDefaults:
    def test_the_rule_decides_and_the_run_is_traced_by_phase(self, tiny_tokenizer):
        task, tracer = make_task(), Tracer()
        run = assemble(OrchestrationOptions(), module(tiny_tokenizer, [GOOD]), verifier(True), tracer=tracer,
                       rl_start_fn=no_rl)
        result = solve(task, run)
        assert result["validated"] is not False and result["accepted_source"] == "llm"
        phases = set(tracer.summary())
        assert {"symbolic", "llm", "verify", "decide"} <= phases
        assert "rl_wait" not in phases and run.decisions == []

    def test_the_second_model_is_asked_once_however_many_pieces_want_its_word(self, tiny_tokenizer):
        verify = verifier(True)
        run = assemble(OrchestrationOptions(feedback=True, refine_rounds=2), module(tiny_tokenizer, [GOOD]), verify,
                       rl_start_fn=no_rl)
        solve(make_task(), run)
        assert len(verify.calls) == 1


class TestTheRefinementLoop:
    def test_a_wrong_first_answer_is_corrected_within_one_round_of_the_graph(self, tiny_tokenizer):
        m = module(tiny_tokenizer, [WRONG_SHAPE, GOOD])
        run = assemble(OrchestrationOptions(refine_rounds=3), m, verifier(True), rl_start_fn=no_rl)
        result = solve(make_task(), run)
        assert result["accepted_source"] == "llm"
        calls = m.runner.prompts
        assert len(calls) == 2 and "Memory" not in calls[0] and "1x3" in calls[1]
        assert run.module.last.attempts[0].notes and run.module.last.accepted

    def test_without_the_option_the_model_is_asked_the_same_thing_twice(self, tiny_tokenizer):
        m = module(tiny_tokenizer, [WRONG_SHAPE, GOOD])
        run = assemble(OrchestrationOptions(), m, judge, rl_start_fn=no_rl)
        solve(make_task(), run)
        assert len(m.runner.prompts) == 2 and "Memory" not in m.runner.prompts[1]


class TestTheFeedbackOnTheGraphsRetry:
    def test_the_retry_after_a_refusal_has_the_answer_and_the_refusal_in_front_of_it(self, tiny_tokenizer):
        m = module(tiny_tokenizer, [SAME_SHAPE_WRONG, GOOD])
        run = assemble(OrchestrationOptions(feedback=True), m, verifier(False, True), rl_start_fn=no_rl)
        result = solve(make_task(), run)
        assert result["accepted_source"] == "llm"
        first, second = m.runner.prompts
        assert "Memory" not in first and "Attempt 1" in second and "second model" in second.lower()
        assert SAME_SHAPE_WRONG.splitlines()[0] in second


class TestTheToolsOption:
    def test_a_model_that_asks_is_answered_and_asked_again(self, tiny_tokenizer):
        m = module(tiny_tokenizer, ["REQUEST: summary", GOOD])
        options = OrchestrationOptions(info_tools=True)
        run = assemble(options, m, verifier(True), rl_start_fn=no_rl, tracer=Tracer())
        result = solve(make_task(), run)
        assert result["accepted_source"] == "llm"
        assert "REQUEST: <tool>" in m.runner.prompts[0] and "Requested information" in m.runner.prompts[1]
        assert "tools" in run.tracer.summary()


class TestTheDecisionOption:
    def test_a_model_that_says_give_up_ends_the_agent_where_the_rule_would_have_retried(self, tiny_tokenizer):
        m = module(tiny_tokenizer, [WRONG_SHAPE, GOOD])
        decider = module(tiny_tokenizer, ['{"status": "INVALID", "action": "give_up", "reasoning": "stop"}'])
        run = assemble(OrchestrationOptions(decision="llm"), m, judge, decider=decider, rl_start_fn=no_rl)
        result = solve(make_task(), run)
        assert len(m.runner.prompts) == 1 and result["accepted_source"] != "llm"
        assert run.decisions[-1]["model"] == "give_up" and run.decisions[-1]["used_model"] is True

    def test_a_model_that_cannot_be_read_leaves_the_rule_and_the_run_is_as_without_it(self, tiny_tokenizer):
        m = module(tiny_tokenizer, [WRONG_SHAPE, GOOD])
        decider = module(tiny_tokenizer, ["no idea"])
        run = assemble(OrchestrationOptions(decision="llm"), m, judge, decider=decider, rl_start_fn=no_rl)
        result = solve(make_task(), run)
        assert result["accepted_source"] == "llm" and len(m.runner.prompts) == 2
        assert all(not d["used_model"] for d in run.decisions)


class TestTheWaitForRL:
    def test_the_time_the_graph_waits_for_rl_is_a_span(self, tiny_tokenizer):
        class Handle:
            def poll(self):
                return None

            def wait(self, timeout):
                return {"status": "ok", "solution": np.array([[0, 0, 1], [0, 0, 2]])}

            def cancel(self, timeout=5.0):
                pass

            process = type("P", (), {"is_alive": lambda self: False})()

        tracer = Tracer()
        run = assemble(OrchestrationOptions(), module(tiny_tokenizer, [WRONG_SHAPE]), verifier(True), tracer=tracer,
                       rl_start_fn=lambda task: Handle())
        result = solve(make_task(), run)
        assert result["accepted_source"] == "rl" and tracer.summary()["rl_wait"]["calls"] == 1


def test_the_options_default_to_the_system_as_it_was():
    options = OrchestrationOptions()
    assert (options.decision, options.coordinator, options.refine_rounds, options.feedback, options.info_tools) == \
        ("deterministic", "deterministic", 0, False, False)
    from orchestration.configs import ExperimentConfig as Config
    config = Config.from_dict({"system": {"orchestration": {"decision": "llm", "refine_rounds": 2}}})
    assert config.system.orchestration.decision == "llm" and config.system.orchestration.refine_rounds == 2
    assert Config().system.orchestration == options


class TestTheCoordinatorOption:
    AGENTS = [{"index": 0, "name": "First"}, {"index": 1, "name": "Second"}, {"index": 2, "name": "Third"}]

    def factory(self, entry):
        from orchestration.graph import AgentInvConfig, ModuleInvConfig
        return AgentInvConfig(agent_index=entry["index"], agent_name=entry["name"],
                              initial_module=ModuleInvConfig(0, "symbolic"),
                              available_modules=[{"index": 0, "name": "symbolic"}, {"index": 1, "name": "subsymbolic"}])

    def names(self, result):
        return [r.name for r in result["history"] if r.level == "agent"]

    def test_a_failed_agent_hands_the_task_on_and_the_roster_runs_out(self, tiny_tokenizer):
        run = assemble(OrchestrationOptions(), module(tiny_tokenizer, [WRONG_SHAPE]), judge, rl_start_fn=no_rl,
                       agent_factory=self.factory)
        result = solve_with_orchestration(make_task(), run, agents=self.AGENTS, auxiliary_info={
            "grid_repr_type": "concise", "test_input_grid": make_task().test_subtask.train_inp},
            system_run_config=SystemRunConfig(max_system_iterations=5, agent_run_config=AgentRunConfig(max_agent_iterations=1)))
        assert self.names(result) == ["First", "Second", "Third"] and not result["validated"]

    def test_an_agent_that_validates_ends_it(self, tiny_tokenizer):
        run = assemble(OrchestrationOptions(), module(tiny_tokenizer, [GOOD]), judge, rl_start_fn=no_rl,
                       agent_factory=self.factory)
        result = solve_with_orchestration(make_task(), run, agents=self.AGENTS, auxiliary_info={
            "grid_repr_type": "concise", "test_input_grid": make_task().test_subtask.train_inp})
        assert self.names(result) == ["First"] and result["validated"]

    def test_the_model_coordinator_picks_the_next_agent_among_those_left(self, tiny_tokenizer):
        decider = module(tiny_tokenizer, ['{"status": "INVALID", "delegate_to_agent": 2}'])
        run = assemble(OrchestrationOptions(coordinator="llm"), module(tiny_tokenizer, [WRONG_SHAPE]), judge,
                       decider=decider, rl_start_fn=no_rl, agent_factory=self.factory)
        result = solve_with_orchestration(make_task(), run, agents=self.AGENTS, auxiliary_info={
            "grid_repr_type": "concise", "test_input_grid": make_task().test_subtask.train_inp},
            system_run_config=SystemRunConfig(max_system_iterations=5, agent_run_config=AgentRunConfig(max_agent_iterations=1)))
        assert self.names(result)[:2] == ["First", "Third"]
