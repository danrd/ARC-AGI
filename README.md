## About ARC-AGI

The **Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI)** is a benchmark designed to measure an AI system's fluid intelligence — its ability to efficiently learn new skills and solve novel, unseen problems. 

Unlike traditional machine learning benchmarks that test pattern matching on vast amounts of specialized training data, ARC-AGI focuses on few-shot learning based on core human knowledge priors (e.g., object cohesion, basic geometry, and topology). The fundamental goal of the benchmark is to shift AI research away from brute-force memorization and toward systems capable of genuine abstraction, reasoning, and generalization.

### Challenge Milestones
The ARC-AGI dataset has evolved through three primary iterations, each expanding the scope of the challenge:

* **Version 1 (Classic):** The original dataset establishing the baseline for artificial fluid intelligence. It focuses on core cognitive priors — such as object cohesion, basic geometry, and topology — evaluated through static input-output grids.
* **Version 2 (Advanced):** A scaled progression of the classic dataset. It maintains the fundamental spirit and structural format of the original tasks but introduces significantly higher conceptual complexity and compositional difficulty.
* **Version 3 (Current / Interactive):** The latest and most ambitious iteration of the benchmark. This version shifts the paradigm from static grid transformations to interactive problem-solving, emphasizing sequential decision-making and environmental feedback.

---

## Proposed Approach: A Neuro-Symbolic Multi-Agent Architecture

This repository introduces a novel framework tailored for the ARC-AGI challenge, built on the synthesis of **Multi-Agent Systems (MAS)** and **Neuro-Symbolic AI**. The core objective of this architecture is to maximize flexibility in both task distribution and feature processing.

### Key Concepts

1. **Agent Specialization:** At the multi-agent level, the system dynamically distributes responsibilities. Instead of a monolithic solver, specialized agents tackle different cognitive and procedural aspects of a given ARC task.
2. **Tri-Modal Task Representation:** To handle the extreme diversity of ARC grids, each agent is equipped to process features using three distinct conceptual representations:
   * **Symbolic:** Exact rule-based logic, structural representations, and discrete transformations.
   * **Sub-symbolic:** Neural-based pattern recognition, latent embeddings, and heuristic approximations.
   * **Interactive:** Environment-driven exploration, trial-and-error state manipulation, and iterative feedback loops.

---

## Implementation

The repository structure mirrors the tri-modal design directly:

* `symbolic/` — the **Symbolic** path: rule-based solvers and structural task analysis.
* `subsymbolic/` — the **Sub-symbolic** path: LLM prompt construction and inference backends.
* `rl/` — the **Interactive** path: the ARC grid-world environment and RL-based training.
* `orchestration/` — implements the integration of these three concepts into the actual multi-agent system: a coordinator/agent graph (built on LangGraph) that embodies the approach's core idea in line with current multi-agent system design practice, rather than hardcoding a single solving strategy.

Supporting code lives alongside these: `data/` holds datasets, prompt templates, and per-module configs; `utils/` holds repository-wide helpers not tied to one conceptual approach; `scripts/` holds the measurement tools (comparing prompt arms, measuring the search over a dataset); `tests/` holds the test suite.

`ARCHITECTURE.md` says how the packages depend on each other, where to start reading for a given question, and what tends to bite. Its module map at the bottom is generated (`python scripts/module_map.py --write`), and the layering it describes is a checked contract (`lint-imports`).

### Working on it

```
pip install -e ".[rl,test]"          # or the extras you need: hf, vllm, llama-cpp, openrouter, logging
                                     # ruff and import-linter are in the dev group: uv sync --group dev
pytest                               # the test suite
ruff check .                         # lint
lint-imports                         # the layering contract
python scripts/module_map.py --check # ARCHITECTURE.md's module map is current
python -m orchestration              # one task through the coordinator -> agent -> module graph
```

---

## Repository Status & Experimental Data

Currently, the `main` branch contains the implementation of the **core components and foundational architecture** of our approach.

⚠️ **Experimental Hold:** The full experimental pipeline, specific solver configurations, and benchmark evaluations remain private at this time. To maintain integrity during the ongoing stage of the challenge, the experimental branch will remain hidden. The complete codebase and results will be fully open-sourced upon the conclusion of the current challenge phase and the subsequent publication of our accompanying research paper.

---

## License

This repository is licensed under the [MIT License](LICENSE).