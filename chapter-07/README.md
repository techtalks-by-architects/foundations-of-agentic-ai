# Chapter 7: Evaluation, Testing & Production Readiness — Source Code

Five examples covering the testing and operational layers for agentic systems.

## Setup

See `README.md` for one-time setup (shared venv, `.env`, and dependencies).

## Examples

| Folder | Level | What it shows |
|---|---|---|
| `01-unit-tests/` | Unit | Deterministic tests for tools, guardrails, validation (no LLM) |
| `02-integration-tests/` | Integration | Tests with real LLM calls asserting on behavior, not exact text |
| `03-eval-suite/` | Eval | Systematic quality measurement across categories with pass rates |
| `04-llm-judge/` | Judge | Using a second LLM to evaluate response quality on multiple criteria |
| `05-cost-monitor/` | Cost | Token tracking, budget enforcement, per-request cost reporting |

## Running

All examples run from the repo root:

```bash

# 01 — Unit Tests (no LLM needed)
python chapter-07/01-unit-tests/test_tools.py
python -m pytest chapter-07/01-unit-tests/test_tools.py -v

# 02 — Integration Tests (needs LLM)
python chapter-07/02-integration-tests/test_agent.py

# 03 — Eval Suite (needs LLM)
python chapter-07/03-eval-suite/eval.py
python chapter-07/03-eval-suite/eval.py --verbose

# 04 — LLM-as-Judge (needs LLM)
python chapter-07/04-llm-judge/judge.py

# 05 — Cost Monitor (needs LLM)
python chapter-07/05-cost-monitor/monitor.py "Where is my order ORD-42?"
python chapter-07/05-cost-monitor/monitor.py "Cancel order ORD-43 because I changed my mind"
```

## Key Concepts

| Concept | Example | Key idea |
|---|---|---|
| Deterministic testing | 01 | Test everything around the LLM without the LLM |
| Behavioral assertions | 02 | Assert on tool calls and patterns, not exact text |
| Pass rate tolerance | 02 | Run each test 3x, require majority pass |
| Category tracking | 03 | Measure quality per category, catch regressions |
| LLM-as-judge | 04 | Score responses on helpfulness, accuracy, tone, completeness |
| Token budgets | 05 | Track and cap token usage per request |
| Cost estimation | 05 | Calculate $ cost from token counts |
