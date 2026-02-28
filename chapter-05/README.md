# Chapter 5: Orchestration & Control — Source Code

Six examples showing orchestration patterns — three built from scratch with
raw Python + OpenAI, three using LangGraph.

## Setup

See `README.md` for one-time setup (shared venv, `.env`, and dependencies).

## Examples

### Raw Python (no framework)

| Folder | Pattern | What it shows |
|---|---|---|
| `01-sequential-pipeline/` | Sequential Pipeline | Order cancellation: lookup → policy → cancel/reject |
| `02-router/` | Router | Classify message → dispatch to specialist agent |
| `03-handoff/` | Handoff | Support agent gathers context, hands off to billing |

### LangGraph

| Folder | Pattern | What it shows |
|---|---|---|
| `04-langgraph-pipeline/` | Sequential Pipeline | Same cancellation flow as a typed state graph with conditional edges |
| `05-langgraph-router/` | Router | Same routing as conditional edges from a classifier node |
| `06-langgraph-hitl/` | Human-in-the-Loop | Refund workflow that pauses for human approval (interrupt + checkpoint) |

## Running

All examples run from the repo root:

```bash

# Raw Python examples
python chapter-05/01-sequential-pipeline/main.py ORD-42
python chapter-05/02-router/main.py "Where is my order ORD-42?"
python chapter-05/03-handoff/main.py "I received the wrong item in ORD-42, I want a refund"

# LangGraph examples
python chapter-05/04-langgraph-pipeline/main.py ORD-42
python chapter-05/05-langgraph-router/main.py "I want to cancel order ORD-43"
python chapter-05/06-langgraph-hitl/main.py ORD-42   # will pause for approval input
```

Every example supports `--help`.

## Key Differences: Raw vs. LangGraph

| | Raw Python | LangGraph |
|---|---|---|
| **Pipeline** | Function composition (`a → b → c`) | StateGraph with typed state + conditional edges |
| **Router** | `dict` lookup | Conditional edges from classifier node |
| **HITL** | Hard to build (need state persistence + pause/resume) | Built-in: `interrupt_before` + `MemorySaver` |
| **Dependencies** | `openai` only | `langgraph`, `langchain-openai` |
| **Debugging** | Print/log statements | Graph visualization + checkpoint inspection |
