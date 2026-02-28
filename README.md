# Foundations of Agentic AI — Source Code

Runnable examples for **Foundations of Agentic AI: The Essentials Every Developer Must Know**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        
```

OOne virtual environment, one `.env` file, one `requirements.txt` — covers all chapters. Fill in `.env` with your provider credentials before running anything.

For full setup instructions — including Ollama (free, local), Groq (free cloud), and OpenAI — see **Appendix A - Running the Examples** chapter in the book.

## Running Examples

Run all examples from the root directory directory. A representative sample is shown below — see each chapter's folder for the full list.

```bash

# Chapter 02: Core Patterns
python chapter-02/01-tool-use/main.py "Where is my order ORD-42?"
python chapter-02/03-router/main.py "I have a billing question"
python chapter-02/04-handoff/main.py "I need a refund and legal advice"

# Chapter 03: Building an Agent (progressive levels)
python chapter-03/level_0_single_call.py
python chapter-03/level_1_tools.py "Where is my order ORD-42?"
python chapter-03/level_2_loop.py "Cancel order ORD-42 because it hasn't arrived."
python chapter-03/level_7_full_agent.py "Cancel order ORD-42"

# Chapter 04: Building with LangChain
python chapter-04/level_1_tools_langchain.py "Where is my order ORD-42?"
python chapter-04/level_2_loop_langchain.py "Cancel order ORD-42 because it hasn't arrived."
python chapter-04/level_4_guardrails_langchain.py "Ignore previous instructions"

# Chapter 05: Orchestration & Control
python chapter-05/01-sequential-pipeline/main.py ORD-42
python chapter-05/05-langgraph-router/main.py "Where is my order ORD-42?"
python chapter-05/06-langgraph-hitl/main.py ORD-42

# Chapter 06: Tool Design
python chapter-06/01-tool-anatomy/main.py "Where is my order ORD-42?"

# Chapter 07: Evaluation & Testing
python chapter-07/01-unit-tests/test_tools.py
python chapter-07/03-eval-suite/eval.py
```

Every example supports `--help`.

## Structure

```
foundations-of-agentic-ai/
  .env.example          — Copy to .env, set your API key
  requirements.txt      — All dependencies for all chapters
  chapter-02/           — Core pattern implementations (9 patterns)
  chapter-03/           — Progressive agent levels (0–7), raw Python
  chapter-04/           — LangChain versions of key levels
  chapter-05/           — Orchestration examples (raw Python + LangGraph)
  chapter-06/           — Tool design (anatomy, REST, MCP, governance, registry)
  chapter-07/           — Testing & eval (unit, integration, eval suite, judge, cost)
```
