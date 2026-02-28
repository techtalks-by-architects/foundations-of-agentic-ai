"""
Pattern 5: Orchestrator–Workers

A single orchestrator agent breaks a goal into subtasks and delegates each to a
specialist worker (research, code, test). Workers are invoked as tools; the
orchestrator sees only their results, then synthesizes a final answer. This is
scatter-gather for agentic systems: one agent plans, many workers execute.

Run from repo root:
  python chapter-02/05-orchestrator-workers/main.py [ "Add rate limiting to the /api/orders endpoint" ]
Requires OPENAI_API_KEY in environment (or uses mock responses).
"""
import json
import sys
from pathlib import Path

# Allow importing shared LLM client from chapter-02/common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import chat


# -----------------------------------------------------------------------------
# Orchestrator prompt: plan first, then call workers step by step
# -----------------------------------------------------------------------------

ORCHESTRATOR_PROMPT = """You are a project orchestrator.
Given a goal, break it into subtasks. For each subtask, call the appropriate worker.
After all workers respond, synthesize a final result.

Available workers:
- research_worker: finds information, reads docs, summarizes findings
- code_worker: writes, reviews, or fixes code
- test_worker: writes and runs tests

Respond with a plan first, then execute it step by step."""


# -----------------------------------------------------------------------------
# Workers: each is a tool the orchestrator can call (here, simplified LLM calls)
# -----------------------------------------------------------------------------

def research_worker(query: str) -> str:
    """Delegate a research question to the research agent."""
    response = chat(messages=[
        {"role": "system", "content": "You are a research assistant. Summarize findings concisely."},
        {"role": "user", "content": query},
    ])
    return response.content or f"[Research] Looked into: {query[:50]}..."


def code_worker(task: str, context: str = "") -> str:
    """Delegate a coding task to the code agent."""
    response = chat(messages=[
        {"role": "system", "content": "You are a coding assistant. Provide short code or steps."},
        {"role": "user", "content": f"Task: {task}\nContext: {context}"},
    ])
    return response.content or f"[Code] Draft for: {task[:50]}..."


def test_worker(code: str, requirements: str = "") -> str:
    """Delegate testing to the test agent."""
    response = chat(messages=[
        {"role": "system", "content": "You are a test assistant. Suggest test cases briefly."},
        {"role": "user", "content": f"Code snippet: {code[:200]}\nRequirements: {requirements}"},
    ])
    return response.content or "[Test] Suggested test cases."


WORKERS = [research_worker, code_worker, test_worker]


def execute_tool(name: str, arguments: dict) -> str:
    """Run the worker by name and return its result as a string."""
    fn = {f.__name__: f for f in WORKERS}.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown worker: {name}"})
    try:
        return fn(**arguments)
    except Exception as e:
        return json.dumps({"error": str(e)})


# -----------------------------------------------------------------------------
# Orchestrator loop: same shape as ReAct, but "tools" are worker agents
# -----------------------------------------------------------------------------

def run(goal: str, max_iterations: int = 10) -> str:
    messages = [
        {"role": "system", "content": ORCHESTRATOR_PROMPT},
        {"role": "user", "content": goal},
    ]
    for i in range(max_iterations):
        response = chat(messages=messages, tools=WORKERS)
        if response.tool_calls:
            for tc in response.tool_calls:
                result = execute_tool(tc.name, tc.arguments)
                print(f"  [Worker] {tc.name} -> {result[:60]}...")
            # Append assistant message and tool results for next turn
            tool_calls_api = [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}} for tc in response.tool_calls]
            messages.append({"role": "assistant", "content": response.content or None, "tool_calls": tool_calls_api})
            for tc in response.tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": execute_tool(tc.name, tc.arguments)})
        else:
            # Orchestrator produced final synthesis
            return response.content or ""
    return "[Orchestrator did not converge]"


# -----------------------------------------------------------------------------
# Example usages (run from chapter-02)
# -----------------------------------------------------------------------------
#
#   # Default: rate limiting task
#   python chapter-02/05-orchestrator-workers/main.py
#
#   # Add rate limiting
#   python chapter-02/05-orchestrator-workers/main.py "Add rate limiting to the /api/orders endpoint"
#
#   # Custom goal
#   python chapter-02/05-orchestrator-workers/main.py "Document the auth flow and add a unit test"
#
#   # Show this help
#   python chapter-02/05-orchestrator-workers/main.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python chapter-02/05-orchestrator-workers/main.py [ \"goal\" ]")
        print()
        print("Examples:")
        print('  python chapter-02/05-orchestrator-workers/main.py "Add rate limiting to the /api/orders endpoint"')
        print('  python chapter-02/05-orchestrator-workers/main.py "Document the auth flow and add a unit test"')
        sys.exit(0)
    goal = sys.argv[1] if len(sys.argv) > 1 else "Add rate limiting to the /api/orders endpoint"
    print("Goal:", goal)
    print("Result:", run(goal))
