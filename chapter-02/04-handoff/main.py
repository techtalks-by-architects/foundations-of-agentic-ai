"""
Pattern 4: Handoff

A triage agent works on a task but reaches a point where a different agent—with
different tools or expertise—should take over. Handoff transfers control AND
context mid-execution:

  1. Triage agent tries to answer using its own tools (knowledge base).
  2. If it can't fully handle the request, it calls handoff_to_specialist
     with a department and a summary of what it's already tried.
  3. The platform catches the handoff, packages the context (summary +
     conversation history), and spins up a specialist agent.
  4. The specialist continues with full context — no information is lost.

Unlike routing (which dispatches BEFORE any agent runs), handoff happens
DURING the loop — Agent A has already done some work.

Run from repo root:
  python chapter-02/04-handoff/main.py
  python chapter-02/04-handoff/main.py "I need a refund and legal advice about my contract."
  python chapter-02/04-handoff/main.py "What's your refund policy?"
  python chapter-02/04-handoff/main.py --help
"""
import json
import os
import sys
from pathlib import Path

# Allow importing shared LLM client from chapter-02/common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import chat, ToolCall

MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")


# =============================================================================
# Triage agent: tries the knowledge base first, then hands off to a specialist
# =============================================================================

TRIAGE_SYSTEM = """You are a triage agent.

1. Search the knowledge base ONCE to find relevant information.
2. If the knowledge base has the answer, respond directly to the user. Do NOT search again.
3. If the question requires specialist knowledge (engineering, legal, finance), hand off to the right department with a summary of what you've already tried.

Do not repeat searches. Once you have information from the knowledge base, use it to answer."""

# Simulated knowledge base
KNOWLEDGE_BASE = {
    "refund": "Refunds are processed within 5-7 business days.",
    "shipping": "Standard shipping takes 3-5 days. Express is 1-2 days.",
}


def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant articles."""
    q = query.lower()
    for k, v in KNOWLEDGE_BASE.items():
        if k in q:
            return v
    return "No relevant article found."


# =============================================================================
# The handoff tool — this is how context is transferred structurally.
#
# The triage agent calls this like any other tool, but instead of returning
# data, it spins up a different agent with the conversation context.
# The specialist gets:
#   (1) department — which specialist to route to
#   (2) summary — what the triage agent already tried (written by the LLM)
#   (3) conversation history — passed by the platform, not by the LLM
#
# This is a "structured transfer of context" — not just a redirect, but a
# package of everything the next agent needs to continue.
# =============================================================================

def handoff_to_specialist(department: str, summary: str) -> str:
    """Transfer this conversation to a specialist agent.

    Args:
        department: 'engineering', 'legal', or 'finance'
        summary: one-paragraph summary of the issue and what's been tried
    """
    specialist_fn = SPECIALISTS.get(department, specialist_general)

    # In production, this would spin up a real specialist agent with its
    # own tools, system prompt, and the conversation history. Here we
    # simulate it with a function that receives the context.
    return specialist_fn(summary=summary)


# =============================================================================
# Specialists: in production these would be full agents with their own tools.
# Each receives the summary so it knows what the triage agent already tried.
# =============================================================================

def specialist_engineering(summary: str) -> str:
    return f"[Engineering] We'll investigate this. Context from triage: {summary[:120]}"


def specialist_legal(summary: str) -> str:
    return f"[Legal] This has been escalated to our legal team. Context from triage: {summary[:120]}"


def specialist_finance(summary: str) -> str:
    return f"[Finance] Our billing team will follow up. Context from triage: {summary[:120]}"


def specialist_general(summary: str) -> str:
    return f"[General] A team member will review this. Context from triage: {summary[:120]}"


SPECIALISTS = {
    "engineering": specialist_engineering,
    "legal": specialist_legal,
    "finance": specialist_finance,
}


# =============================================================================
# Tool execution
# =============================================================================

TOOLS = [search_knowledge_base, handoff_to_specialist]


def execute_tool(name: str, arguments: dict) -> str:
    """Run the named tool with the given arguments."""
    if name == "handoff_to_specialist":
        return handoff_to_specialist(
            arguments.get("department", "engineering"),
            arguments.get("summary", ""),
        )
    if name == "search_knowledge_base":
        return search_knowledge_base(arguments.get("query", ""))
    return json.dumps({"error": f"Unknown tool: {name}"})


# =============================================================================
# Triage loop
#
# The loop runs like a normal ReAct agent. The key difference: when the agent
# calls handoff_to_specialist, we detect it, execute the handoff (which
# transfers context to the specialist), print the handoff, and stop.
# This is a one-way handoff — the specialist finishes the task.
# =============================================================================

def run(user_message: str, max_iterations: int = 10) -> str:
    messages = [
        {"role": "system", "content": TRIAGE_SYSTEM},
        {"role": "user", "content": user_message},
    ]
    for i in range(max_iterations):
        response = chat(messages=messages, tools=TOOLS)

        if response.tool_calls:
            tc = response.tool_calls[0]
            result = execute_tool(tc.name, tc.arguments)

            if tc.name == "handoff_to_specialist":
                # Handoff detected: triage decided it can't handle this.
                # The tool has already transferred context to the specialist.
                dept = tc.arguments.get("department", "unknown")
                summary = tc.arguments.get("summary", "")
                print(f"  [Handoff] Triage → {dept}")
                print(f"  [Context] \"{summary[:100]}\"")
                return result

            # Normal tool call (e.g., knowledge base search) — continue the loop
            print(f"  [Tool] {tc.name}({tc.arguments}) → {result[:80]}")
            tool_calls_api = [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }]
            messages.append({"role": "assistant", "content": response.content or None, "tool_calls": tool_calls_api})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            # Triage agent answered directly — no handoff needed
            return response.content or ""

    return "[Triage did not converge]"


# =============================================================================
# Example usages (run from chapter-02)
# =============================================================================
#
#   # Default: refund + legal (may trigger handoff)
#   python chapter-02/04-handoff/main.py
#
#   # Complex question -> triage searches KB, then hands off to legal
#   python chapter-02/04-handoff/main.py "I need a refund and legal advice about my contract."
#
#   # Simple question -> knowledge base only, no handoff
#   python chapter-02/04-handoff/main.py "What's your refund policy?"
#
#   # Show this help
#   python chapter-02/04-handoff/main.py --help
#
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python chapter-02/04-handoff/main.py [ \"your message\" ]")
        print()
        print("Examples:")
        print('  python chapter-02/04-handoff/main.py "I need a refund and legal advice about my contract."')
        print('  python chapter-02/04-handoff/main.py "What\'s your refund policy?"')
        sys.exit(0)

    query = sys.argv[1] if len(sys.argv) > 1 else "I need a refund and legal advice about my contract."
    print("User:", query)
    print("Response:", run(query))
