"""
Pattern 2: ReAct (Reason + Act)

ReAct interleaves *thought* (the model reasons in natural language) with *action*
(the model calls a tool). After each action, the result is fed back as an
*observation*, and the loop continues until the model produces a final answer.
This gives you an inspectable trace of why each action was chosen—essential
for debugging and auditing.

Run from repo root:
  python chapter-02/02-react/main.py [ "What's the status of order ORD-7890?" ]
Requires OPENAI_API_KEY in environment (or uses mock responses).
"""
import json
import sys
from pathlib import Path

# Allow importing shared LLM client from chapter-02/common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import chat, ToolCall


# -----------------------------------------------------------------------------
# System prompt: ask the model to think step by step (thought → action → observation)
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a customer support agent.
When you need information, use the available tools.
Think step by step. For each step:
1. Thought: reason about what you know and what you need.
2. Action: call a tool if needed.
3. Observation: review the result.
Repeat until you can answer the user's question."""


# -----------------------------------------------------------------------------
# Tools: allow-list for this agent (same idea as Pattern 1: Tool Use)
# -----------------------------------------------------------------------------

def search_orders(customer_id: str, status: str = "all") -> list[dict]:
    """Search orders for a customer, optionally filtered by status."""
    orders = [
        {"order_id": "ORD-1234", "status": "shipped", "total": 59.99},
        {"order_id": "ORD-7890", "status": "pending", "total": 119.50},
    ]
    if status != "all":
        orders = [o for o in orders if o["status"] == status]
    return orders


TOOLS = [search_orders]


def execute_tool(name: str, arguments: dict) -> str:
    """Run the tool by name and return its result as a JSON string for the conversation."""
    fn = {f.__name__: f for f in TOOLS}.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return json.dumps(fn(**arguments))
    except Exception as e:
        return json.dumps({"error": str(e)})


# -----------------------------------------------------------------------------
# ReAct loop: thought → action → observation, bounded and observable
# -----------------------------------------------------------------------------

def run(user_message: str, max_iterations: int = 10) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for i in range(max_iterations):
        # 1. Decide: LLM may return tool calls (action) or a final answer
        response = chat(messages=messages, tools=TOOLS)

        if response.tool_calls:
            # 2. Act: execute each tool and log for observability (ReAct trace)
            for tc in response.tool_calls:
                result = execute_tool(tc.name, tc.arguments)
                print(f"  [Trace] iteration={i} action={tc.name} args={tc.arguments} -> {result[:80]}...")
            # Build assistant message in OpenAI format for the next API call
            tool_calls_api = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in response.tool_calls
            ]
            messages.append({"role": "assistant", "content": response.content or None, "tool_calls": tool_calls_api})
            # 3. Observe: append each tool result so the model can reason about it next turn
            for tc in response.tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": execute_tool(tc.name, tc.arguments)})
        else:
            # No tool calls: model produced a final answer
            return response.content or ""

    # Safety: bounded loop — escalate instead of looping forever
    print("[Escalating: agent did not converge]")
    return "I've escalated your request to a human agent."


# -----------------------------------------------------------------------------
# Example usages (run from chapter-02)
# -----------------------------------------------------------------------------
#
#   # Default: ask about order ORD-7890
#   python chapter-02/02-react/main.py
#
#   # Status of a specific order (triggers search_orders, then answer)
#   python chapter-02/02-react/main.py "What's the status of order ORD-7890?"
#
#   # List my orders
#   python chapter-02/02-react/main.py "List my orders"
#
#   # Show this help
#   python chapter-02/02-react/main.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python chapter-02/02-react/main.py [ \"your message\" ]")
        print()
        print("Examples:")
        print('  python chapter-02/02-react/main.py                                    # "What\'s the status of order ORD-7890?"')
        print('  python chapter-02/02-react/main.py "What\'s the status of order ORD-7890?"')
        print('  python chapter-02/02-react/main.py "List my orders"')
        sys.exit(0)
    query = "What's the status of order ORD-7890?" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    print("User:", query)
    print("Agent:", run(query))
