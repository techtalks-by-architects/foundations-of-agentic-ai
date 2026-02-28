"""
Pattern 9: Human in the Loop

Sensitive tool calls (e.g. refund, delete) are proposed by the agent but
executed only after human approval. The human is a gate in the control flow:
the platform intercepts certain tools, presents the proposal to a human, and
runs the tool only if approved. Rejection is fed back to the agent so it can
adjust.

Run from repo root:
  python chapter-02/09-human-in-the-loop/main.py [ "I need a refund for order ORD-1234" ]
Requires OPENAI_API_KEY in environment (or uses mock responses).
Demo: approval is a console prompt (y/n/edit); in production use a queue/UI.
"""
import json
import sys
import uuid
from pathlib import Path

# Allow importing shared LLM client from chapter-02/common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import chat, ToolCall


# -----------------------------------------------------------------------------
# Agent prompt: knows that refunds require human approval
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a support agent. You can look up orders and request refunds.
Refunds require human approval—you propose them, and a human will approve or reject.
Use search_orders first to get order details, then request_refund if the user wants a refund."""

# Which tools must be gated: allow-list (could also be metadata on the tool)
REQUIRES_APPROVAL = {"request_refund"}


# -----------------------------------------------------------------------------
# Tools: search_orders runs immediately; request_refund goes through HITL
# -----------------------------------------------------------------------------

def search_orders(customer_id: str, status: str = "all") -> list[dict]:
    """Search orders for a customer."""
    return [
        {"order_id": "ORD-1234", "status": "shipped", "total": 59.99},
        {"order_id": "ORD-1235", "status": "delivered", "total": 29.99},
    ]


def request_refund(order_id: str, amount: float, reason: str) -> dict:
    """Request a refund. Requires human approval before execution."""
    return {"order_id": order_id, "refunded": True, "amount": amount, "reason": reason}


TOOLS = [search_orders, request_refund]


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool with no approval (used after approval or for non-gated tools)."""
    fn = {f.__name__: f for f in TOOLS}.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return json.dumps(fn(**arguments))
    except Exception as e:
        return json.dumps({"error": str(e)})


# -----------------------------------------------------------------------------
# Human gate: demo uses console input; production would use queue + UI + webhook
# -----------------------------------------------------------------------------

def wait_for_human_approval(proposal: dict) -> dict:
    """Block until human approves or rejects. Returns {"approved": bool, "reason": str?}."""
    print(f"\n  [HITL] Proposal: {proposal['tool']}({proposal['arguments']})")
    answer = input("  Approve? (y/n/edit): ").strip().lower()
    if answer == "y":
        return {"approved": True}
    if answer == "edit":
        reason = input("  Rejection reason (for agent): ").strip()
        return {"approved": False, "reason": reason or "Rejected by human"}
    return {"approved": False, "reason": "Rejected by human"}


def execute_tool_with_hitl(name: str, arguments: dict) -> str:
    """If tool is in REQUIRES_APPROVAL, wait for human; else execute immediately."""
    if name not in REQUIRES_APPROVAL:
        return execute_tool(name, arguments)
    proposal = {"tool": name, "arguments": arguments, "request_id": str(uuid.uuid4())}
    approval = wait_for_human_approval(proposal)
    if approval["approved"]:
        return execute_tool(name, arguments)
    return json.dumps({"error": "Rejected by human", "reason": approval.get("reason", "")})


# -----------------------------------------------------------------------------
# Agent loop: same as Tool Use / ReAct, but tool execution goes through HITL
# -----------------------------------------------------------------------------

def run(user_message: str, max_iterations: int = 10) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    for i in range(max_iterations):
        response = chat(messages=messages, tools=TOOLS)
        if response.tool_calls:
            tool_calls_api = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in response.tool_calls
            ]
            messages.append({"role": "assistant", "content": response.content or None, "tool_calls": tool_calls_api})
            for tc in response.tool_calls:
                result = execute_tool_with_hitl(tc.name, tc.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            return response.content or ""
    return "[Agent did not converge]"


# -----------------------------------------------------------------------------
# Example usages (run from chapter-02)
# -----------------------------------------------------------------------------
#
#   # Default: refund request (will prompt for approval in console)
#   python chapter-02/09-human-in-the-loop/main.py
#
#   # Refund for order -> may call search_orders then request_refund (HITL)
#   python chapter-02/09-human-in-the-loop/main.py "I need a refund for order ORD-1234"
#
#   # Show this help
#   python chapter-02/09-human-in-the-loop/main.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python chapter-02/09-human-in-the-loop/main.py [ \"your message\" ]")
        print()
        print("Examples:")
        print('  python chapter-02/09-human-in-the-loop/main.py "I need a refund for order ORD-1234"')
        sys.exit(0)
    query = "I need a refund for order ORD-1234" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    print("User:", query)
    print("Agent:", run(query))
