"""
Pattern 1: Tool Use

The agent receives a list of tools (name, description, parameters). During the
control loop, the model can choose to *call* a tool instead of replying to the
user. The *platform* executes the tool and feeds the result back into the loop.
The model never executes code—it only decides which tool to call and with what
arguments.


python chapter-02/01-tool-use/main.py [ "Cancel my pending orders" ]
Requires OPENAI_API_KEY in environment (or uses mock responses).
"""
import json
import sys
from pathlib import Path

# Allow importing shared LLM client from chapter-02/common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import chat, ChatResponse, ToolCall


# -----------------------------------------------------------------------------
# Tools: allow-list of capabilities the agent can invoke
# -----------------------------------------------------------------------------
# Each tool is a plain function. The LLM sees its name and docstring (description)
# and chooses when to call it. The platform runs the function—deterministic
# execution, not the model. This is the governance boundary (mental model #5).

def search_orders(customer_id: str, status: str = "all") -> list[dict]:
    """Search orders for a customer, optionally filtered by status."""
    # In production this would call your orders API
    orders = [
        {"order_id": "ORD-1234", "status": "shipped", "total": 59.99},
        {"order_id": "ORD-1235", "status": "pending", "total": 29.99},
    ]
    if status != "all":
        orders = [o for o in orders if o["status"] == status]
    return orders


def cancel_order(order_id: str, reason: str) -> dict:
    """Cancel an order. Requires a reason. Only works for pending orders."""
    # In production: validate order_id, check status, call API, audit log
    return {"order_id": order_id, "cancelled": True, "reason": reason}


TOOLS = [search_orders, cancel_order]


def execute_tool(name: str, arguments: dict) -> str:
    """
    Execute a tool by name with the given arguments.
    Returns the result as a JSON string so it can be appended to the conversation.
    This is where you would add validation, auth checks, and rate limiting.
    """
    fn = {f.__name__: f for f in TOOLS}.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**arguments)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# -----------------------------------------------------------------------------
# Control loop: observe (messages) → decide (LLM) → act (execute_tool) → repeat
# -----------------------------------------------------------------------------

def run(user_message: str, max_iterations: int = 10) -> str:
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_iterations):
        # 1. Decide: send conversation to LLM; it may return text and/or tool calls
        response = chat(messages=messages, tools=TOOLS)
        # print(f"Response: {response}")
        
        if response.tool_calls:
            # 2. Act: the model chose to call one or more tools
            # Build the assistant message in OpenAI format (required for next API call)
            tool_calls_for_api = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in response.tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": tool_calls_for_api,
            })
            # Append each tool result so the model can "observe" what happened
            for tc in response.tool_calls:
                result = execute_tool(tc.name, tc.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            # 3. Loop: we don't return yet; send updated messages back to the LLM
        else:
            # No tool calls: the model produced a final answer for the user
            return response.content or ""

    # Safety: prevent infinite loops
    return "[Agent did not converge]"


# -----------------------------------------------------------------------------
# Example usages (run from chapter-02)
# -----------------------------------------------------------------------------
#
#   # Default: "Cancel my pending orders"
#   python chapter-02/01-tool-use/main.py
#
#   # List orders (triggers search_orders)
#   python chapter-02/01-tool-use/main.py "List my orders"
#
#   # Cancel pending orders (triggers search_orders then cancel_order)
#   python chapter-02/01-tool-use/main.py "Cancel my pending orders"
#
#   # Custom query
#   python chapter-02/01-tool-use/main.py "What's the status of ORD-1235?"
#
#   # Show this help
#   python chapter-02/01-tool-use/main.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python chapter-02/01-tool-use/main.py [ \"your message\" ]")
        print()
        print("Examples:")
        print('  python chapter-02/01-tool-use/main.py  "Cancel my pending orders"')
        print('  python chapter-02/01-tool-use/main.py "List my orders"')
        print('  python chapter-02/01-tool-use/main.py "What\'s the status of ORD-1235?"')
        sys.exit(0)
    query = "Cancel my pending orders" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    print("User:", query)
    print("Agent:", run(query))
