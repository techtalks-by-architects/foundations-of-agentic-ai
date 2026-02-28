"""
Tool Anatomy: Good Tool Design  (Chapter 6, Example 1)

Demonstrates the six properties of a well-designed tool:
  1. Clear, verb-based name
  2. Descriptive parameter names with descriptions
  3. One-sentence description with constraints
  4. Predictable, structured output (including errors as data)
  5. Idempotent (safe to call twice)
  6. Minimal, focused scope (one tool per action)

The agent uses four well-designed tools to handle order support requests.
Try different queries to see how tool names and descriptions guide the LLM.

Run from repo root:
    python chapter-06/01-tool-anatomy/main.py
    python chapter-06/01-tool-anatomy/main.py "Where is my order ORD-42?"
    python chapter-06/01-tool-anatomy/main.py "Cancel order ORD-43"
    python chapter-06/01-tool-anatomy/main.py "Cancel order ORD-42"
    python chapter-06/01-tool-anatomy/main.py --help
"""
import json
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

client = OpenAI()  # reads OPENAI_BASE_URL and OPENAI_API_KEY from env
MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_ITERATIONS = 10


# =============================================================================
# Simulated order database
# =============================================================================

ORDERS_DB = {
    "ORD-42": {"order_id": "ORD-42", "status": "shipped", "eta": "Feb 16", "total": 129.99, "item": "Wireless Headphones"},
    "ORD-43": {"order_id": "ORD-43", "status": "pending", "eta": "Feb 20", "total": 49.99, "item": "USB-C Cable"},
    "ORD-44": {"order_id": "ORD-44", "status": "delivered", "eta": "Feb 10", "total": 89.00, "item": "Bluetooth Speaker"},
}

# Track cancellations to demonstrate idempotency
_cancelled = set()


# =============================================================================
# Well-designed tools — note how each follows the six properties
# =============================================================================

def get_order_status(order_id: str) -> dict:
    """Look up an order by ID. Returns status, item, total, and estimated delivery date."""
    #
    # Property 1: verb-based name — "get_order_status"
    # Property 2: parameter name is "order_id", not "id" or "x"
    # Property 3: description says WHAT it does and WHAT it returns
    # Property 4: structured output (dict with consistent fields)
    # Property 6: focused scope — only reads, never modifies
    #
    order = ORDERS_DB.get(order_id)
    if not order:
        # Property 4: errors as data, not exceptions
        return {"error": "order_not_found", "message": f"No order with ID {order_id}. Check the ID and try again."}
    return order


def cancel_order(order_id: str, reason: str) -> dict:
    """Cancel a pending order by ID. Only works for orders that have NOT shipped. Requires a reason."""
    #
    # Property 1: "cancel_order" — clear verb + noun
    # Property 3: description includes CONSTRAINTS ("only works for orders that have NOT shipped")
    # Property 5: idempotent — calling twice returns the same result
    # Property 6: focused — only cancels, doesn't look up or refund
    #

    # --- Input validation (Ch.5: never trust LLM arguments) ---
    if not order_id.startswith("ORD-"):
        return {"error": "invalid_format", "message": "Order ID must start with ORD-. Example: ORD-42"}
    if not reason or len(reason.strip()) == 0:
        return {"error": "missing_reason", "message": "A cancellation reason is required."}
    if len(reason) > 500:
        return {"error": "reason_too_long", "message": "Reason must be under 500 characters."}

    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": "order_not_found", "message": f"No order with ID {order_id}."}

    if order["status"] == "delivered":
        return {"error": "already_delivered", "message": f"Order {order_id} is already delivered. Use the return process instead."}
    if order["status"] == "shipped":
        return {"error": "already_shipped", "message": f"Order {order_id} has already shipped. It can be refused on delivery."}

    # --- Idempotency: if already cancelled, return same result ---
    if order_id in _cancelled:
        return {"order_id": order_id, "cancelled": True, "refund_amount": order["total"],
                "note": "This order was already cancelled. No additional action taken."}

    _cancelled.add(order_id)
    return {"order_id": order_id, "cancelled": True, "refund_amount": order["total"],
            "refund_eta": "3-5 business days"}


def search_orders(customer_id: str, status: str = "") -> dict:
    """Search orders for a customer. Optionally filter by status (pending, shipped, delivered)."""
    #
    # Property 1: "search_orders" — clear action
    # Property 2: "status" has a description listing valid values
    # Property 6: read-only — searching, not modifying
    #
    results = list(ORDERS_DB.values())
    if status:
        results = [o for o in results if o["status"] == status.lower()]
    return {"customer_id": customer_id, "orders": results, "count": len(results)}


def request_refund(order_id: str, reason: str) -> dict:
    """Request a refund for a delivered order. Only works for orders with status 'delivered'."""
    #
    # Separate from cancel_order — different action, different preconditions
    # Property 6: focused scope — refund ≠ cancel
    #
    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": "order_not_found", "message": f"No order with ID {order_id}."}
    if order["status"] != "delivered":
        return {"error": "not_delivered", "message": f"Refunds are only for delivered orders. Order {order_id} status: {order['status']}."}
    return {"order_id": order_id, "refund_initiated": True, "refund_amount": order["total"],
            "refund_eta": "5-7 business days"}


# =============================================================================
# Tool schema — what the LLM reads to decide which tool to call
# Note how descriptions include constraints and examples.
# =============================================================================

TOOLS = [get_order_status, cancel_order, search_orders, request_refund]

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up an order by ID. Returns status, item, total, and estimated delivery date.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "Acme order ID, e.g. ORD-42"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel a pending order by ID. Only works for orders that have NOT shipped. Requires a reason.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Acme order ID, e.g. ORD-42"},
                    "reason": {"type": "string", "description": "Why the customer is cancelling"},
                },
                "required": ["order_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_orders",
            "description": "Search orders for a customer. Optionally filter by status (pending, shipped, delivered).",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID, e.g. CUST-1"},
                    "status": {"type": "string", "description": "Filter by status: pending, shipped, or delivered. Leave empty for all."},
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_refund",
            "description": "Request a refund for a delivered order. Only works for orders with status 'delivered'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Acme order ID, e.g. ORD-42"},
                    "reason": {"type": "string", "description": "Why the customer wants a refund"},
                },
                "required": ["order_id", "reason"],
            },
        },
    },
]

TOOL_MAP = {fn.__name__: fn for fn in TOOLS}


# =============================================================================
# Agent loop
# =============================================================================

SYSTEM_PROMPT = """You are an Acme Corp customer support agent.

Help the user with order inquiries, cancellations, and refunds.
Use the tools provided. Only use data from tools — never make up order info.
If a tool returns an error, explain it to the user in a friendly way."""


def run(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    for i in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                fn = TOOL_MAP.get(tc.function.name)
                if fn is None:
                    result = {"error": "unknown_tool", "message": f"Tool '{tc.function.name}' does not exist."}
                else:
                    args = json.loads(tc.function.arguments)
                    result = fn(**args)
                print(f"  [{i}] {tc.function.name}({tc.function.arguments}) -> {json.dumps(result)}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        else:
            return msg.content or ""
    return "Agent did not converge. Please try again."


# =============================================================================
# Example usages
# =============================================================================
#
#   python chapter-06/01-tool-anatomy/main.py                                      # default query
#   python chapter-06/01-tool-anatomy/main.py "Where is my order ORD-42?"          # → get_order_status
#   python chapter-06/01-tool-anatomy/main.py "Cancel order ORD-43"                # → cancel_order (pending, succeeds)
#   python chapter-06/01-tool-anatomy/main.py "Cancel order ORD-42"                # → cancel_order (shipped, error as data)
#   python chapter-06/01-tool-anatomy/main.py "Show me all my orders"              # → search_orders
#   python chapter-06/01-tool-anatomy/main.py "I want a refund for ORD-44"         # → request_refund (delivered)
#   python chapter-06/01-tool-anatomy/main.py "I want a refund for ORD-43"         # → request_refund (pending, error)
#   python chapter-06/01-tool-anatomy/main.py --help
#
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-06/01-tool-anatomy/main.py ["your message"]')
        print()
        print("Demonstrates well-designed tools with an agent.")
        print("Four tools: get_order_status, cancel_order, search_orders, request_refund")
        print()
        print("Examples:")
        print('  python chapter-06/01-tool-anatomy/main.py "Where is my order ORD-42?"')
        print('  python chapter-06/01-tool-anatomy/main.py "Cancel order ORD-43"')
        print('  python chapter-06/01-tool-anatomy/main.py "Cancel order ORD-42"        # shipped → error as data')
        print('  python chapter-06/01-tool-anatomy/main.py "I want a refund for ORD-44"')
        sys.exit(0)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Where is my order ORD-42?"
    print("User:", query)
    print("Agent:", run(query))
