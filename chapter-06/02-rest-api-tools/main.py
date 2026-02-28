"""
REST API Tools: Agent calling a REST API  (Chapter 6, Example 2 — agent)

Demonstrates wrapping REST API endpoints as agent tools. The agent calls
a real HTTP server (server.py) instead of in-process functions.

Prerequisites:
    1. Start the server:  python server.py   (runs on localhost:5001)
    2. Run the agent:     python main.py "Cancel order ORD-43"

Architecture notes:
  - Each tool is a thin wrapper: validate args → HTTP call → return JSON.
  - Errors from the API are returned as data (structured JSON), not exceptions.
  - The agent doesn't know (or care) that tools are HTTP calls — same interface.

Run from repo root:
    python chapter-06/02-rest-api-tools/main.py
    python chapter-06/02-rest-api-tools/main.py "Where is my order ORD-42?"
    python chapter-06/02-rest-api-tools/main.py "Cancel order ORD-43 because I found a better price"
    python chapter-06/02-rest-api-tools/main.py --help
    python chapter-06/02-rest-api-tools/main.py --help
"""
import json
import os
import sys

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

client = OpenAI()  # reads OPENAI_BASE_URL and OPENAI_API_KEY from env
MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_ITERATIONS = 10
API_BASE = "http://localhost:5001"


# =============================================================================
# Tool wrappers: thin functions that call the REST API
# Each follows the same pattern: validate → call → return structured JSON
# =============================================================================

def get_order_status(order_id: str) -> dict:
    """Look up an order by ID. Returns status, item, total, and delivery date."""
    try:
        resp = requests.get(f"{API_BASE}/orders/{order_id}", timeout=5)
        return resp.json()
    except requests.ConnectionError:
        return {"error": "service_unavailable", "message": "Orders API is not reachable. Is server.py running?"}
    except Exception as e:
        return {"error": "request_failed", "message": str(e)}


def cancel_order(order_id: str, reason: str) -> dict:
    """Cancel a pending order by ID. Only works for orders that have NOT shipped."""
    try:
        resp = requests.post(
            f"{API_BASE}/orders/{order_id}/cancel",
            json={"reason": reason},
            timeout=5,
        )
        return resp.json()
    except requests.ConnectionError:
        return {"error": "service_unavailable", "message": "Orders API is not reachable. Is server.py running?"}
    except Exception as e:
        return {"error": "request_failed", "message": str(e)}


def search_orders(status: str = "") -> dict:
    """List all orders. Optionally filter by status (pending, shipped, delivered)."""
    try:
        params = {"status": status} if status else {}
        resp = requests.get(f"{API_BASE}/orders", params=params, timeout=5)
        return resp.json()
    except requests.ConnectionError:
        return {"error": "service_unavailable", "message": "Orders API is not reachable. Is server.py running?"}
    except Exception as e:
        return {"error": "request_failed", "message": str(e)}


TOOL_MAP = {
    "get_order_status": get_order_status,
    "cancel_order": cancel_order,
    "search_orders": search_orders,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up an order by ID. Returns status, item, total, and delivery date.",
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
            "description": "List all orders. Optionally filter by status: pending, shipped, or delivered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Filter by status (pending/shipped/delivered). Omit for all."},
                },
                "required": [],
            },
        },
    },
]


# =============================================================================
# Agent loop (same structure as always — tools are HTTP calls now)
# =============================================================================

SYSTEM_PROMPT = """You are an Acme Corp support agent.
Help with order status, cancellations, and listing orders.
Use tools to get data — never make up order information."""


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
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = fn(**args) if fn else {"error": "unknown_tool"}
                print(f"  [{i}] {tc.function.name}({tc.function.arguments}) -> {json.dumps(result)}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        else:
            return msg.content or ""
    return "Agent did not converge."


# =============================================================================
# Example usages
# =============================================================================
#
#   # Terminal 1: start the API server
#   python server.py
#
#   # Terminal 2: run the agent
#   python chapter-06/02-rest-api-tools/main.py "Where is my order ORD-42?"
#   python chapter-06/02-rest-api-tools/main.py "Cancel order ORD-43 because I found a better price"
#   python chapter-06/02-rest-api-tools/main.py "Show me all pending orders"
#   python chapter-06/02-rest-api-tools/main.py "Cancel order ORD-42"    # shipped → structured error from API
#   python chapter-06/02-rest-api-tools/main.py --help
#
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-06/02-rest-api-tools/main.py ["your message"]')
        print()
        print("Agent that calls a REST API via tool wrappers.")
        print("Requires server.py running on localhost:5001.")
        print()
        print("Start server:  python server.py")
        print()
        print("Examples:")
        print('  python chapter-06/02-rest-api-tools/main.py "Where is my order ORD-42?"')
        print('  python chapter-06/02-rest-api-tools/main.py "Cancel order ORD-43 because I found a better price"')
        print('  python chapter-06/02-rest-api-tools/main.py "Show me all pending orders"')
        sys.exit(0)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Where is my order ORD-42?"
    print("User:", query)
    print("Agent:", run(query))
