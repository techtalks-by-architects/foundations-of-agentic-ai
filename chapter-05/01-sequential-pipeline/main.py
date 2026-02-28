"""
Sequential Pipeline: Order Cancellation  (Chapter 5, Pattern 1)

Three agents chained in sequence — each one's output feeds the next:

    Input (order_id)
      │
      ▼
    lookup_agent    ── look up order details
      │
      ▼
    policy_agent    ── check cancellation policy (deterministic)
      │
      ▼
    cancellation_agent ── cancel or reject
      │
      ▼
    Result (message to user)

Architecture notes:
  - Each agent is a plain function with a clear contract (input → output).
  - The pipeline is linear — no branches, no loops.
  - Control flow lives in YOUR code, not in the LLM.
  - You can test, retry, or timeout each step independently.

Run from repo root:
    python chapter-05/01-sequential-pipeline/main.py
    python chapter-05/01-sequential-pipeline/main.py ORD-42
    python chapter-05/01-sequential-pipeline/main.py ORD-99
    python chapter-05/01-sequential-pipeline/main.py --help
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


# =============================================================================
# Simulated order database
# In production: call your orders microservice
# =============================================================================

ORDERS_DB = {
    "ORD-42": {"order_id": "ORD-42", "status": "pending",   "shipped": False, "total": 129.99},
    "ORD-43": {"order_id": "ORD-43", "status": "shipped",   "shipped": True,  "total": 49.99},
    "ORD-44": {"order_id": "ORD-44", "status": "delivered",  "shipped": True,  "total": 89.00},
}


# =============================================================================
# Agent 1 — Lookup: retrieve order details
# =============================================================================

def lookup_agent(order_id: str) -> dict:
    """
    Look up an order by ID.
    Returns the order dict, or a 'not found' dict.
    This agent does NOT use the LLM — it's a deterministic service call.
    """
    order = ORDERS_DB.get(order_id)
    if order:
        print(f"  [lookup] Found order: {json.dumps(order)}")
        return order
    print(f"  [lookup] Order {order_id} not found")
    return {"order_id": order_id, "status": "not_found", "shipped": False, "total": 0}


# =============================================================================
# Agent 2 — Policy: check cancellation rules (deterministic)
# =============================================================================

def policy_agent(order: dict) -> dict:
    """
    Apply cancellation policy.  Pure business logic — no LLM needed.

    Rules:
      - delivered  → cannot cancel (use return process)
      - shipped    → cannot cancel (refuse on delivery)
      - not_found  → cannot cancel
      - otherwise  → cancellable
    """
    status = order.get("status", "")

    if status == "not_found":
        order["cancellable"] = False
        order["reason"] = "Order not found."
    elif status == "delivered":
        order["cancellable"] = False
        order["reason"] = "Order already delivered. Please use the return process instead."
    elif order.get("shipped"):
        order["cancellable"] = False
        order["reason"] = "Order already shipped. It can be refused on delivery."
    else:
        order["cancellable"] = True
        order["reason"] = ""

    print(f"  [policy] cancellable={order['cancellable']}  reason={order.get('reason', '')}")
    return order


# =============================================================================
# Agent 3 — Cancellation: execute or explain
# Uses the LLM to produce a friendly, natural-language response.
# =============================================================================

def cancellation_agent(order: dict) -> str:
    """
    If cancellable: cancel the order and tell the user.
    If not: explain why in a helpful tone (via LLM).
    """
    if order["cancellable"]:
        # In production: call your orders service to actually cancel
        print(f"  [cancel] Processing cancellation for {order['order_id']}")
        return (
            f"Order {order['order_id']} has been cancelled. "
            f"A refund of ${order['total']:.2f} will be initiated within 3-5 business days."
        )

    # Use the LLM for a polite, context-aware rejection message
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly Acme Corp support agent. "
                    "The customer wanted to cancel an order but it's not possible. "
                    "Explain why politely and suggest next steps. Be concise (2-3 sentences)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Order: {order['order_id']}, Status: {order['status']}, "
                    f"Reason cannot cancel: {order['reason']}"
                ),
            },
        ],
    )
    return response.choices[0].message.content or order["reason"]


# =============================================================================
# Pipeline: the three agents chained together
# =============================================================================

def cancel_pipeline(order_id: str) -> str:
    """
    Sequential pipeline: lookup → policy → cancellation.
    Each step's output is the next step's input.
    """
    print(f"\n--- Cancel Pipeline for {order_id} ---")
    order = lookup_agent(order_id)
    checked = policy_agent(order)
    result = cancellation_agent(checked)
    return result


# =============================================================================
# Example usages
# =============================================================================
#
#   python chapter-05/01-sequential-pipeline/main.py              # defaults to ORD-42 (pending → cancellable)
#   python chapter-05/01-sequential-pipeline/main.py ORD-42       # pending order → will be cancelled
#   python chapter-05/01-sequential-pipeline/main.py ORD-43       # shipped order → rejection with explanation
#   python chapter-05/01-sequential-pipeline/main.py ORD-44       # delivered order → rejection with explanation
#   python chapter-05/01-sequential-pipeline/main.py ORD-99       # not found
#   python chapter-05/01-sequential-pipeline/main.py --help
#
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python chapter-05/01-sequential-pipeline/main.py [ORDER_ID]")
        print()
        print("Sequential pipeline: lookup → policy check → cancel or reject.")
        print("Run from the repo root.")
        print()
        print("Examples:")
        print("  python chapter-05/01-sequential-pipeline/main.py ORD-42   # pending → cancelled")
        print("  python chapter-05/01-sequential-pipeline/main.py ORD-43   # shipped → rejected")
        print("  python chapter-05/01-sequential-pipeline/main.py ORD-44   # delivered → rejected")
        print("  python chapter-05/01-sequential-pipeline/main.py ORD-99   # not found")
        sys.exit(0)

    order_id = sys.argv[1] if len(sys.argv) > 1 else "ORD-42"
    print("Result:", cancel_pipeline(order_id))
