"""
Router: Acme Support Desk  (Chapter 5, Pattern 2)

A classifier agent examines the incoming message and dispatches it to the
right specialist agent. Each specialist has its own system prompt and tools.

                    ┌→ order_status_agent
    Input → Router ──→ cancellation_agent
                    ├→ billing_agent
                    └→ general_agent

Architecture notes:
  - The router is a single, cheap LLM call (classify only).
  - Each specialist has a small, focused tool set.
  - Control flow is a dict lookup: explicit, testable, easy to extend.
  - Adding a new category = add a specialist function + one dict entry.

Run from repo root:
    python chapter-05/02-router/main.py
    python chapter-05/02-router/main.py "Where is my order ORD-42?"
    python chapter-05/02-router/main.py "I want to cancel my order"
    python chapter-05/02-router/main.py "I need a refund"
    python chapter-05/02-router/main.py --help
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
# =============================================================================

ORDERS_DB = {
    "ORD-42": {"order_id": "ORD-42", "status": "shipped", "eta": "Feb 16", "total": 129.99},
    "ORD-43": {"order_id": "ORD-43", "status": "pending", "eta": "Feb 20", "total": 49.99},
}


# =============================================================================
# Specialist agents — each with its own system prompt and capabilities
# =============================================================================

def order_status_agent(message: str) -> str:
    """
    Handles: 'Where is my order?' questions.
    Tools: get_order_status (simulated here as a DB lookup).
    """
    print("  [routed → order_status_agent]")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Acme Corp order status agent. "
                    "Use the order data provided to answer the customer. "
                    "Be concise and helpful."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Customer asks: {message}\n\n"
                    f"Available orders: {json.dumps(list(ORDERS_DB.values()))}"
                ),
            },
        ],
    )
    return response.choices[0].message.content or ""


def cancellation_agent(message: str) -> str:
    """
    Handles: cancellation requests.
    Tools: get_order_status, cancel_order.
    """
    print("  [routed → cancellation_agent]")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Acme Corp cancellation agent. "
                    "Check if the order can be cancelled (only pending orders can). "
                    "If it can, confirm the cancellation. If not, explain why."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Customer asks: {message}\n\n"
                    f"Available orders: {json.dumps(list(ORDERS_DB.values()))}"
                ),
            },
        ],
    )
    return response.choices[0].message.content or ""


def billing_agent(message: str) -> str:
    """
    Handles: billing, invoice, and refund questions.
    Tools: get_invoice, request_refund.
    """
    print("  [routed → billing_agent]")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Acme Corp billing agent. "
                    "Help with invoices, payment questions, and refund requests. "
                    "Be concise."
                ),
            },
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content or ""


def general_agent(message: str) -> str:
    """
    Catch-all for anything that doesn't fit the other categories.
    No special tools — just general Acme Corp knowledge.
    """
    print("  [routed → general_agent]")
    return (
        "I can help with order status, cancellations, and billing. "
        "Could you tell me more about what you need?"
    )


# =============================================================================
# Category registry
# Adding a new category = add a function above + one entry here.
# =============================================================================

CATEGORIES = {
    "order_status": order_status_agent,
    "cancellation": cancellation_agent,
    "billing": billing_agent,
    "general": general_agent,
}


# =============================================================================
# Router: classify → dispatch
# =============================================================================

def classify(message: str) -> str:
    """
    Use the LLM to classify the customer message into one of the categories.
    This is a single, cheap LLM call — no tools, no loop.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Classify the customer message into exactly one category: "
                    f"{', '.join(CATEGORIES.keys())}. "
                    f"Reply with ONLY the category name, nothing else."
                ),
            },
            {"role": "user", "content": message},
        ],
    )
    category = response.choices[0].message.content.strip().lower()
    print(f"  [classify] category = {category}")
    return category


def route(message: str) -> str:
    """
    Full router: classify the message, then dispatch to the right specialist.
    """
    category = classify(message)
    handler = CATEGORIES.get(category, general_agent)
    return handler(message)


# =============================================================================
# Example usages
# =============================================================================
#
#   python chapter-05/02-router/main.py                                    # default: order status
#   python chapter-05/02-router/main.py "Where is my order ORD-42?"        # → order_status_agent
#   python chapter-05/02-router/main.py "I want to cancel order ORD-43"    # → cancellation_agent
#   python chapter-05/02-router/main.py "I need a refund for my last order" # → billing_agent
#   python chapter-05/02-router/main.py "What are your store hours?"       # → general_agent
#   python chapter-05/02-router/main.py --help
#
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-05/02-router/main.py ["your message"]')
        print()
        print("Router: classifies your message and dispatches to the right agent.")
        print("Run from the repo root.")
        print(f"Categories: {', '.join(CATEGORIES.keys())}")
        print()
        print("Examples:")
        print('  python chapter-05/02-router/main.py "Where is my order ORD-42?"')
        print('  python chapter-05/02-router/main.py "I want to cancel order ORD-43"')
        print('  python chapter-05/02-router/main.py "I need a refund"')
        print('  python chapter-05/02-router/main.py "What are your store hours?"')
        sys.exit(0)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Where is my order ORD-42?"
    print("User:", query)
    print("Agent:", route(query))
