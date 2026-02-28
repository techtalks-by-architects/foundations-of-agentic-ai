"""
LangGraph: Acme Support Router  (Chapter 5, Pattern 5)

The same router pattern from example 02, expressed as a LangGraph state graph
with a conditional edge for routing.

                        ┌→ order_status ──→ END
    START → classify ──→├→ cancellation ──→ END
                        ├→ billing ───────→ END
                        └→ general ───────→ END

What LangGraph gives you:
  - The routing logic is a named function (route) attached to an explicit edge.
  - You can visualize the graph, test `route` independently,
    and add new branches without changing existing nodes.

Run from repo root:
    python chapter-05/05-langgraph-router/main.py
    python chapter-05/05-langgraph-router/main.py "Where is my order ORD-42?"
    python chapter-05/05-langgraph-router/main.py "I want to cancel order ORD-43"
    python chapter-05/05-langgraph-router/main.py --help
"""
import json
import os
import sys
from typing import TypedDict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from langgraph.graph import StateGraph, START, END

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
# State: shared across all nodes
# =============================================================================

class SupportState(TypedDict):
    message: str        # user's input message
    category: str       # set by classify node
    response: str       # set by handler node


# =============================================================================
# Node: classify — single LLM call to determine category
# =============================================================================

CATEGORIES = ["order_status", "cancellation", "billing", "general"]

def classify(state: SupportState) -> dict:
    """Classify the customer message. One cheap LLM call, no tools."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Classify the customer message into exactly one category: "
                    f"{', '.join(CATEGORIES)}. "
                    f"Reply with ONLY the category name, nothing else."
                ),
            },
            {"role": "user", "content": state["message"]},
        ],
    )
    category = response.choices[0].message.content.strip().lower()
    # Ensure we got a valid category; default to general
    if category not in CATEGORIES:
        category = "general"
    print(f"  [classify] category = {category}")
    return {"category": category}


# =============================================================================
# Handler nodes — one per category, each with its own LLM prompt / tools
# =============================================================================

def handle_order_status(state: SupportState) -> dict:
    """Where is my order?"""
    print("  [routed → order_status]")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an Acme order status agent. Be concise."},
            {"role": "user", "content": f"{state['message']}\n\nOrders: {json.dumps(list(ORDERS_DB.values()))}"},
        ],
    )
    return {"response": response.choices[0].message.content or ""}


def handle_cancellation(state: SupportState) -> dict:
    """Cancel my order."""
    print("  [routed → cancellation]")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an Acme cancellation agent. Only pending orders can be cancelled. Be concise."},
            {"role": "user", "content": f"{state['message']}\n\nOrders: {json.dumps(list(ORDERS_DB.values()))}"},
        ],
    )
    return {"response": response.choices[0].message.content or ""}


def handle_billing(state: SupportState) -> dict:
    """Refund / invoice questions."""
    print("  [routed → billing]")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an Acme billing agent. Help with invoices and refunds. Be concise."},
            {"role": "user", "content": state["message"]},
        ],
    )
    return {"response": response.choices[0].message.content or ""}


def handle_general(state: SupportState) -> dict:
    """Catch-all."""
    print("  [routed → general]")
    return {"response": "I can help with order status, cancellations, and billing. Could you tell me more?"}


# =============================================================================
# Conditional edge: route based on category
# =============================================================================

def route(state: SupportState) -> str:
    """Return the node name to route to based on the classified category."""
    return state["category"]


# =============================================================================
# Build and compile the graph
# =============================================================================

def build_graph():
    graph = StateGraph(SupportState)

    # Nodes
    graph.add_node("classify", classify)
    graph.add_node("order_status", handle_order_status)
    graph.add_node("cancellation", handle_cancellation)
    graph.add_node("billing", handle_billing)
    graph.add_node("general", handle_general)

    # Edges
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route)
    graph.add_edge("order_status", END)
    graph.add_edge("cancellation", END)
    graph.add_edge("billing", END)
    graph.add_edge("general", END)

    return graph.compile()


# =============================================================================
# Example usages
# =============================================================================
#
#   python chapter-05/05-langgraph-router/main.py                                     # default: order status
#   python chapter-05/05-langgraph-router/main.py "Where is my order ORD-42?"         # → order_status
#   python chapter-05/05-langgraph-router/main.py "I want to cancel order ORD-43"     # → cancellation
#   python chapter-05/05-langgraph-router/main.py "I need a refund for my last order" # → billing
#   python chapter-05/05-langgraph-router/main.py "What are your store hours?"        # → general
#   python chapter-05/05-langgraph-router/main.py --help
#
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-05/05-langgraph-router/main.py ["your message"]')
        print()
        print("LangGraph router: classify → dispatch to specialist.")
        print("Run from the repo root.")
        print(f"Categories: {', '.join(CATEGORIES)}")
        print()
        print("Examples:")
        print('  python chapter-05/05-langgraph-router/main.py "Where is my order ORD-42?"')
        print('  python chapter-05/05-langgraph-router/main.py "I want to cancel order ORD-43"')
        print('  python chapter-05/05-langgraph-router/main.py "I need a refund"')
        sys.exit(0)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Where is my order ORD-42?"
    print("User:", query)

    app = build_graph()
    result = app.invoke({"message": query})
    print("Agent:", result["response"])
