"""
LangGraph: Order Cancellation Pipeline  (Chapter 5, Pattern 4)

The same sequential pipeline from example 01, expressed as a LangGraph
state graph with typed state and conditional edges.

    ┌─────────┐     ┌──────────────┐     ┌────────┐
    │  lookup  │ ──▶ │ check_policy │ ──▶ │ cancel │ ──▶ END
    └─────────┘     └──────────────┘  │  └────────┘
                                      │  ┌────────┐
                                      └▶ │ reject │ ──▶ END
                                         └────────┘

What LangGraph gives you:
  - Typed state (CancelState) shared across all nodes
  - Conditional edges (route_after_policy) — explicit and testable
  - Checkpointing support (add MemorySaver to inspect/resume at any node)
  - Graph visualization out of the box

What it costs:
  - A dependency (langgraph, langchain-openai)
  - A new mental model (graph nodes, state reducers, edges)

Run from repo root:
    python chapter-05/04-langgraph-pipeline/main.py
    python chapter-05/04-langgraph-pipeline/main.py ORD-42
    python chapter-05/04-langgraph-pipeline/main.py ORD-43
    python chapter-05/04-langgraph-pipeline/main.py --help
"""
import sys
from typing import TypedDict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langgraph.graph import StateGraph, START, END


# =============================================================================
# Simulated order database (same as example 01)
# =============================================================================

ORDERS_DB = {
    "ORD-42": {"order_id": "ORD-42", "status": "pending",   "shipped": False, "total": 129.99},
    "ORD-43": {"order_id": "ORD-43", "status": "shipped",   "shipped": True,  "total": 49.99},
    "ORD-44": {"order_id": "ORD-44", "status": "delivered",  "shipped": True,  "total": 89.00},
}


# =============================================================================
# State: typed dictionary shared across all nodes
# Each node reads from state and returns a partial update.
# =============================================================================

class CancelState(TypedDict):
    order_id: str       # input
    status: str         # set by lookup
    shipped: bool       # set by lookup
    total: float        # set by lookup
    cancellable: bool   # set by check_policy
    reason: str         # set by check_policy
    result: str         # set by cancel or reject


# =============================================================================
# Node: lookup — retrieve order details
# =============================================================================

def lookup(state: CancelState) -> dict:
    """Look up the order in the database. Returns partial state update."""
    order = ORDERS_DB.get(state["order_id"])
    if order:
        print(f"  [lookup] Found: {order}")
        return {"status": order["status"], "shipped": order["shipped"], "total": order["total"]}
    print(f"  [lookup] Not found: {state['order_id']}")
    return {"status": "not_found", "shipped": False, "total": 0.0}


# =============================================================================
# Node: check_policy — deterministic business rules
# =============================================================================

def check_policy(state: CancelState) -> dict:
    """Apply cancellation policy. Pure logic, no LLM."""
    if state["status"] == "not_found":
        result = {"cancellable": False, "reason": "Order not found."}
    elif state["status"] == "delivered":
        result = {"cancellable": False, "reason": "Already delivered. Use return process."}
    elif state["shipped"]:
        result = {"cancellable": False, "reason": "Already shipped. Refuse on delivery."}
    else:
        result = {"cancellable": True, "reason": ""}
    print(f"  [policy] cancellable={result['cancellable']}  reason={result.get('reason', '')}")
    return result


# =============================================================================
# Node: cancel — execute the cancellation
# =============================================================================

def cancel(state: CancelState) -> dict:
    """Process the cancellation (in production: call your orders service)."""
    msg = (
        f"Order {state['order_id']} has been cancelled. "
        f"Refund of ${state['total']:.2f} will be initiated within 3-5 business days."
    )
    print(f"  [cancel] {msg}")
    return {"result": msg}


# =============================================================================
# Node: reject — explain why cancellation is not possible
# =============================================================================

def reject(state: CancelState) -> dict:
    """Produce a rejection message."""
    msg = f"Cannot cancel order {state['order_id']}: {state['reason']}"
    print(f"  [reject] {msg}")
    return {"result": msg}


# =============================================================================
# Conditional edge: route based on policy result
# =============================================================================

def route_after_policy(state: CancelState) -> str:
    """Pick the next node based on whether the order is cancellable."""
    return "cancel" if state["cancellable"] else "reject"


# =============================================================================
# Build and compile the graph
# =============================================================================

def build_graph():
    graph = StateGraph(CancelState)

    # Add nodes
    graph.add_node("lookup", lookup)
    graph.add_node("check_policy", check_policy)
    graph.add_node("cancel", cancel)
    graph.add_node("reject", reject)

    # Add edges
    graph.add_edge(START, "lookup")
    graph.add_edge("lookup", "check_policy")
    graph.add_conditional_edges("check_policy", route_after_policy)
    graph.add_edge("cancel", END)
    graph.add_edge("reject", END)

    return graph.compile()


# =============================================================================
# Example usages
# =============================================================================
#
#   python chapter-05/04-langgraph-pipeline/main.py              # defaults to ORD-42 (pending → cancelled)
#   python chapter-05/04-langgraph-pipeline/main.py ORD-42       # pending → cancelled
#   python chapter-05/04-langgraph-pipeline/main.py ORD-43       # shipped → rejected
#   python chapter-05/04-langgraph-pipeline/main.py ORD-44       # delivered → rejected
#   python chapter-05/04-langgraph-pipeline/main.py ORD-99       # not found → rejected
#   python chapter-05/04-langgraph-pipeline/main.py --help
#
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python chapter-05/04-langgraph-pipeline/main.py [ORDER_ID]")
        print()
        print("LangGraph pipeline: lookup → policy check → cancel or reject.")
        print("Run from the repo root.")
        print()
        print("Examples:")
        print("  python chapter-05/04-langgraph-pipeline/main.py ORD-42   # pending → cancelled")
        print("  python chapter-05/04-langgraph-pipeline/main.py ORD-43   # shipped → rejected")
        print("  python chapter-05/04-langgraph-pipeline/main.py ORD-44   # delivered → rejected")
        print("  python chapter-05/04-langgraph-pipeline/main.py ORD-99   # not found")
        sys.exit(0)

    order_id = sys.argv[1] if len(sys.argv) > 1 else "ORD-42"
    print(f"\n--- LangGraph Cancel Pipeline for {order_id} ---")

    app = build_graph()
    result = app.invoke({"order_id": order_id})
    print("\nResult:", result["result"])
