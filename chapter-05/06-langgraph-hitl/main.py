"""
LangGraph: Human-in-the-Loop Refund  (Chapter 5, Pattern 6)

A refund workflow that pauses for human approval before processing payment.
This is where LangGraph's value becomes clear: pause/resume with persistent
state is significant engineering to build from scratch.

    START → lookup → check_eligibility ──→ prepare_refund → [INTERRUPT] → process_refund → END
                                       └→ reject ──────────────────────────────────────→ END

Flow:
  1. Look up the order
  2. Check refund eligibility (policy rules)
  3. If eligible: prepare the refund details, then PAUSE for human approval
  4. Human reviews and approves/rejects
  5. If approved: process the refund. If not: reject.

What LangGraph gives you:
  - interrupt_before on "process_refund" pauses the graph
  - MemorySaver persists state across the pause
  - update_state lets the human inject approval
  - invoke(None, config) resumes from the checkpoint

Run from repo root:
    python chapter-05/06-langgraph-hitl/main.py
    python chapter-05/06-langgraph-hitl/main.py ORD-42
    python chapter-05/06-langgraph-hitl/main.py ORD-44
    python chapter-05/06-langgraph-hitl/main.py --help
"""
import sys
from typing import TypedDict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# =============================================================================
# Simulated order database
# =============================================================================

ORDERS_DB = {
    "ORD-42": {"order_id": "ORD-42", "status": "delivered", "total": 129.99, "item": "Wireless Headphones", "days_since_delivery": 5},
    "ORD-43": {"order_id": "ORD-43", "status": "shipped",   "total": 49.99,  "item": "USB-C Cable",         "days_since_delivery": 0},
    "ORD-44": {"order_id": "ORD-44", "status": "delivered", "total": 89.00,  "item": "Bluetooth Speaker",   "days_since_delivery": 45},
}

REFUND_WINDOW_DAYS = 30  # Policy: refunds within 30 days of delivery


# =============================================================================
# State: flows through the graph
# =============================================================================

class RefundState(TypedDict):
    order_id: str
    status: str
    item: str
    total: float
    days_since_delivery: int
    eligible: bool
    reason: str
    approved: bool
    result: str


# =============================================================================
# Node: lookup — retrieve order details
# =============================================================================

def lookup(state: RefundState) -> dict:
    order = ORDERS_DB.get(state["order_id"])
    if order:
        print(f"  [lookup] Found: {order['order_id']} — {order['item']} — ${order['total']}")
        return {
            "status": order["status"],
            "item": order["item"],
            "total": order["total"],
            "days_since_delivery": order["days_since_delivery"],
        }
    print(f"  [lookup] Not found: {state['order_id']}")
    return {"status": "not_found", "item": "", "total": 0.0, "days_since_delivery": 0}


# =============================================================================
# Node: check_eligibility — deterministic policy rules
# =============================================================================

def check_eligibility(state: RefundState) -> dict:
    if state["status"] == "not_found":
        result = {"eligible": False, "reason": "Order not found."}
    elif state["status"] != "delivered":
        result = {"eligible": False, "reason": f"Order status is '{state['status']}'. Refunds only for delivered orders."}
    elif state["days_since_delivery"] > REFUND_WINDOW_DAYS:
        result = {"eligible": False, "reason": f"Refund window expired ({state['days_since_delivery']} days > {REFUND_WINDOW_DAYS}-day policy)."}
    else:
        result = {"eligible": True, "reason": ""}
    print(f"  [eligibility] eligible={result['eligible']}  reason={result.get('reason', '')}")
    return result


# =============================================================================
# Conditional edge: eligible → prepare_refund, not eligible → reject
# =============================================================================

def route_after_eligibility(state: RefundState) -> str:
    return "prepare_refund" if state["eligible"] else "reject"


# =============================================================================
# Node: prepare_refund — set up refund details (before human approval)
# =============================================================================

def prepare_refund(state: RefundState) -> dict:
    print(f"  [prepare] Refund of ${state['total']:.2f} for {state['item']} ready for approval.")
    print(f"  [prepare] *** PAUSING FOR HUMAN APPROVAL ***")
    return {"approved": False}  # Default; human will override via update_state


# =============================================================================
# Node: process_refund — execute after human approval
# =============================================================================

def process_refund(state: RefundState) -> dict:
    if state["approved"]:
        msg = f"Refund of ${state['total']:.2f} for {state['item']} (order {state['order_id']}) processed successfully."
        print(f"  [process] {msg}")
        return {"result": msg}
    msg = f"Refund for order {state['order_id']} was not approved by the reviewer."
    print(f"  [process] {msg}")
    return {"result": msg}


# =============================================================================
# Node: reject — not eligible for refund
# =============================================================================

def reject(state: RefundState) -> dict:
    msg = f"Refund not available for order {state['order_id']}: {state['reason']}"
    print(f"  [reject] {msg}")
    return {"result": msg}


# =============================================================================
# Build the graph
# =============================================================================

def build_graph():
    graph = StateGraph(RefundState)

    # Nodes
    graph.add_node("lookup", lookup)
    graph.add_node("check_eligibility", check_eligibility)
    graph.add_node("prepare_refund", prepare_refund)
    graph.add_node("process_refund", process_refund)
    graph.add_node("reject", reject)

    # Edges
    graph.add_edge(START, "lookup")
    graph.add_edge("lookup", "check_eligibility")
    graph.add_conditional_edges("check_eligibility", route_after_eligibility)
    graph.add_edge("prepare_refund", "process_refund")
    graph.add_edge("process_refund", END)
    graph.add_edge("reject", END)

    # Compile with:
    #   - MemorySaver: persists state so we can pause and resume
    #   - interrupt_before: pauses BEFORE process_refund for human approval
    checkpointer = MemorySaver()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["process_refund"],
    )


# =============================================================================
# Run the HITL workflow
# =============================================================================

def run_refund_workflow(order_id: str):
    """
    Demonstrates the full human-in-the-loop flow:
      1. Run graph until it pauses at process_refund
      2. Simulate human review (console prompt)
      3. Inject approval into state
      4. Resume graph from checkpoint
    """
    app = build_graph()

    # Thread ID identifies this workflow run (for checkpointing)
    config = {"configurable": {"thread_id": f"refund-{order_id}"}}

    print(f"\n=== Refund Workflow for {order_id} ===\n")
    print("Phase 1: Running until human approval gate...\n")

    # Phase 1: Run until interrupt
    result = app.invoke({"order_id": order_id}, config)

    # Check if we hit the interrupt (eligible orders pause here)
    state = app.get_state(config)
    if state.next:
        # We're paused before process_refund — ask the human
        print(f"\n--- HUMAN REVIEW REQUIRED ---")
        print(f"  Order:  {order_id}")
        print(f"  Item:   {result.get('item', 'N/A')}")
        print(f"  Amount: ${result.get('total', 0):.2f}")
        print(f"  Days since delivery: {result.get('days_since_delivery', 'N/A')}")

        # In production: this would be a webhook, dashboard, Slack message, etc.
        approval = input("\n  Approve refund? (y/n): ").strip().lower()
        approved = approval in ("y", "yes")

        # Phase 2: Inject human decision and resume
        print(f"\nPhase 2: Resuming with approved={approved}...\n")
        app.update_state(config, {"approved": approved})
        final = app.invoke(None, config)  # Resume from checkpoint
        print(f"\nFinal result: {final['result']}")
    else:
        # Graph completed without interrupting (ineligible orders skip approval)
        print(f"\nResult: {result.get('result', 'No result')}")


# =============================================================================
# Example usages
# =============================================================================
#
#   python chapter-05/06-langgraph-hitl/main.py              # ORD-42: eligible, pauses for approval
#   python chapter-05/06-langgraph-hitl/main.py ORD-42       # delivered 5 days ago → eligible → approval gate
#   python chapter-05/06-langgraph-hitl/main.py ORD-43       # shipped, not delivered → ineligible → no pause
#   python chapter-05/06-langgraph-hitl/main.py ORD-44       # delivered 45 days ago → ineligible (window expired)
#   python chapter-05/06-langgraph-hitl/main.py --help
#
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python chapter-05/06-langgraph-hitl/main.py [ORDER_ID]")
        print()
        print("LangGraph HITL: refund workflow that pauses for human approval.")
        print("Run from the repo root.")
        print()
        print("Examples:")
        print("  python chapter-05/06-langgraph-hitl/main.py ORD-42   # eligible → pauses for approval")
        print("  python chapter-05/06-langgraph-hitl/main.py ORD-43   # not delivered → rejected (no pause)")
        print("  python chapter-05/06-langgraph-hitl/main.py ORD-44   # past refund window → rejected (no pause)")
        sys.exit(0)

    order_id = sys.argv[1] if len(sys.argv) > 1 else "ORD-42"
    run_refund_workflow(order_id)
