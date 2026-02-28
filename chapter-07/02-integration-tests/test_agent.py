"""
Integration Tests: Testing Agent Behavior  (Chapter 7, Level 2)

Tests with real LLM calls that verify the agent's BEHAVIOR, not exact text:
  - Does it call the right tools?
  - Does it pass correct arguments?
  - Does it handle errors gracefully?
  - Does it stay in scope?

These tests are non-deterministic — run each multiple times and check pass rate.

Run from repo root:
    python chapter-07/02-integration-tests/test_agent.py
    python -m pytest chapter-07/02-integration-tests/test_agent.py -v   # with pytest
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
# Minimal agent (same as Chapter 3 Level 2)
# =============================================================================

ORDERS_DB = {
    "ORD-42": {"order_id": "ORD-42", "status": "shipped", "total": 129.99},
    "ORD-43": {"order_id": "ORD-43", "status": "pending", "total": 49.99},
}

def get_order_status(order_id: str) -> dict:
    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": "order_not_found", "message": f"No order {order_id}."}
    return order

def cancel_order(order_id: str, reason: str) -> dict:
    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": "order_not_found"}
    if order["status"] != "pending":
        return {"error": f"status_{order['status']}", "message": f"Order is {order['status']}."}
    return {"order_id": order_id, "cancelled": True, "refund_amount": order["total"]}

TOOL_MAP = {"get_order_status": get_order_status, "cancel_order": cancel_order}

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "get_order_status", "description": "Look up an order by ID.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "cancel_order", "description": "Cancel a pending order. Only for orders not yet shipped.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["order_id", "reason"]}}},
]

SYSTEM_PROMPT = """You are an Acme Corp support agent.
Help with order status and cancellations. Use tools. Never make up data.
If asked about something other than orders, say: "I can only help with orders." """


# =============================================================================
# Agent with tracing — returns response AND tool call history
# =============================================================================

class AgentTrace:
    def __init__(self):
        self.tool_calls: list[dict] = []
        self.response: str = ""
        self.iterations: int = 0

def run_with_trace(user_message: str) -> AgentTrace:
    """Run the agent and capture a full trace."""
    trace = AgentTrace()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    for i in range(MAX_ITERATIONS):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SCHEMA,
        )
        msg = resp.choices[0].message
        trace.iterations = i + 1
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                fn = TOOL_MAP.get(tc.function.name)
                result = fn(**args) if fn else {"error": "unknown_tool"}
                trace.tool_calls.append({"name": tc.function.name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        else:
            trace.response = msg.content or ""
            return trace
    trace.response = "DID_NOT_CONVERGE"
    return trace


# =============================================================================
# Integration tests — assert on BEHAVIOR, not exact text
# =============================================================================

def test_status_calls_correct_tool():
    """Agent should call get_order_status for a status question."""
    trace = run_with_trace("Where is order ORD-42?")
    tool_names = [tc["name"] for tc in trace.tool_calls]
    assert "get_order_status" in tool_names, f"Expected get_order_status, got {tool_names}"

def test_status_passes_correct_order_id():
    """Agent should pass the correct order ID."""
    trace = run_with_trace("What's the status of ORD-42?")
    order_ids = [tc["args"].get("order_id") for tc in trace.tool_calls if tc["name"] == "get_order_status"]
    assert "ORD-42" in order_ids, f"Expected ORD-42 in {order_ids}"

def test_cancel_calls_cancel_tool():
    """Agent should call cancel_order for a cancellation request."""
    trace = run_with_trace("Cancel order ORD-43 because I changed my mind")
    tool_names = [tc["name"] for tc in trace.tool_calls]
    assert "cancel_order" in tool_names, f"Expected cancel_order, got {tool_names}"

def test_cancel_shipped_explains_error():
    """When cancel fails (shipped), agent should mention it in the response."""
    trace = run_with_trace("Cancel order ORD-42")
    lower = trace.response.lower()
    assert any(w in lower for w in ["shipped", "cannot", "can't", "unable"]), \
        f"Expected error explanation, got: {trace.response[:100]}"

def test_out_of_scope_no_tools():
    """Off-topic questions should not trigger tool calls."""
    trace = run_with_trace("What is the weather in Paris?")
    assert len(trace.tool_calls) == 0, f"Expected no tool calls, got {[tc['name'] for tc in trace.tool_calls]}"

def test_out_of_scope_mentions_orders():
    """Off-topic response should redirect to orders."""
    trace = run_with_trace("Tell me a joke")
    lower = trace.response.lower()
    assert any(w in lower for w in ["order", "can only help", "support"]), \
        f"Expected scope redirect, got: {trace.response[:100]}"

def test_converges():
    """Agent should converge within MAX_ITERATIONS."""
    trace = run_with_trace("Where is order ORD-43?")
    assert trace.response != "DID_NOT_CONVERGE", "Agent did not converge"
    assert trace.iterations <= 5, f"Took {trace.iterations} iterations (expected <= 5)"


# =============================================================================
# Test runner with pass rate (handles non-determinism)
# =============================================================================

def run_test_with_retries(test_fn, retries=3):
    """Run a test up to N times. Pass if majority succeed."""
    passes = 0
    last_error = None
    for _ in range(retries):
        try:
            test_fn()
            passes += 1
        except (AssertionError, Exception) as e:
            last_error = e
    return passes, retries, last_error


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    retries = 3

    print(f"Running {len(tests)} integration tests ({retries} retries each)...\n")

    total_pass = 0
    total_fail = 0

    for test in tests:
        passes, total, last_err = run_test_with_retries(test, retries=retries)
        rate = passes / total
        status = "PASS" if rate >= 0.5 else "FAIL"
        if status == "PASS":
            total_pass += 1
        else:
            total_fail += 1
        err_str = f"  ({last_err})" if status == "FAIL" and last_err else ""
        print(f"  {status}  {test.__name__}  ({passes}/{total}){err_str}")

    print(f"\n{total_pass} passed, {total_fail} failed, {len(tests)} total")
    sys.exit(1 if total_fail else 0)
