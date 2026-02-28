"""
Level 6: Fallback and Cost Control

Same traced + guarded agent, but with two ceilings (iteration count and token
count) and a fallback chain:

    Primary agent → Simpler model (no tools) → Static fallback

Every iteration costs money. Token budgets are like compute budgets for batch
jobs: set them explicitly, monitor them, and kill runaway processes.

Run from repo root:
    python chapter-03/level_6_fallback.py [ "Cancel order ORD-42" ]
"""
import json
import re
import os
import sys
import time
import uuid

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

client = OpenAI()  # reads OPENAI_BASE_URL and OPENAI_API_KEY from env
MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")

# -----------------------------------------------------------------------------
# Ceilings
# -----------------------------------------------------------------------------
MAX_ITERATIONS = 10       # Iteration ceiling — prevents runaway loops
MAX_TOKENS_PER_RUN = 4000 # Cost ceiling — prevents runaway spend


# -----------------------------------------------------------------------------
# System prompt
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a customer support agent for Acme Corp.

Your goal: help the user with order inquiries and cancellations.

Rules:
- Only look up or cancel orders. Do not answer questions about other topics.
- Always check order status before cancelling.
- If the order is already delivered, do not cancel. Explain why.
- If the user asks something outside your scope, say: "I can only help with orders."
- Never make up order information. Only use data from tools.

Available tools: get_order_status, cancel_order."""


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------

def get_order_status(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped", "eta": "Feb 16"}

def cancel_order(order_id: str, reason: str) -> dict:
    return {"order_id": order_id, "cancelled": True}

TOOL_MAP = {"get_order_status": get_order_status, "cancel_order": cancel_order}
TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "get_order_status", "description": "Look up an order by ID.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "cancel_order", "description": "Cancel an order. Requires a reason.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["order_id", "reason"]}}},
]


# -----------------------------------------------------------------------------
# Guardrails (from Level 4)
# -----------------------------------------------------------------------------

def check_input(message: str) -> tuple[bool, str]:
    lower = message.lower()
    if any(p in lower for p in ["ignore previous", "system prompt", "you are now"]):
        return False, "Possible prompt injection detected."
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", message):
        return False, "Please don't include sensitive information like SSNs."
    return True, ""

def check_output(response: str) -> tuple[bool, str]:
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", response):
        return False, "Response contained sensitive data."
    return True, ""


# -----------------------------------------------------------------------------
# Logging (from Level 5)
# -----------------------------------------------------------------------------

def log_event(trace_id: str, event_type: str, data: dict):
    timestamp = time.strftime("%H:%M:%S")
    print(f"  [{timestamp}][trace={trace_id[:8]}] {event_type}: {json.dumps(data)[:200]}")


# -----------------------------------------------------------------------------
# Primary agent with cost control
# -----------------------------------------------------------------------------

def run_primary(user_message: str, trace_id: str) -> str | None:
    """Run the primary agent. Returns answer or None on failure."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    token_count = 0
    log_event(trace_id, "primary_start", {"user_message": user_message})

    for i in range(MAX_ITERATIONS):
        t0 = time.time()
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SCHEMA,
        )
        latency_ms = (time.time() - t0) * 1000

        # Track tokens
        if response.usage:
            token_count += response.usage.total_tokens
        if token_count > MAX_TOKENS_PER_RUN:
            log_event(trace_id, "cost_ceiling", {"tokens": token_count})
            return None  # Trigger fallback

        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                fn = TOOL_MAP[tc.function.name]
                args = json.loads(tc.function.arguments)
                result = fn(**args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
                log_event(trace_id, "tool_call", {
                    "iteration": i, "tool": tc.function.name,
                    "args": args, "result": result,
                    "latency_ms": round(latency_ms),
                    "tokens_so_far": token_count,
                })
        else:
            log_event(trace_id, "primary_answer", {
                "iteration": i,
                "answer": (msg.content or "")[:100],
                "tokens_total": token_count,
            })
            return msg.content or ""

    log_event(trace_id, "primary_did_not_converge", {"iterations": MAX_ITERATIONS})
    return None  # Trigger fallback


# -----------------------------------------------------------------------------
# Fallback chain: simpler model → static response
# -----------------------------------------------------------------------------

def run_fallback(user_message: str, trace_id: str) -> str:
    """Simpler model, no tools, single turn."""
    log_event(trace_id, "fallback_start", {})
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
                {"role": "user", "content": user_message},
            ],
        )
        answer = response.choices[0].message.content or ""
        log_event(trace_id, "fallback_answer", {"answer": answer[:100]})
        return answer
    except Exception as e:
        log_event(trace_id, "fallback_error", {"error": str(e)[:100]})
        return ""


STATIC_RESPONSE = "I'm having trouble right now. Please try again or contact support@acme.com."


# -----------------------------------------------------------------------------
# Full run with fallback chain
# -----------------------------------------------------------------------------

def run_with_fallback(user_message: str) -> str:
    """Primary → Fallback → Static. Guardrails wrap everything."""
    trace_id = str(uuid.uuid4())

    # Input guardrail
    ok, reason = check_input(user_message)
    if not ok:
        log_event(trace_id, "input_blocked", {"reason": reason})
        return f"I can't process that request. ({reason})"

    # Try primary
    try:
        answer = run_primary(user_message, trace_id)
    except Exception as e:
        log_event(trace_id, "primary_error", {"error": str(e)[:100]})
        answer = None

    # If primary failed, try fallback
    if not answer:
        answer = run_fallback(user_message, trace_id)

    # If fallback also failed, static
    if not answer:
        log_event(trace_id, "static_fallback", {})
        answer = STATIC_RESPONSE

    # Output guardrail
    ok, reason = check_output(answer)
    if not ok:
        log_event(trace_id, "output_blocked", {"reason": reason})
        return "I wasn't able to generate a safe response. Please try again."

    return answer


# -----------------------------------------------------------------------------
# Example usages
# -----------------------------------------------------------------------------
#
#   python chapter-03/level_6_fallback.py
#   python chapter-03/level_6_fallback.py "Cancel order ORD-42"
#   python chapter-03/level_6_fallback.py "What's the status of ORD-99?"
#   python chapter-03/level_6_fallback.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-03/level_6_fallback.py [ "your message" ]')
        print('\nExamples:')
        print('  python chapter-03/level_6_fallback.py "Cancel order ORD-42"')
        print('  python chapter-03/level_6_fallback.py "What\'s the status of ORD-99?"')
        sys.exit(0)
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Cancel order ORD-42 because it hasn't arrived."
    print("User:", query)
    print("Agent:", run_with_fallback(query))
