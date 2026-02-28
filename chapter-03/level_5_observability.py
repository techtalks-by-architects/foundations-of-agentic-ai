"""
Level 5: Adding Observability

Same guarded agent as Level 4, but every iteration, tool call, and final answer
is logged with a trace ID, latency, and structured data.

Without observability, debugging an agentic system is guesswork.

Run from repo root:
    python chapter-03/level_5_observability.py [ "Cancel order ORD-42" ]
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
MAX_ITERATIONS = 10


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
# Logging / Tracing
# In production: send to your tracing system (OpenTelemetry, Datadog, etc.)
# Here we print structured JSON to stdout for demonstration.
# -----------------------------------------------------------------------------

def log_event(trace_id: str, event_type: str, data: dict):
    """Log a structured event. In production: emit to your telemetry pipeline."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"  [{timestamp}][trace={trace_id[:8]}] {event_type}: {json.dumps(data)[:200]}")


# -----------------------------------------------------------------------------
# Agent loop with tracing
# -----------------------------------------------------------------------------

def run_agent_with_tracing(user_message: str) -> str:
    """The core agent loop with structured trace logging at every step."""
    trace_id = str(uuid.uuid4())
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    log_event(trace_id, "start", {"user_message": user_message})

    for i in range(MAX_ITERATIONS):
        t0 = time.time()
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SCHEMA,
        )
        latency_ms = (time.time() - t0) * 1000
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                fn = TOOL_MAP[tc.function.name]
                args = json.loads(tc.function.arguments)
                result = fn(**args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
                log_event(trace_id, "tool_call", {
                    "iteration": i,
                    "tool": tc.function.name,
                    "args": args,
                    "result": result,
                    "latency_ms": round(latency_ms),
                })
        else:
            log_event(trace_id, "final_answer", {
                "iteration": i,
                "answer": (msg.content or "")[:100],
                "latency_ms": round(latency_ms),
                "total_iterations": i + 1,
            })
            return msg.content or ""

    log_event(trace_id, "did_not_converge", {"iterations": MAX_ITERATIONS})
    return "I couldn't resolve your request. Escalating to a human."


# -----------------------------------------------------------------------------
# Guarded + traced agent
# -----------------------------------------------------------------------------

def guarded_run(user_message: str) -> str:
    ok, reason = check_input(user_message)
    if not ok:
        print(f"  [Guardrail] Input blocked: {reason}")
        return f"I can't process that request. ({reason})"
    agent_response = run_agent_with_tracing(user_message)
    ok, reason = check_output(agent_response)
    if not ok:
        print(f"  [Guardrail] Output blocked: {reason}")
        return "I wasn't able to generate a safe response. Please try again."
    return agent_response


# -----------------------------------------------------------------------------
# Example usages
# -----------------------------------------------------------------------------
#
#   python chapter-03/level_5_observability.py
#   python chapter-03/level_5_observability.py "Cancel order ORD-42"
#   python chapter-03/level_5_observability.py "What's the status of ORD-99?"
#   python chapter-03/level_5_observability.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-03/level_5_observability.py [ "your message" ]')
        print('\nExamples:')
        print('  python chapter-03/level_5_observability.py "Cancel order ORD-42"')
        print('  python chapter-03/level_5_observability.py "What\'s the status of ORD-99?"')
        sys.exit(0)
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Cancel order ORD-42 because it hasn't arrived."
    print("User:", query)
    print("Agent:", guarded_run(query))
