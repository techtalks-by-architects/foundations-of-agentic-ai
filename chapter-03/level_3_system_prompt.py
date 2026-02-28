"""
Level 3: System Prompt and Goal Design

Same loop as Level 2, but with a system prompt that defines goals, boundaries,
policies, and refusal behavior. Constraints are advisory (the model may ignore
them); hard constraints need guardrails in code (Level 4).

Run from repo root:
    python chapter-03/level_3_system_prompt.py [ "Cancel order ORD-42" ]
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


# -----------------------------------------------------------------------------
# System prompt: goals, boundaries, policies, refusal behavior
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
# Tools (same as Level 2)
# -----------------------------------------------------------------------------

def get_order_status(order_id: str) -> dict:
    """Look up an order by ID."""
    return {"order_id": order_id, "status": "shipped", "eta": "Feb 16"}

def cancel_order(order_id: str, reason: str) -> dict:
    """Cancel an order. Requires a reason."""
    return {"order_id": order_id, "cancelled": True}

TOOL_MAP = {"get_order_status": get_order_status, "cancel_order": cancel_order}

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "get_order_status", "description": "Look up an order by ID.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "cancel_order", "description": "Cancel an order. Requires a reason.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["order_id", "reason"]}}},
]


# -----------------------------------------------------------------------------
# Agent loop (same shape as Level 2, now with the system prompt)
# -----------------------------------------------------------------------------

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
                fn = TOOL_MAP[tc.function.name]
                result = fn(**json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
                print(f"  [{i}] {tc.function.name}({tc.function.arguments}) -> {json.dumps(result)}")
        else:
            return msg.content or ""
    return "Agent did not converge. Escalating to a human."


# -----------------------------------------------------------------------------
# Example usages
# -----------------------------------------------------------------------------
#
#   python chapter-03/level_3_system_prompt.py
#   python chapter-03/level_3_system_prompt.py "Cancel order ORD-42"
#   python chapter-03/level_3_system_prompt.py "What's the weather?"   # should refuse (out of scope)
#   python chapter-03/level_3_system_prompt.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-03/level_3_system_prompt.py [ "your message" ]')
        print('\nExamples:')
        print('  python chapter-03/level_3_system_prompt.py "Cancel order ORD-42"')
        print('  python chapter-03/level_3_system_prompt.py "What\'s the weather?"  # should refuse')
        sys.exit(0)
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Cancel order ORD-42 because it hasn't arrived."
    print("User:", query)
    print("Agent:", run(query))
