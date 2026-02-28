"""
Level 2: The Loop (A Real Agent)

The model observes tool results, decides the next action, and repeats until the
goal is met or the iteration budget is exhausted. This is where an agent begins.

Run from repo root:
    python chapter-03/level_2_loop.py [ "Cancel order ORD-42 because it hasn't arrived." ]
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
# Tools: the allow-list of capabilities
# -----------------------------------------------------------------------------

def get_order_status(order_id: str) -> dict:
    """Look up an order by ID."""
    return {"order_id": order_id, "status": "shipped", "eta": "Feb 16"}


def cancel_order(order_id: str, reason: str) -> dict:
    """Cancel an order. Requires a reason."""
    return {"order_id": order_id, "cancelled": True}


TOOL_MAP = {
    "get_order_status": get_order_status,
    "cancel_order": cancel_order,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up an order by ID.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an order. Requires a reason.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id", "reason"],
            },
        },
    },
]


# -----------------------------------------------------------------------------
# Agent loop: observe -> decide -> act -> repeat
# -----------------------------------------------------------------------------

def run(user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful support agent. Use tools when needed."},
        {"role": "user", "content": user_message},
    ]

    for i in range(MAX_ITERATIONS):
        # 1. Decide: LLM may return tool calls or a final answer
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            # 2. Act: execute each tool and feed results back
            messages.append(msg)
            for tc in msg.tool_calls:
                fn = TOOL_MAP[tc.function.name]
                result = fn(**json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
                print(f"  [{i}] {tc.function.name}({tc.function.arguments}) -> {json.dumps(result)}")
            # 3. Loop continues: next iteration observes tool results
        else:
            # Final answer
            return msg.content or ""

    # Safety: bounded loop exhausted
    return "Agent did not converge. Escalating to a human."


# -----------------------------------------------------------------------------
# Example usages
# -----------------------------------------------------------------------------
#
#   python chapter-03/level_2_loop.py
#   python chapter-03/level_2_loop.py "Cancel order ORD-42 because it hasn't arrived."
#   python chapter-03/level_2_loop.py "What's the status of ORD-42?"
#   python chapter-03/level_2_loop.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-03/level_2_loop.py [ "your message" ]')
        print('\nExamples:')
        print('  python chapter-03/level_2_loop.py "Cancel order ORD-42 because it hasn\'t arrived."')
        print('  python chapter-03/level_2_loop.py "What\'s the status of ORD-42?"')
        sys.exit(0)
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Cancel order ORD-42 because it hasn't arrived."
    print("User:", query)
    print("Agent:", run(query))
