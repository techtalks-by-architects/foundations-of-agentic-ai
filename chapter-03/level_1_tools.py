"""
Level 1: LLM + Tools (Single-Turn Tool Use)

The model chooses whether to call a tool; the platform runs it. Two LLM calls
(decide, then answer), but no loop. Still not an agent.

Run from repo root:
    python chapter-03/level_1_tools.py [ "Where is my order ORD-42?" ]
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


# -----------------------------------------------------------------------------
# Tool: the only capability the model can invoke
# -----------------------------------------------------------------------------

def get_order_status(order_id: str) -> dict:
    """Look up an order by ID. Returns status and details."""
    # In production: call your orders service
    return {"order_id": order_id, "status": "shipped", "eta": "Feb 16"}


TOOLS_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "Look up an order by ID. Returns status and details.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order ID"}},
            "required": ["order_id"],
        },
    },
}]

TOOL_MAP = {"get_order_status": get_order_status}


# -----------------------------------------------------------------------------
# Single-turn tool use: decide -> execute tool -> answer
# -----------------------------------------------------------------------------

def ask(user_message: str) -> str:
    """One tool call at most, then a final answer. No loop."""
    # System prompt nudges the model to actually USE the tools rather than
    # asking the user for info it could look up itself. Without this, smaller
    # local models (e.g. qwen2.5:7b) often respond with clarifying questions
    # instead of calling the tool — even when the tool schema is provided.
    messages = [
        {"role": "system", "content": "You are a helpful support agent. "
         "When the user asks about an order, use the get_order_status tool "
         "to look it up. Do not ask the user for information you can look up yourself."},
        {"role": "user", "content": user_message},
    ]

    # 1. First LLM call: may return a tool call or a direct answer
    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS_SCHEMA,
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        # 2. Execute the tool (platform runs it, not the model)
        tc = msg.tool_calls[0]
        fn = TOOL_MAP[tc.function.name]
        result = fn(**json.loads(tc.function.arguments))
        print(f"  [Tool] {tc.function.name}({tc.function.arguments}) -> {json.dumps(result)}")

        # 3. Second LLM call: model uses the tool result to answer the user
        messages.append(msg)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        final = client.chat.completions.create(model=MODEL, messages=messages)
        return final.choices[0].message.content or ""
    else:
        return msg.content or ""


# -----------------------------------------------------------------------------
# Example usages
# -----------------------------------------------------------------------------
#
#   python chapter-03/level_1_tools.py
#   python chapter-03/level_1_tools.py "Where is my order ORD-42?"
#   python chapter-03/level_1_tools.py "What time is it?"   # no tool call expected
#   python chapter-03/level_1_tools.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-03/level_1_tools.py [ "your message" ]')
        print('\nExamples:')
        print('  python chapter-03/level_1_tools.py "Where is my order ORD-42?"')
        sys.exit(0)
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Where is my order ORD-42?"
    print("User:", query)
    print("Agent:", ask(query))
