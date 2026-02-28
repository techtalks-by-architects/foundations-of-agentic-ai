"""
Level 3 (LangChain version): System Prompt + Multiple Tools

The same Level 3 agent from Chapter 3, rebuilt with LangChain.
Compare to chapter-03/level_3_system_prompt.py.

What changed:
  - @tool replaces 12-line JSON schemas for each tool
  - create_agent replaces the manual agent loop
  - system_prompt parameter replaces manual message construction
  - The system prompt content and policies are still YOUR design

What didn't change:
  - The tool functions are identical
  - The system prompt rules are identical
  - The business logic is identical

Run from repo root:
    python chapter-04/level_3_system_prompt_langchain.py
    python chapter-04/level_3_system_prompt_langchain.py "Cancel order ORD-42"
    python chapter-04/level_3_system_prompt_langchain.py "What's the weather?"
    python chapter-04/level_3_system_prompt_langchain.py --help
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent


MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")


# -----------------------------------------------------------------------------
# Tools: same functions as Chapter 3, with @tool decorator
# LangChain generates the JSON schema from type hints + docstring.
# -----------------------------------------------------------------------------

@tool
def get_order_status(order_id: str) -> dict:
    """Look up an order by ID. Returns order status and estimated delivery."""
    return {"order_id": order_id, "status": "shipped", "eta": "Feb 16"}


@tool
def cancel_order(order_id: str, reason: str) -> dict:
    """Cancel an order. Requires a reason for the cancellation."""
    return {"order_id": order_id, "cancelled": True}


# -----------------------------------------------------------------------------
# System prompt: same rules as Chapter 3, Level 3
# The prompt content is still YOUR design decision.
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
# Agent: create_agent wires model + tools + prompt into a runnable agent
# The agent loop, tool dispatch, and message management are handled internally.
# -----------------------------------------------------------------------------

llm = ChatOpenAI(model=MODEL)

agent = create_agent(
    model=llm,
    tools=[get_order_status, cancel_order],
    system_prompt=SYSTEM_PROMPT,
)


# -----------------------------------------------------------------------------
# Example usages
# -----------------------------------------------------------------------------
#
#   python level_3_system_prompt_langchain.py
#   python level_3_system_prompt_langchain.py "Cancel order ORD-42"
#   python level_3_system_prompt_langchain.py "What's the weather?"
#   python level_3_system_prompt_langchain.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-04/level_3_system_prompt_langchain.py ["your message"]')
        print()
        print("LangChain version of Level 3 (system prompt + multiple tools).")
        print()
        print("Examples:")
        print('  python chapter-04/level_3_system_prompt_langchain.py "Cancel order ORD-42"')
        print('  python chapter-04/level_3_system_prompt_langchain.py "What\'s the weather?"')
        sys.exit(0)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Cancel order ORD-42"
    print("User:", query)
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    print("Agent:", result["messages"][-1].content)
