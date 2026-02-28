"""
Level 2 (LangChain version): The Agent Loop

The same Level 2 example rebuilt with LangChain. Compare this to
chapter-03/level_2_loop.py to see what disappeared.

In Chapter 3, Level 2 was where you built the loop:
    for i in range(MAX_ITERATIONS):
        response = client.chat.completions.create(...)
        if msg.tool_calls:
            # execute tools, append results
        else:
            return msg.content

With LangChain, create_agent() handles this entire loop internally.
You provide the tools and the system prompt; the framework runs
decide → execute → observe → repeat until done.

What you trade: visibility into the loop. You can no longer insert
custom logic between iterations (logging, early exit, side effects)
without understanding LangGraph's execution model.

What you get: zero boilerplate for the most common case — loop until
the model produces a final answer or exhausts the iteration budget.

Run from repo root:
    python chapter-04/level_2_loop_langchain.py
    python chapter-04/level_2_loop_langchain.py "Cancel order ORD-42 because it hasn't arrived."
    python chapter-04/level_2_loop_langchain.py "What's the status of ORD-42?"
    python chapter-04/level_2_loop_langchain.py --help
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
# Tools: same functions as Chapter 3 Level 2, with @tool decorator.
# No TOOL_MAP, no TOOLS_SCHEMA — LangChain generates both from the decorator.
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
# Agent: create_agent replaces the entire loop from Chapter 3 Level 2.
#
# In Level 2 you wrote:
#   for i in range(MAX_ITERATIONS):
#       response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS_SCHEMA)
#       if msg.tool_calls:  ... execute, append ...
#       else: return msg.content
#
# create_agent() does exactly this under the hood. The loop, tool dispatch,
# and message management are all handled by the framework.
# -----------------------------------------------------------------------------

llm = ChatOpenAI(model=MODEL)

agent = create_agent(
    model=llm,
    tools=[get_order_status, cancel_order],
    system_prompt="You are a helpful support agent. Use tools when needed.",
)


# -----------------------------------------------------------------------------
# Example usages
# -----------------------------------------------------------------------------
#
#   python level_2_loop_langchain.py
#   python level_2_loop_langchain.py "Cancel order ORD-42 because it hasn't arrived."
#   python level_2_loop_langchain.py "What's the status of ORD-42?"
#   python level_2_loop_langchain.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-04/level_2_loop_langchain.py ["your message"]')
        print()
        print("LangChain version of Level 2 (the agent loop).")
        print("The loop you built manually in Chapter 3 is now inside create_agent().")
        print()
        print("Examples:")
        print('  python chapter-04/level_2_loop_langchain.py "Cancel order ORD-42 because it hasn\'t arrived."')
        print('  python chapter-04/level_2_loop_langchain.py "What\'s the status of ORD-42?"')
        sys.exit(0)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Cancel order ORD-42 because it hasn't arrived."
    print("User:", query)
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    print("Agent:", result["messages"][-1].content)
