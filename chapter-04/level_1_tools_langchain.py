"""
Level 1 (LangChain version): LLM + Tools — Single-Turn Tool Use

The same Level 1 example rebuilt with LangChain. Compare this to
level_1_tools.py to see what the framework gives you:

  Raw Python (level_1_tools.py)          LangChain (this file)
  ─────────────────────────────          ──────────────────────
  Manual JSON tool schema                @tool decorator (schema auto-generated)
  Manual message list management         Handled by the agent internally
  Manual tool dispatch (TOOL_MAP)        Agent dispatches tools automatically
  Manual two-call flow                   create_agent runs the full loop

What you trade: a dependency (langchain, langchain-openai) and a new
abstraction layer. What you get: less boilerplate, automatic schema
generation, and a consistent interface across LLM providers.

Run from repo root:
    python chapter-04/level_1_tools_langchain.py
    python chapter-04/level_1_tools_langchain.py "Where is my order ORD-42?"
    python chapter-04/level_1_tools_langchain.py "What time is it?"
    python chapter-04/level_1_tools_langchain.py --help
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
# Tool: same function as level_1_tools.py, but with @tool decorator.
# LangChain auto-generates the JSON schema from the type hints and docstring.
# No manual TOOLS_SCHEMA dict needed.
# -----------------------------------------------------------------------------

@tool
def get_order_status(order_id: str) -> dict:
    """Look up an order by ID. Returns order status and estimated delivery."""
    # In production: call your orders service
    return {"order_id": order_id, "status": "shipped", "eta": "Feb 16"}


# -----------------------------------------------------------------------------
# Agent setup: model + tools + system prompt → agent
#
# ChatOpenAI reads OPENAI_BASE_URL and OPENAI_API_KEY from env,
# so it works with Ollama, Groq, and OpenAI out of the box.
#
# create_agent() wires model + tools + prompt into a runnable agent.
# Under the hood it builds a LangGraph state graph that handles
# tool dispatch, message management, and the decide-execute loop.
# -----------------------------------------------------------------------------

llm = ChatOpenAI(model=MODEL)

agent = create_agent(
    model=llm,
    tools=[get_order_status],
    system_prompt=(
        "You are a helpful support agent. "
        "When the user asks about an order, use the get_order_status tool "
        "to look it up. Do not ask the user for information you can look up yourself."
    ),
)


# -----------------------------------------------------------------------------
# Example usages
# -----------------------------------------------------------------------------
#
#   python level_1_tools_langchain.py
#   python level_1_tools_langchain.py "Where is my order ORD-42?"
#   python level_1_tools_langchain.py "What time is it?"
#   python level_1_tools_langchain.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-04/level_1_tools_langchain.py ["your message"]')
        print()
        print("LangChain version of Level 1 (single-turn tool use).")
        print()
        print("Examples:")
        print('  python chapter-04/level_1_tools_langchain.py "Where is my order ORD-42?"')
        print('  python chapter-04/level_1_tools_langchain.py "What time is it?"')
        sys.exit(0)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Where is my order ORD-42?"
    print("User:", query)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    # The last message in the result is the agent's final answer
    print("Agent:", result["messages"][-1].content)
