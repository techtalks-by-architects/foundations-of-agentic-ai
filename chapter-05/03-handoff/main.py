"""
Handoff with Shared State  (Chapter 5, Pattern 3)

A general support agent starts the conversation. If it determines the customer
needs a refund, it hands off to a billing specialist — passing along the
order context it already gathered.

    User message
      │
      ▼
    support_agent (triage)
      │  ── looks up order, gathers context
      │  ── decides: can I handle this, or hand off?
      │
      ▼  (handoff with context)
    billing_agent (specialist)
      │  ── receives order context + conversation history
      │  ── processes the refund
      │
      ▼
    Result

Architecture notes:
  - The key decision is WHAT STATE transfers with the handoff:
      • Full message history   — comprehensive, but expensive in tokens
      • Structured context dict — cheaper, what we demonstrate here
      • LLM-generated summary  — middle ground (captures nuance, saves tokens)
  - The support agent uses tools (get_order_status) and a handoff tool.
  - The billing agent receives pre-built context so it doesn't re-query.

Run from repo root:
    python chapter-05/03-handoff/main.py
    python chapter-05/03-handoff/main.py "I received the wrong item in order ORD-42, I want my money back"
    python chapter-05/03-handoff/main.py "Where is my order ORD-42?"
    python chapter-05/03-handoff/main.py --help
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
# Simulated order database
# =============================================================================

ORDERS_DB = {
    "ORD-42": {"order_id": "ORD-42", "status": "delivered", "total": 129.99, "item": "Wireless Headphones"},
    "ORD-43": {"order_id": "ORD-43", "status": "shipped",   "total": 49.99,  "item": "USB-C Cable"},
}


# =============================================================================
# Tools for the support agent
# =============================================================================

def get_order_status(order_id: str) -> dict:
    """Look up an order by ID."""
    order = ORDERS_DB.get(order_id)
    if order:
        return order
    return {"error": f"Order {order_id} not found"}


SUPPORT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up an order by ID. Returns order details.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "e.g. ORD-42"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "handoff_to_billing",
            "description": "Hand off to billing specialist when the customer needs a refund. Pass the order_id and a brief summary of the issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order to refund"},
                    "issue_summary": {"type": "string", "description": "Brief summary of why the customer wants a refund"},
                },
                "required": ["order_id", "issue_summary"],
            },
        },
    },
]


# =============================================================================
# Support agent (triage) — can look up orders and hand off to billing
# =============================================================================

SUPPORT_PROMPT = """You are a general support agent for Acme Corp.

You can help customers with order status questions.
If the customer needs a refund or has a billing issue, call handoff_to_billing
with the order_id and a brief issue summary. Do NOT try to process refunds yourself.

Available tools: get_order_status, handoff_to_billing."""


def support_agent(user_message: str) -> str:
    """
    Triage agent. Runs an agent loop with tools.
    If it calls handoff_to_billing, we transfer to the billing agent.
    """
    messages = [
        {"role": "system", "content": SUPPORT_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for i in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=SUPPORT_TOOLS_SCHEMA,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            # Final answer — no handoff needed
            return msg.content or ""

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            if name == "get_order_status":
                result = get_order_status(**args)
                print(f"  [support][{i}] get_order_status({args}) -> {json.dumps(result)}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

            elif name == "handoff_to_billing":
                # === HANDOFF ===
                # Build context to pass to the billing agent
                order = ORDERS_DB.get(args["order_id"], {})
                context = {
                    "order_id": args["order_id"],
                    "issue_summary": args["issue_summary"],
                    "order_details": order,
                }
                print(f"  [support][{i}] HANDOFF to billing: {json.dumps(context)}")
                # Transfer to billing agent with the gathered context
                return billing_agent(context)

    return "I couldn't resolve your request. Let me connect you with a human agent."


# =============================================================================
# Billing agent (specialist) — receives context from handoff
# =============================================================================

def billing_agent(context: dict) -> str:
    """
    Billing specialist. Receives structured context from the support agent.
    Does NOT need to re-query the order — the context is pre-built.
    """
    print(f"  [billing] Received handoff for {context['order_id']}")

    order = context.get("order_details", {})
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Acme Corp billing specialist. "
                    "A support agent has handed off this customer to you. "
                    "Process the refund if the order qualifies. "
                    "Be concise and professional (2-3 sentences)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Handoff context:\n"
                    f"  Order: {context['order_id']}\n"
                    f"  Status: {order.get('status', 'unknown')}\n"
                    f"  Item: {order.get('item', 'unknown')}\n"
                    f"  Total: ${order.get('total', 0):.2f}\n"
                    f"  Issue: {context['issue_summary']}\n\n"
                    f"Process this refund request."
                ),
            },
        ],
    )
    return response.choices[0].message.content or ""


# =============================================================================
# Example usages
# =============================================================================
#
#   python chapter-05/03-handoff/main.py
#   python chapter-05/03-handoff/main.py "I received the wrong item in order ORD-42, I want my money back"
#   python chapter-05/03-handoff/main.py "Where is my order ORD-43?"    # no handoff needed
#   python chapter-05/03-handoff/main.py "I need a refund for ORD-42"   # triggers handoff
#   python chapter-05/03-handoff/main.py --help
#
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-05/03-handoff/main.py ["your message"]')
        print()
        print("Handoff pattern: support agent triages, hands off to billing if needed.")
        print("Run from the repo root.")
        print()
        print("Examples:")
        print('  python chapter-05/03-handoff/main.py "I received the wrong item in ORD-42, I want a refund"')
        print('  python chapter-05/03-handoff/main.py "Where is my order ORD-43?"')
        sys.exit(0)

    query = (
        " ".join(sys.argv[1:]) if len(sys.argv) > 1
        else "I received the wrong item in order ORD-42. I want my money back."
    )
    print("User:", query)
    print("Agent:", support_agent(query))
