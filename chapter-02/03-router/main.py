"""
Pattern 3: Router

Not every request needs the same agent or the same tools. A router examines the
incoming message and dispatches to the right handler—orders, billing, FAQ, or
human escalation. Each route can have its own agent with a smaller sandbox
(principle of least privilege). The router can be a single LLM call, a
classifier, or simple rules.

Run from repo root:
  python chapter-02/03-router/main.py [ "Where is my order ORD-1234?" ]
Requires OPENAI_API_KEY in environment (or uses mock responses).
"""
import sys
from pathlib import Path

# Allow importing shared LLM client from chapter-02/common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import chat


# -----------------------------------------------------------------------------
# Router: classify the request into a category (one LLM call, no tools)
# -----------------------------------------------------------------------------

ROUTER_PROMPT = """Classify the user's request into one of these categories:
- "orders": questions about order status, shipping, cancellation
- "billing": questions about charges, invoices, payment methods
- "faq": general questions answerable from documentation
- "escalate": anything sensitive, angry, or unclear

Respond with only the category name."""


def route(user_message: str) -> str:
    """
    Single classification call: no agent loop, no tools. Returns one of
    orders | billing | faq | escalate. Defaults to escalate if unclear.
    """
    response = chat(messages=[
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": user_message},
    ])
    raw = (response.content or "escalate").strip().lower()
    # Parse response: find first matching category (handles "orders" or "category: orders")
    for cat in ("orders", "billing", "faq", "escalate"):
        if cat in raw:
            return cat
    return "escalate"


# -----------------------------------------------------------------------------
# Handlers: one per category (in production these could be full ReAct agents)
# -----------------------------------------------------------------------------

def orders_agent(msg: str) -> str:
    """Orders domain: could be a full agent with order tools only."""
    return f"[Orders agent] I'll look up your order status for: {msg[:50]}..."


def billing_agent(msg: str) -> str:
    """Billing domain: could be a full agent with billing tools only."""
    return f"[Billing agent] I'll check your invoices and payment methods for: {msg[:50]}..."


def faq_lookup(msg: str) -> str:
    """FAQ: often just retrieval, no agent needed."""
    return "[FAQ] Here's a link to our docs: https://example.com/faq"


def transfer_to_human(msg: str) -> str:
    """Default and fallback: no AI, hand off to human."""
    return "I've transferred you to a human agent. Please hold."


HANDLERS = {
    "orders": orders_agent,
    "billing": billing_agent,
    "faq": faq_lookup,
    "escalate": transfer_to_human,
}


# -----------------------------------------------------------------------------
# Run: route then dispatch to the chosen handler
# -----------------------------------------------------------------------------

def run(user_message: str) -> str:
    """Route the message to a handler and return its response."""
    category = route(user_message)
    # Always have a fallback: unknown category -> escalate
    handler = HANDLERS.get(category, transfer_to_human)
    return handler(user_message)


# -----------------------------------------------------------------------------
# Example usages (run from chapter-02)
# -----------------------------------------------------------------------------
#
#   # Default: order status question
#   python chapter-02/03-router/main.py
#
#   # Order question -> orders agent
#   python chapter-02/03-router/main.py "Where is my order ORD-1234?"
#
#   # Billing question -> billing agent
#   python chapter-02/03-router/main.py "I have a billing question"
#
#   # FAQ -> faq lookup
#   python chapter-02/03-router/main.py "How do I reset my password?"
#
#   # Show this help
#   python chapter-02/03-router/main.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python chapter-02/03-router/main.py [ \"your message\" ]")
        print()
        print("Examples:")
        print('  python chapter-02/03-router/main.py                              # "Where is my order ORD-1234?"')
        print('  python chapter-02/03-router/main.py "Where is my order ORD-1234?"')
        print('  python chapter-02/03-router/main.py "I have a billing question"')
        sys.exit(0)
    query = "Where is my order ORD-1234?" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    print("User:", query)
    print("Routed to:", route(query))
    print("Response:", run(query))
