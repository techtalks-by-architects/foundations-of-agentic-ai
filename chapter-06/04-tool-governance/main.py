"""
Tool Governance: Validation, Auth, Audit, Rate Limiting  (Chapter 6, Example 4)

Demonstrates the security and governance layer that wraps every tool call:

    Agent decides to call tool
      │
      ▼
    Input validation     ── format, length, allowed values
      │
      ▼
    Authorization check  ── does this user own this order?
      │
      ▼
    Rate limit check     ── too many calls this session?
      │
      ▼
    Execute tool         ── the actual business logic
      │
      ▼
    Audit log            ── who, what, when, result
      │
      ▼
    Return to agent

Architecture notes:
  - Governance is a wrapper, not inside the tool. You can compose it.
  - Every check returns structured errors the LLM can understand.
  - The audit trail is your compliance/debugging record.

Run from repo root:
    python chapter-06/04-tool-governance/main.py
    python chapter-06/04-tool-governance/main.py "Cancel order ORD-42"
    python chapter-06/04-tool-governance/main.py "Cancel order ORD-43"
    python chapter-06/04-tool-governance/main.py --help
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field

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
# Simulated data
# =============================================================================

ORDERS_DB = {
    "ORD-42": {"order_id": "ORD-42", "status": "shipped", "total": 129.99, "owner": "user-1"},
    "ORD-43": {"order_id": "ORD-43", "status": "pending", "total": 49.99, "owner": "user-1"},
    "ORD-44": {"order_id": "ORD-44", "status": "delivered", "total": 89.00, "owner": "user-2"},
}

# Simulated current user (in production: from session/JWT)
CURRENT_USER = "user-1"


# =============================================================================
# Governance layer: wraps every tool call
# =============================================================================

@dataclass
class AuditEntry:
    timestamp: str
    user: str
    tool: str
    args: dict
    result: dict
    blocked_by: str = ""

# In production: send to your logging/compliance system
audit_log: list[AuditEntry] = []


@dataclass
class RateLimiter:
    """Simple per-session rate limiter."""
    max_calls: int = 20
    window_seconds: float = 60.0
    calls: list = field(default_factory=list)

    def check(self) -> bool:
        now = time.time()
        # Remove expired entries
        self.calls = [t for t in self.calls if now - t < self.window_seconds]
        if len(self.calls) >= self.max_calls:
            return False
        self.calls.append(now)
        return True


rate_limiter = RateLimiter(max_calls=20, window_seconds=60)


def validate_order_id(order_id: str) -> str | None:
    """Return error message if invalid, None if OK."""
    if not isinstance(order_id, str):
        return "order_id must be a string."
    if not order_id.startswith("ORD-"):
        return f"Invalid order ID format: '{order_id}'. Expected ORD-XXXX."
    if len(order_id) > 20:
        return "Order ID too long."
    return None


def check_authorization(order_id: str, user: str) -> str | None:
    """Return error message if unauthorized, None if OK."""
    order = ORDERS_DB.get(order_id)
    if not order:
        return None  # Let the tool handle 'not found'
    if order.get("owner") != user:
        return f"You don't have permission to access order {order_id}."
    return None


def governed_tool_call(tool_name: str, args: dict, tool_fn, user: str = CURRENT_USER) -> dict:
    """
    Wraps a tool call with governance checks:
      1. Input validation
      2. Authorization
      3. Rate limiting
      4. Execute
      5. Audit logging
    """
    entry = AuditEntry(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        user=user,
        tool=tool_name,
        args=args,
        result={},
    )

    # 1. Input validation
    order_id = args.get("order_id", "")
    if order_id:
        err = validate_order_id(order_id)
        if err:
            entry.result = {"error": "validation_failed", "message": err}
            entry.blocked_by = "validation"
            audit_log.append(entry)
            print(f"  [governance] BLOCKED by validation: {err}")
            return entry.result

    # 2. Authorization
    if order_id:
        err = check_authorization(order_id, user)
        if err:
            entry.result = {"error": "unauthorized", "message": err}
            entry.blocked_by = "authorization"
            audit_log.append(entry)
            print(f"  [governance] BLOCKED by auth: {err}")
            return entry.result

    # 3. Rate limiting
    if not rate_limiter.check():
        entry.result = {"error": "rate_limited", "message": "Too many requests. Please slow down."}
        entry.blocked_by = "rate_limit"
        audit_log.append(entry)
        print(f"  [governance] BLOCKED by rate limit")
        return entry.result

    # 4. Execute the tool
    result = tool_fn(**args)
    entry.result = result
    audit_log.append(entry)

    # 5. Audit log output
    print(f"  [audit] user={user} tool={tool_name} args={args} -> {json.dumps(result)[:100]}")

    return result


# =============================================================================
# Tools (raw business logic — governance wraps them)
# =============================================================================

def _get_order_status(order_id: str) -> dict:
    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": "order_not_found", "message": f"No order with ID {order_id}."}
    # Don't expose internal fields like 'owner'
    return {k: v for k, v in order.items() if k != "owner"}


def _cancel_order(order_id: str, reason: str) -> dict:
    if not reason.strip():
        return {"error": "missing_reason", "message": "Reason is required."}
    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": "order_not_found", "message": f"No order with ID {order_id}."}
    if order["status"] != "pending":
        return {"error": f"status_{order['status']}", "message": f"Only pending orders can be cancelled. This order is {order['status']}."}
    return {"order_id": order_id, "cancelled": True, "refund_amount": order["total"]}


def _search_orders() -> dict:
    # Only return orders owned by the current user
    user_orders = [
        {k: v for k, v in o.items() if k != "owner"}
        for o in ORDERS_DB.values()
        if o.get("owner") == CURRENT_USER
    ]
    return {"orders": user_orders, "count": len(user_orders)}


RAW_TOOLS = {
    "get_order_status": _get_order_status,
    "cancel_order": _cancel_order,
    "search_orders": _search_orders,
}

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "get_order_status", "description": "Look up an order by ID.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string", "description": "Acme order ID, e.g. ORD-42"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "cancel_order", "description": "Cancel a pending order. Only works for orders not yet shipped.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string", "description": "Acme order ID"}, "reason": {"type": "string", "description": "Cancellation reason"}}, "required": ["order_id", "reason"]}}},
    {"type": "function", "function": {"name": "search_orders", "description": "List all your orders.", "parameters": {"type": "object", "properties": {}, "required": []}}},
]


# =============================================================================
# Agent loop — uses governed_tool_call instead of calling tools directly
# =============================================================================

SYSTEM_PROMPT = """You are an Acme Corp support agent for the current user.
Help with order status, cancellations, and listing orders.
Use tools to get data. Never make up order information."""


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
                tool_fn = RAW_TOOLS.get(tc.function.name)
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                # === GOVERNED CALL ===
                result = governed_tool_call(tc.function.name, args, tool_fn)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        else:
            return msg.content or ""
    return "Agent did not converge."


# =============================================================================
# Example usages
# =============================================================================
#
#   python chapter-06/04-tool-governance/main.py                                      # default: status check
#   python chapter-06/04-tool-governance/main.py "Where is my order ORD-42?"          # owned by user-1 → OK
#   python chapter-06/04-tool-governance/main.py "Where is my order ORD-44?"          # owned by user-2 → BLOCKED
#   python chapter-06/04-tool-governance/main.py "Cancel order ORD-43"                # pending, owned → OK
#   python chapter-06/04-tool-governance/main.py "Cancel order EVIL-1"                # bad format → validation error
#   python chapter-06/04-tool-governance/main.py "Show me all my orders"              # only user-1's orders
#   python chapter-06/04-tool-governance/main.py --help
#
#   After running, the audit log is printed showing every tool call.
#
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-06/04-tool-governance/main.py ["your message"]')
        print()
        print("Demonstrates tool governance: validation, auth, rate limiting, audit.")
        print(f"Current user: {CURRENT_USER} (owns ORD-42, ORD-43)")
        print()
        print("Examples:")
        print('  python chapter-06/04-tool-governance/main.py "Where is my order ORD-42?"      # authorized')
        print('  python chapter-06/04-tool-governance/main.py "Where is my order ORD-44?"      # unauthorized (user-2)')
        print('  python chapter-06/04-tool-governance/main.py "Cancel order ORD-43"             # allowed')
        print('  python chapter-06/04-tool-governance/main.py "Cancel order EVIL-1"             # bad format')
        sys.exit(0)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Where is my order ORD-42?"
    print("User:", query)
    print(f"(Current user: {CURRENT_USER})\n")
    print("Agent:", run(query))

    # Print audit log summary
    print(f"\n--- Audit Log ({len(audit_log)} entries) ---")
    for e in audit_log:
        blocked = f" BLOCKED:{e.blocked_by}" if e.blocked_by else ""
        print(f"  {e.timestamp} | user={e.user} | {e.tool}({e.args}){blocked}")
