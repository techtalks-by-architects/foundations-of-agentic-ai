"""
Tool Registry: Static vs Dynamic Discovery  (Chapter 6, Example 5)

Demonstrates two approaches to making tools available to an agent:

  1. Static registry  — tools are fixed at deploy time (what we've done so far)
  2. Dynamic registry — tools are discovered at runtime from a catalog

Also shows tool selection at scale: when you have many tools, a simple
relevance filter picks the top-K tools per request, keeping the LLM's
context focused.

Architecture notes:
  - Static: simple, predictable, easy to test. Right for most cases.
  - Dynamic: needed when tools are published by different teams on different
    schedules, or when tenants have different capability sets.
  - At scale (50+ tools): the LLM degrades. Use a router or relevance
    filter to pick a small subset per request.

Run from repo root:
    python chapter-06/05-tool-registry/main.py
    python chapter-06/05-tool-registry/main.py "Cancel my order"
    python chapter-06/05-tool-registry/main.py "Search the knowledge base for return policy"
    python chapter-06/05-tool-registry/main.py --help
"""
import json
import os
import sys
from dataclasses import dataclass

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
# Tool catalog: simulates a registry of all available tools
# In production: this could be an MCP discovery call, a database, or a
# service catalog API.
# =============================================================================

@dataclass
class ToolEntry:
    """A tool in the registry."""
    name: str
    description: str
    parameters: dict
    tags: list[str]           # for filtering/relevance
    handler: object = None    # the actual function


def _get_order_status(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped", "eta": "Feb 16", "total": 129.99}

def _cancel_order(order_id: str, reason: str) -> dict:
    return {"order_id": order_id, "cancelled": True, "refund_amount": 49.99}

def _search_orders(status: str = "") -> dict:
    return {"orders": [{"order_id": "ORD-42", "status": "shipped"}], "count": 1}

def _request_refund(order_id: str, reason: str) -> dict:
    return {"order_id": order_id, "refund_initiated": True, "refund_amount": 89.00}

def _search_knowledge_base(query: str) -> dict:
    return {"results": [{"title": "Return Policy", "content": "Returns within 30 days..."}]}

def _submit_feedback(message: str, rating: int) -> dict:
    return {"submitted": True, "message": "Thank you for your feedback."}

def _get_shipping_rates(destination: str, weight_kg: float) -> dict:
    return {"destination": destination, "standard": 5.99, "express": 12.99}

def _track_shipment(tracking_id: str) -> dict:
    return {"tracking_id": tracking_id, "status": "in_transit", "location": "Distribution Center"}


# The full catalog — imagine this grows to 50+ tools across teams
TOOL_CATALOG: list[ToolEntry] = [
    ToolEntry("get_order_status", "Look up an order by ID. Returns status, item, total, and delivery date.",
              {"type": "object", "properties": {"order_id": {"type": "string", "description": "e.g. ORD-42"}}, "required": ["order_id"]},
              tags=["orders", "status"], handler=_get_order_status),

    ToolEntry("cancel_order", "Cancel a pending order by ID. Only works for orders not yet shipped.",
              {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["order_id", "reason"]},
              tags=["orders", "cancel"], handler=_cancel_order),

    ToolEntry("search_orders", "List orders. Optionally filter by status.",
              {"type": "object", "properties": {"status": {"type": "string"}}, "required": []},
              tags=["orders", "search"], handler=_search_orders),

    ToolEntry("request_refund", "Request a refund for a delivered order.",
              {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["order_id", "reason"]},
              tags=["orders", "billing", "refund"], handler=_request_refund),

    ToolEntry("search_knowledge_base", "Search the support knowledge base for articles and policies.",
              {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]},
              tags=["support", "knowledge", "search"], handler=_search_knowledge_base),

    ToolEntry("submit_feedback", "Submit customer feedback with a rating (1-5).",
              {"type": "object", "properties": {"message": {"type": "string"}, "rating": {"type": "integer"}}, "required": ["message", "rating"]},
              tags=["support", "feedback"], handler=_submit_feedback),

    ToolEntry("get_shipping_rates", "Get shipping rates for a destination.",
              {"type": "object", "properties": {"destination": {"type": "string"}, "weight_kg": {"type": "number"}}, "required": ["destination", "weight_kg"]},
              tags=["shipping", "rates"], handler=_get_shipping_rates),

    ToolEntry("track_shipment", "Track a shipment by tracking ID.",
              {"type": "object", "properties": {"tracking_id": {"type": "string"}}, "required": ["tracking_id"]},
              tags=["shipping", "tracking"], handler=_track_shipment),
]


# =============================================================================
# Approach 1: Static Registry — fixed at deploy time
# =============================================================================

class StaticRegistry:
    """Tools are hard-coded. Simple, predictable, testable."""

    def __init__(self, tool_names: list[str]):
        self.tools = [t for t in TOOL_CATALOG if t.name in tool_names]

    def get_tools(self) -> list[ToolEntry]:
        return self.tools


# =============================================================================
# Approach 2: Dynamic Registry — discovered at runtime
# =============================================================================

class DynamicRegistry:
    """
    Tools are discovered from the catalog at runtime.
    Supports filtering by tags and a simple relevance score.
    """

    def __init__(self, catalog: list[ToolEntry]):
        self.catalog = catalog

    def get_all_tools(self) -> list[ToolEntry]:
        """Return everything in the catalog."""
        return self.catalog

    def get_tools_by_tags(self, tags: list[str]) -> list[ToolEntry]:
        """Return tools that match any of the given tags."""
        return [t for t in self.catalog if any(tag in t.tags for tag in tags)]

    def get_relevant_tools(self, query: str, top_k: int = 4) -> list[ToolEntry]:
        """
        Simple keyword-based relevance filter.
        In production: embed query + tool descriptions, use cosine similarity.
        """
        query_lower = query.lower()
        scored = []
        for tool in self.catalog:
            score = 0
            # Check if query words appear in tool name, description, or tags
            for word in query_lower.split():
                if word in tool.name.lower():
                    score += 3
                if word in tool.description.lower():
                    score += 2
                if word in " ".join(tool.tags).lower():
                    score += 1
            scored.append((score, tool))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for score, t in scored[:top_k] if score > 0]


# =============================================================================
# Convert registry tools to OpenAI format
# =============================================================================

def to_openai_schema(tools: list[ToolEntry]) -> list[dict]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in tools
    ]


def to_handler_map(tools: list[ToolEntry]) -> dict:
    return {t.name: t.handler for t in tools}


# =============================================================================
# Agent loop — tools are provided by the registry
# =============================================================================

SYSTEM_PROMPT = """You are an Acme Corp support agent.
Use the available tools to help the customer. Only use data from tools."""


def run(user_message: str, registry_mode: str = "dynamic") -> str:
    """
    Run the agent with tools from the chosen registry.
    registry_mode: "static" (fixed 3 tools) or "dynamic" (query-relevant tools)
    """
    if registry_mode == "static":
        # Static: always the same 3 tools, regardless of query
        registry = StaticRegistry(["get_order_status", "cancel_order", "search_orders"])
        tools = registry.get_tools()
        print(f"  [registry] Static: {[t.name for t in tools]}")
    else:
        # Dynamic: pick relevant tools based on the query
        registry = DynamicRegistry(TOOL_CATALOG)
        tools = registry.get_relevant_tools(user_message, top_k=4)
        if not tools:
            tools = registry.get_tools_by_tags(["orders"])  # fallback
        print(f"  [registry] Dynamic: selected {[t.name for t in tools]} from {len(TOOL_CATALOG)} available")

    schema = to_openai_schema(tools)
    handlers = to_handler_map(tools)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    for i in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=schema,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                fn = handlers.get(tc.function.name)
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = fn(**args) if fn else {"error": "unknown_tool"}
                print(f"  [{i}] {tc.function.name}({json.dumps(args)}) -> {json.dumps(result)[:120]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        else:
            return msg.content or ""
    return "Agent did not converge."


# =============================================================================
# Example usages
# =============================================================================
#
#   # Dynamic registry (default): picks relevant tools per query
#   python chapter-06/05-tool-registry/main.py "Where is my order ORD-42?"
#   python chapter-06/05-tool-registry/main.py "Cancel my order ORD-43"
#   python chapter-06/05-tool-registry/main.py "Search the knowledge base for return policy"
#   python chapter-06/05-tool-registry/main.py "Track shipment TRK-123"
#
#   # Static registry: always uses the same 3 tools
#   python chapter-06/05-tool-registry/main.py --static "Where is my order ORD-42?"
#
#   python chapter-06/05-tool-registry/main.py --help
#
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-06/05-tool-registry/main.py [--static] ["your message"]')
        print()
        print("Demonstrates static vs dynamic tool registries.")
        print(f"Catalog has {len(TOOL_CATALOG)} tools: {[t.name for t in TOOL_CATALOG]}")
        print()
        print("Modes:")
        print("  (default)  Dynamic: selects relevant tools per query")
        print("  --static   Static: always uses 3 fixed tools")
        print()
        print("Examples:")
        print('  python chapter-06/05-tool-registry/main.py "Where is my order ORD-42?"')
        print('  python chapter-06/05-tool-registry/main.py "Search the knowledge base for return policy"')
        print('  python chapter-06/05-tool-registry/main.py "Track shipment TRK-123"')
        print('  python chapter-06/05-tool-registry/main.py --static "Cancel my order"')
        sys.exit(0)

    mode = "dynamic"
    args = sys.argv[1:]
    if args and args[0] == "--static":
        mode = "static"
        args = args[1:]

    query = " ".join(args) if args else "Where is my order ORD-42?"
    print("User:", query)
    print(f"Mode: {mode}\n")
    print("Agent:", run(query, registry_mode=mode))
