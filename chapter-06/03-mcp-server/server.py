"""
MCP Server: Acme Orders  (Chapter 6, Example 3)

A Model Context Protocol (MCP) server that exposes Acme order tools.
Any MCP-compatible client (Claude Desktop, Cursor, custom agents) can
discover and call these tools.

MCP is the "USB-C for AI tools" — build the tool once, use it with any
MCP-compatible agent.

What this demonstrates:
  - Tools exposed via @server.tool() decorator
  - Structured input/output with type hints
  - Runtime discovery (client calls list_tools to see what's available)

Requirements:
    pip install mcp

Run (stdio transport — for integration with MCP clients):
    python server.py

Run with MCP Inspector (for testing):
    npx @modelcontextprotocol/inspector python server.py
"""
from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("acme-orders")


# =============================================================================
# Simulated order database
# =============================================================================

ORDERS_DB = {
    "ORD-42": {"order_id": "ORD-42", "status": "shipped", "eta": "Feb 16", "total": 129.99, "item": "Wireless Headphones"},
    "ORD-43": {"order_id": "ORD-43", "status": "pending", "eta": "Feb 20", "total": 49.99, "item": "USB-C Cable"},
    "ORD-44": {"order_id": "ORD-44", "status": "delivered", "eta": "Feb 10", "total": 89.00, "item": "Bluetooth Speaker"},
}

_cancelled: set[str] = set()


# =============================================================================
# Tools — same well-designed tools, now exposed via MCP
# Any MCP client can discover these at runtime.
# =============================================================================

@mcp.tool()
def get_order_status(order_id: str) -> dict:
    """Look up an order by ID. Returns status, item, total, and delivery date."""
    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": "order_not_found", "message": f"No order with ID {order_id}."}
    return order


@mcp.tool()
def cancel_order(order_id: str, reason: str) -> dict:
    """Cancel a pending order by ID. Only works for orders that have NOT shipped. Requires a reason."""
    if not order_id.startswith("ORD-"):
        return {"error": "invalid_format", "message": "Order ID must start with ORD-."}
    if not reason.strip():
        return {"error": "missing_reason", "message": "A reason is required."}

    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": "order_not_found", "message": f"No order with ID {order_id}."}
    if order["status"] in ("shipped", "delivered"):
        return {"error": f"already_{order['status']}", "message": f"Order {order_id} is {order['status']} and cannot be cancelled."}

    # Idempotent
    already = order_id in _cancelled
    _cancelled.add(order_id)
    return {
        "order_id": order_id, "cancelled": True, "refund_amount": order["total"],
        "refund_eta": "3-5 business days",
        **({"note": "Already cancelled."} if already else {}),
    }


@mcp.tool()
def search_orders(status: str = "") -> dict:
    """Search orders. Optionally filter by status: pending, shipped, or delivered."""
    results = list(ORDERS_DB.values())
    if status:
        results = [o for o in results if o["status"] == status.lower()]
    return {"orders": results, "count": len(results)}


@mcp.tool()
def request_refund(order_id: str, reason: str) -> dict:
    """Request a refund for a delivered order. Only works for orders with status 'delivered'."""
    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": "order_not_found", "message": f"No order with ID {order_id}."}
    if order["status"] != "delivered":
        return {"error": "not_delivered", "message": f"Refunds only for delivered orders. Status: {order['status']}."}
    return {"order_id": order_id, "refund_initiated": True, "refund_amount": order["total"]}


# =============================================================================
# Resources — data the agent can read (MCP feature beyond tools)
# =============================================================================

@mcp.resource("orders://list")
def list_all_orders() -> str:
    """List of all current orders in the system."""
    import json
    return json.dumps(list(ORDERS_DB.values()), indent=2)


# =============================================================================
# Entry point
# =============================================================================
#
# To run with MCP Inspector (visual testing):
#     npx @modelcontextprotocol/inspector python server.py
#
# To integrate with Claude Desktop, add to claude_desktop_config.json:
#     {
#       "mcpServers": {
#         "acme-orders": {
#           "command": "python",
#           "args": ["/path/to/server.py"]
#         }
#       }
#     }
#
# =============================================================================

if __name__ == "__main__":
    mcp.run()
