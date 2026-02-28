"""
REST API Server: Acme Orders  (Chapter 6, Example 2 — server)

A minimal Flask server that exposes the Acme order database as REST endpoints.
Run this in one terminal, then run main.py in another to see the agent call it.

Endpoints:
    GET  /orders/<order_id>          → order details
    POST /orders/<order_id>/cancel   → cancel an order (body: {"reason": "..."})

Start:
    python chapter-06/02-rest-api-tools/server.py
    # Runs on http://localhost:5001
"""
from flask import Flask, jsonify, request

app = Flask(__name__)

# =============================================================================
# Simulated order database
# =============================================================================

ORDERS_DB = {
    "ORD-42": {"order_id": "ORD-42", "status": "shipped", "eta": "Feb 16", "total": 129.99, "item": "Wireless Headphones"},
    "ORD-43": {"order_id": "ORD-43", "status": "pending", "eta": "Feb 20", "total": 49.99, "item": "USB-C Cable"},
    "ORD-44": {"order_id": "ORD-44", "status": "delivered", "eta": "Feb 10", "total": 89.00, "item": "Bluetooth Speaker"},
}

_cancelled = set()


# =============================================================================
# Endpoints — designed for agentic consumption:
#   - Structured JSON responses (never HTML errors)
#   - Actionable error messages
#   - Consistent field names
# =============================================================================

@app.route("/orders/<order_id>", methods=["GET"])
def get_order(order_id):
    """Look up an order. Returns structured JSON."""
    order = ORDERS_DB.get(order_id)
    if not order:
        return jsonify({"error": "order_not_found", "message": f"No order with ID {order_id}."}), 404
    return jsonify(order)


@app.route("/orders/<order_id>/cancel", methods=["POST"])
def cancel_order(order_id):
    """Cancel an order. Expects JSON body with 'reason'. Idempotent."""
    order = ORDERS_DB.get(order_id)
    if not order:
        return jsonify({"error": "order_not_found", "message": f"No order with ID {order_id}."}), 404

    body = request.get_json(force=True, silent=True) or {}
    reason = body.get("reason", "")
    if not reason:
        return jsonify({"error": "missing_reason", "message": "A 'reason' field is required in the request body."}), 400

    if order["status"] == "delivered":
        return jsonify({"error": "already_delivered", "message": "Order already delivered. Use return process."}), 409
    if order["status"] == "shipped":
        return jsonify({"error": "already_shipped", "message": "Order already shipped. Refuse on delivery."}), 409

    # Idempotent: calling twice returns the same result
    already = order_id in _cancelled
    _cancelled.add(order_id)
    return jsonify({
        "order_id": order_id, "cancelled": True, "refund_amount": order["total"],
        "refund_eta": "3-5 business days",
        **({"note": "Already cancelled. No additional action taken."} if already else {}),
    })


@app.route("/orders", methods=["GET"])
def list_orders():
    """List all orders. Supports ?status= filter."""
    status = request.args.get("status", "").lower()
    orders = list(ORDERS_DB.values())
    if status:
        orders = [o for o in orders if o["status"] == status]
    return jsonify({"orders": orders, "count": len(orders)})


if __name__ == "__main__":
    print("Acme Orders API running on http://localhost:5001")
    print("Endpoints:")
    print("  GET  /orders              — list all orders")
    print("  GET  /orders/<id>         — look up order")
    print("  POST /orders/<id>/cancel  — cancel order")
    app.run(port=5001, debug=True)
