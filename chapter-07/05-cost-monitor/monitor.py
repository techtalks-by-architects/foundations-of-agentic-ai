"""
Cost Monitoring: Token Tracking & Budget Alerts  (Chapter 7)

Demonstrates per-request cost tracking, token budgets, and model selection
as a cost lever. Wraps an agent with cost instrumentation.

Run from repo root:
    python chapter-07/05-cost-monitor/monitor.py
    python chapter-07/05-cost-monitor/monitor.py "Where is my order ORD-42?"
    python chapter-07/05-cost-monitor/monitor.py "Cancel order ORD-43 because I changed my mind"
    python chapter-07/05-cost-monitor/monitor.py --help
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

# Approximate costs per 1K tokens (gpt-4o-mini, as of early 2026)
COST_PER_1K_INPUT = 0.00015
COST_PER_1K_OUTPUT = 0.0006
TOKEN_BUDGET = 4000


# =============================================================================
# Cost tracker
# =============================================================================

@dataclass
class CostReport:
    """Tracks token usage and cost for a single request."""
    iterations: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    budget_exceeded: bool = False
    latency_ms: float = 0.0

    def add_usage(self, usage):
        if usage:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
            self.estimated_cost_usd += (
                (usage.prompt_tokens / 1000) * COST_PER_1K_INPUT
                + (usage.completion_tokens / 1000) * COST_PER_1K_OUTPUT
            )

    def summary(self) -> str:
        return (
            f"  Iterations:     {self.iterations}\n"
            f"  LLM calls:      {self.llm_calls}\n"
            f"  Tool calls:     {self.tool_calls}\n"
            f"  Prompt tokens:  {self.prompt_tokens}\n"
            f"  Compl. tokens:  {self.completion_tokens}\n"
            f"  Total tokens:   {self.total_tokens}\n"
            f"  Est. cost:      ${self.estimated_cost_usd:.6f}\n"
            f"  Budget ({TOKEN_BUDGET}):  {'EXCEEDED' if self.budget_exceeded else 'OK'}\n"
            f"  Latency:        {self.latency_ms:.0f}ms"
        )


# =============================================================================
# Agent with cost instrumentation
# =============================================================================

ORDERS_DB = {
    "ORD-42": {"order_id": "ORD-42", "status": "shipped", "total": 129.99},
    "ORD-43": {"order_id": "ORD-43", "status": "pending", "total": 49.99},
}

def get_order_status(order_id: str) -> dict:
    order = ORDERS_DB.get(order_id)
    return order if order else {"error": "not_found"}

def cancel_order(order_id: str, reason: str) -> dict:
    order = ORDERS_DB.get(order_id)
    if not order:
        return {"error": "not_found"}
    if order["status"] != "pending":
        return {"error": f"status_{order['status']}"}
    return {"order_id": order_id, "cancelled": True}

TOOL_MAP = {"get_order_status": get_order_status, "cancel_order": cancel_order}
TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "get_order_status", "description": "Look up an order by ID.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "cancel_order", "description": "Cancel a pending order.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["order_id", "reason"]}}},
]

SYSTEM_PROMPT = "You are an Acme Corp support agent. Use tools for order data."


def run_with_cost(user_message: str) -> tuple[str, CostReport]:
    """Run the agent and return (response, cost_report)."""
    report = CostReport()
    t0 = time.time()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for i in range(MAX_ITERATIONS):
        report.iterations = i + 1
        report.llm_calls += 1

        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SCHEMA,
        )
        report.add_usage(response.usage)

        # Check token budget
        if report.total_tokens > TOKEN_BUDGET:
            report.budget_exceeded = True
            report.latency_ms = (time.time() - t0) * 1000
            return "I'm having trouble with this request. Please try again or contact support.", report

        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                report.tool_calls += 1
                fn = TOOL_MAP.get(tc.function.name)
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = fn(**args) if fn else {"error": "unknown"}
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        else:
            report.latency_ms = (time.time() - t0) * 1000
            return msg.content or "", report

    report.latency_ms = (time.time() - t0) * 1000
    return "Agent did not converge.", report


# =============================================================================
# Example usages
# =============================================================================
#
#   python chapter-07/05-cost-monitor/monitor.py
#   python chapter-07/05-cost-monitor/monitor.py "Where is my order ORD-42?"
#   python chapter-07/05-cost-monitor/monitor.py "Cancel order ORD-43 because I changed my mind"
#   python chapter-07/05-cost-monitor/monitor.py --help
#
# =============================================================================

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print('Usage: python chapter-07/05-cost-monitor/monitor.py ["your message"]')
        print(f"\nToken budget: {TOKEN_BUDGET}")
        print(f"Cost rates: ${COST_PER_1K_INPUT}/1K input, ${COST_PER_1K_OUTPUT}/1K output")
        sys.exit(0)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Where is my order ORD-42?"
    print("User:", query, "\n")

    answer, report = run_with_cost(query)
    print("Agent:", answer)
    print(f"\n--- Cost Report ---\n{report.summary()}")
