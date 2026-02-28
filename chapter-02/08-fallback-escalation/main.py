"""
Pattern 8: Fallback and Escalation

When the primary agent fails (error, timeout, guardrail violation), try a
simpler fallback agent, then a static response, then human escalation. Every
path returns something—no silent failures. This is the agentic equivalent of
circuit breakers and fallback responses in microservices.

Run from repo root:
  python chapter-02/08-fallback-escalation/main.py [ "What's your refund policy?" ]
Requires OPENAI_API_KEY in environment (or uses mock responses).
"""
import sys
from pathlib import Path

# Allow importing shared LLM client from chapter-02/common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import chat


# -----------------------------------------------------------------------------
# Guardrail check: used to decide if we accept the agent's response or fall back
# -----------------------------------------------------------------------------

class AgentTimeoutError(Exception):
    pass


def passes_guardrails(response: str) -> bool:
    """Simple guardrail: reject if response contains SSN-like pattern."""
    import re
    return not re.search(r"\b\d{3}-\d{2}-\d{4}\b", response)


# -----------------------------------------------------------------------------
# Primary and fallback agents: progressively simpler
# -----------------------------------------------------------------------------

def primary_agent(user_message: str) -> str:
    """Full agent: here simplified to a single LLM call (no tools)."""
    response = chat(messages=[{"role": "user", "content": user_message}])
    return response.content or ""


def fallback_agent(user_message: str) -> str:
    """Simpler agent: single turn, tighter system prompt, no tools."""
    response = chat(messages=[
        {"role": "system", "content": "You are a helpful assistant. Give a brief, safe response."},
        {"role": "user", "content": user_message},
    ])
    return response.content or ""


def escalate_to_human(user_message: str, reason: str) -> None:
    """In production: enqueue to human queue, notify, etc. Here we just log."""
    print(f"[Escalation] reason={reason} message={user_message[:50]}...")


# -----------------------------------------------------------------------------
# Fallback chain: primary (with retries) -> fallback -> static -> human
# -----------------------------------------------------------------------------

class AgentWithFallback:
    def __init__(
        self,
        primary_agent,
        fallback_agent,
        static_response: str,
        max_retries: int = 2,
    ):
        self.primary = primary_agent
        self.fallback = fallback_agent
        self.static_response = static_response
        self.max_retries = max_retries

    def run(self, user_message: str) -> str:
        # 1. Try primary agent (with retries)
        for attempt in range(self.max_retries):
            try:
                response = self.primary(user_message)
                if passes_guardrails(response):
                    return response
                print(f"  [Fallback] Primary guardrail failure, attempt {attempt + 1}")
            except AgentTimeoutError:
                print(f"  [Fallback] Primary timeout, attempt {attempt + 1}")
            except Exception as e:
                print(f"  [Fallback] Primary error: {e}, attempt {attempt + 1}")

        # 2. Try fallback agent
        try:
            response = self.fallback(user_message)
            if passes_guardrails(response):
                print("  [Fallback] Fallback agent succeeded")
                return response
        except Exception as e:
            print(f"  [Fallback] Fallback agent error: {e}")

        # 3. Static response
        if self.static_response:
            print("  [Fallback] Returning static response")
            return self.static_response

        # 4. Human escalation (always return something to the user)
        escalate_to_human(user_message, reason="All agents failed")
        return "I've escalated your request to a team member who will follow up."


# -----------------------------------------------------------------------------
# Example usages (run from chapter-02)
# -----------------------------------------------------------------------------
#
#   # Default: refund policy question
#   python chapter-02/08-fallback-escalation/main.py
#
#   # Refund policy
#   python chapter-02/08-fallback-escalation/main.py "What's your refund policy?"
#
#   # Show this help
#   python chapter-02/08-fallback-escalation/main.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python chapter-02/08-fallback-escalation/main.py [ \"your message\" ]")
        print()
        print("Examples:")
        print('  python chapter-02/08-fallback-escalation/main.py "What\'s your refund policy?"')
        sys.exit(0)
    agent = AgentWithFallback(
        primary_agent=primary_agent,
        fallback_agent=fallback_agent,
        static_response="I'm having trouble right now. Please try again shortly.",
    )
    query = "What's your refund policy?" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    print("User:", query)
    print("Response:", agent.run(query))
