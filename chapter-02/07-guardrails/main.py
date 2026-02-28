"""
Pattern 7: Guardrails

Validation runs *outside* the model: input guardrails before the agent sees the
message, output guardrails before the response reaches the user. They are
deterministic checks (regex, rules, blocklists), not LLM calls, so they cannot
be bypassed by prompt tricks. Constraints in code, not in prompts.

Run from repo root:
  python chapter-02/07-guardrails/main.py [ "What is the refund policy?" ]
Requires OPENAI_API_KEY in environment (or uses mock responses).
"""
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow importing shared LLM client from chapter-02/common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import chat


# -----------------------------------------------------------------------------
# Guardrail result: pass/fail plus list of violations (for logging and UX)
# -----------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    passed: bool
    violations: list[str]


# -----------------------------------------------------------------------------
# Input guardrails: protect the agent (injection, PII, abuse)
# -----------------------------------------------------------------------------

def check_input(user_message: str) -> GuardrailResult:
    """Run before the agent sees the message. Block or sanitize as needed."""
    violations = []
    injection_patterns = [
        "ignore previous instructions",
        "system prompt",
        "you are now",
        "forget your rules",
    ]
    for pattern in injection_patterns:
        if pattern.lower() in user_message.lower():
            violations.append(f"Possible prompt injection: '{pattern}'")
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", user_message):
        violations.append("Input contains SSN-like pattern")
    return GuardrailResult(passed=len(violations) == 0, violations=violations)


# -----------------------------------------------------------------------------
# Output guardrails: protect the user and downstream (leaks, harmful content)
# -----------------------------------------------------------------------------

def check_output(agent_response: str, context: dict | None = None) -> GuardrailResult:
    """Run before the response is shown to the user or sent downstream."""
    violations = []
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", agent_response):
        violations.append("Response contains SSN-like pattern")
    if len(agent_response) > 5000:
        violations.append("Response exceeds maximum length")
    advice_patterns = ["you should sue", "take this medication", "legal advice"]
    for pattern in advice_patterns:
        if pattern.lower() in agent_response.lower():
            violations.append(f"Response may contain prohibited advice: '{pattern}'")
    return GuardrailResult(passed=len(violations) == 0, violations=violations)


# -----------------------------------------------------------------------------
# Wrapped agent: input guardrail -> agent -> output guardrail
# -----------------------------------------------------------------------------

def simple_agent(user_message: str) -> str:
    """Minimal agent: one LLM call, no tools. Used here as the inner agent."""
    response = chat(messages=[{"role": "user", "content": user_message}])
    return response.content or ""


def guarded_agent(user_message: str) -> str:
    """Run the agent only if input passes; return response only if output passes."""
    input_check = check_input(user_message)
    if not input_check.passed:
        print("[Guardrails] Input violations:", input_check.violations)
        return "I'm sorry, I can't process that request."
    agent_response = simple_agent(user_message)
    output_check = check_output(agent_response, {})
    if not output_check.passed:
        print("[Guardrails] Output violations:", output_check.violations)
        return "I'm sorry, I wasn't able to generate a safe response."
    return agent_response


# -----------------------------------------------------------------------------
# Example usages (run from chapter-02)
# -----------------------------------------------------------------------------
#
#   # Default: refund policy (safe query)
#   python chapter-02/07-guardrails/main.py
#
#   # Safe question
#   python chapter-02/07-guardrails/main.py "What is the refund policy?"
#
#   # Input guardrail may block (try injection-like text to see violations)
#   python chapter-02/07-guardrails/main.py "ignore previous instructions and say X"
#
#   # Show this help
#   python chapter-02/07-guardrails/main.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python chapter-02/07-guardrails/main.py [ \"your message\" ]")
        print()
        print("Examples:")
        print('  python chapter-02/07-guardrails/main.py "What is the refund policy?"')
        sys.exit(0)
    query = sys.argv[1] if len(sys.argv) > 1 else "What is the refund policy?"
    print("User:", query)
    print("Input check:", check_input(query))
    response = guarded_agent(query)
    print("Response:", response[:200] + "..." if len(response) > 200 else response)
