"""
Pattern 6: Evaluator–Optimizer

A generator produces output (e.g. code); a separate evaluator reviews it against
criteria and returns APPROVED or feedback. The generator then revises and
resubmits. Loop until approved or max refinements. Two distinct roles keep
evaluation from rubber-stamping the generator's own work.

Run from repo root:
  python chapter-02/06-evaluator-optimizer/main.py [ "A function that parses ISO 8601 dates with timezone support" ]
Requires OPENAI_API_KEY in environment (or uses mock responses).
"""
import sys
from pathlib import Path

# Allow importing shared LLM client from chapter-02/common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import chat


# -----------------------------------------------------------------------------
# Generator and evaluator prompts: different roles for genuine critique
# -----------------------------------------------------------------------------

GENERATOR_PROMPT = """Write a Python function that implements the given requirement.
If you receive feedback, revise your code accordingly. Keep the response to code plus a brief explanation."""

EVALUATOR_PROMPT = """You are a code reviewer. Evaluate the code against these criteria:
1. Correctness: does it implement the requirement?
2. Edge cases: does it handle nulls, empty inputs, boundary values?
3. Readability: is it clean and well-named?

If the code passes all criteria, respond with: APPROVED
Otherwise, respond with specific, actionable feedback (no APPROVED)."""


def generate(requirement: str, feedback: str = "") -> str:
    """Generator: produce code (or revised code given feedback)."""
    prompt = f"Requirement: {requirement}"
    if feedback:
        prompt += f"\n\nPrevious feedback:\n{feedback}"
    response = chat(messages=[
        {"role": "system", "content": GENERATOR_PROMPT},
        {"role": "user", "content": prompt},
    ])
    return response.content or ""


def evaluate(code: str, requirement: str) -> tuple[bool, str]:
    """Evaluator: return approved (bool) and feedback text. APPROVED in content => approved."""
    response = chat(messages=[
        {"role": "system", "content": EVALUATOR_PROMPT},
        {"role": "user", "content": f"Requirement: {requirement}\n\nCode:\n{code}"},
    ])
    content = response.content or ""
    approved = "APPROVED" in content.upper()
    return approved, content


# -----------------------------------------------------------------------------
# Refinement loop: generate -> evaluate -> repeat until approved or max attempts
# -----------------------------------------------------------------------------

def run(requirement: str, max_refinements: int = 3) -> tuple[str, bool, str]:
    feedback = ""
    for attempt in range(max_refinements):
        code = generate(requirement, feedback)
        approved, feedback = evaluate(code, requirement)
        print(f"  Attempt {attempt + 1}: approved={approved}")
        if approved:
            return code, True, feedback
    # Exhausted refinements; return last code and feedback for human review
    return code, False, feedback


# -----------------------------------------------------------------------------
# Example usages (run from chapter-02)
# -----------------------------------------------------------------------------
#
#   # Default: ISO 8601 date parsing
#   python chapter-02/06-evaluator-optimizer/main.py
#
#   # Parse ISO 8601 dates with timezone
#   python chapter-02/06-evaluator-optimizer/main.py "A function that parses ISO 8601 dates with timezone support"
#
#   # Custom requirement
#   python chapter-02/06-evaluator-optimizer/main.py "A function that validates email and returns (local, domain)"
#
#   # Show this help
#   python chapter-02/06-evaluator-optimizer/main.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python chapter-02/06-evaluator-optimizer/main.py [ \"requirement\" ]")
        print()
        print("Examples:")
        print('  python chapter-02/06-evaluator-optimizer/main.py "A function that parses ISO 8601 dates with timezone support"')
        print('  python chapter-02/06-evaluator-optimizer/main.py "A function that validates email"')
        sys.exit(0)
    requirement = sys.argv[1] if len(sys.argv) > 1 else "A function that parses ISO 8601 dates with timezone support"
    print("Requirement:", requirement)
    code, approved, feedback = run(requirement)
    print("Approved:", approved)
    print("Final code/feedback:", code[:500] if len(code) > 500 else code)
