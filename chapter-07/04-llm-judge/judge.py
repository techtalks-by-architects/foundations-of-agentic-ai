"""
LLM-as-Judge: Evaluating Response Quality  (Chapter 7)

Uses a second LLM call to evaluate the quality of an agent's response
when keyword matching isn't enough.

Use cases:
  - Was the tone appropriate?
  - Was the answer helpful and complete?
  - Did the agent explain the error clearly?

Use sparingly — adds cost and its own non-determinism.

Run from repo root:
    python chapter-07/04-llm-judge/judge.py
    python chapter-07/04-llm-judge/judge.py --help
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


# =============================================================================
# LLM-as-Judge: evaluate a question/answer pair
# =============================================================================

def llm_judge(question: str, answer: str, criterion: str) -> dict:
    """
    Use an LLM to evaluate an agent's response.

    Args:
        question: The user's original question.
        answer: The agent's response to evaluate.
        criterion: What to evaluate (e.g., "helpfulness", "accuracy", "tone").

    Returns:
        {"score": 1-5, "reason": "explanation"}
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are evaluating an AI customer support agent's response. "
                    "Score the response from 1 (worst) to 5 (best) on the given criterion. "
                    'Return ONLY valid JSON: {"score": N, "reason": "brief explanation"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User's question: {question}\n\n"
                    f"Agent's response: {answer}\n\n"
                    f"Criterion: {criterion}"
                ),
            },
        ],
    )
    try:
        return json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return {"score": 0, "reason": "Could not parse judge response"}


# =============================================================================
# Multi-criteria evaluation
# =============================================================================

DEFAULT_CRITERIA = [
    "Helpfulness: Does the response address the user's question and provide useful information?",
    "Accuracy: Is the information in the response correct and based on facts (not made up)?",
    "Tone: Is the response professional, friendly, and appropriate for customer support?",
    "Completeness: Does the response cover everything the user needs, including next steps?",
]


def evaluate_response(question: str, answer: str, criteria: list[str] = None) -> list[dict]:
    """Evaluate a response on multiple criteria. Returns list of scores."""
    criteria = criteria or DEFAULT_CRITERIA
    results = []
    for criterion in criteria:
        result = llm_judge(question, answer, criterion)
        result["criterion"] = criterion.split(":")[0] if ":" in criterion else criterion
        results.append(result)
    return results


# =============================================================================
# Demo: evaluate some example responses
# =============================================================================

EXAMPLES = [
    {
        "question": "Where is my order ORD-42?",
        "answer": "Your order ORD-42 has shipped and is estimated to arrive by February 16th. You can track it using the shipping confirmation email we sent.",
    },
    {
        "question": "Cancel order ORD-42",
        "answer": "I'm sorry, but order ORD-42 has already shipped and cannot be cancelled at this point. You can refuse the delivery when it arrives, or contact us after delivery to initiate a return.",
    },
    {
        "question": "Where is my order ORD-42?",
        "answer": "I don't know.",
    },
]


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python judge.py")
        print()
        print("Evaluates example agent responses using LLM-as-judge.")
        print(f"Criteria: {[c.split(':')[0] for c in DEFAULT_CRITERIA]}")
        sys.exit(0)

    for i, ex in enumerate(EXAMPLES):
        print(f"\n{'='*60}")
        print(f"Example {i+1}")
        print(f"  Q: {ex['question']}")
        print(f"  A: {ex['answer'][:80]}...")
        print()

        scores = evaluate_response(ex["question"], ex["answer"])
        total = 0
        for s in scores:
            print(f"  {s['criterion']:<15} {s['score']}/5  {s.get('reason', '')}")
            total += s.get("score", 0)
        avg = total / len(scores) if scores else 0
        print(f"  {'Average':<15} {avg:.1f}/5")
