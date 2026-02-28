"""
Level 0: A Single LLM Call (Not an Agent)

Input in, output out, done. No loop, no tools, no state.
This is what is NOT an agent — just a stateless API call.

Run from repo root:
    python chapter-03/level_0_single_call.py [ "What is a circuit breaker pattern?" ]
"""
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


def ask(question: str) -> str:
    """Single LLM call. No tools, no loop. Equivalent to calling any HTTP API."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content or ""


# -----------------------------------------------------------------------------
# Example usages
# -----------------------------------------------------------------------------
#
#   python chapter-03/level_0_single_call.py
#   python chapter-03/level_0_single_call.py "What is a circuit breaker pattern?"
#   python chapter-03/level_0_single_call.py "Summarize the CAP theorem in two sentences."
#   python chapter-03/level_0_single_call.py --help
#
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print('Usage: python chapter-03/level_0_single_call.py [ "your question" ]')
        print('\nExamples:')
        print('  python chapter-03/level_0_single_call.py "What is a circuit breaker pattern?"')
        print('  python chapter-03/level_0_single_call.py "Summarize the CAP theorem"')
        sys.exit(0)
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is a circuit breaker pattern?"
    print("Question:", question)
    print("Answer:", ask(question))
