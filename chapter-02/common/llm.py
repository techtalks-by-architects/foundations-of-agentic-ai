"""
Shared LLM client for Chapter 02 patterns.
Uses OpenAI API; set OPENAI_API_KEY in environment or .env.
Supports any OpenAI-compatible provider (Ollama, Groq, etc.) via OPENAI_BASE_URL.
"""
import json
import os
import sys
from pathlib import Path

# Load .env from current working directory (run from repo root)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


@dataclass
class ToolCall:
    """Represents a tool call from the model."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """Response from the LLM chat."""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def get_client() -> OpenAI:
    """Return OpenAI client. Exits with a clear message if not configured."""
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print(
            "Error: OPENAI_API_KEY is not set.\n"
            "Set it in your .env file or environment. See Appendix A for setup\n"
            "instructions with Ollama (free, local), Groq (free cloud), or OpenAI.",
            file=sys.stderr,
        )
        sys.exit(1)
    return OpenAI()  # reads OPENAI_BASE_URL and OPENAI_API_KEY from env automatically


def chat(
    messages: list[dict[str, Any]],
    tools: list[Any] | None = None,
    model: str | None = None,
) -> ChatResponse:
    """
    Send messages to the LLM and optionally allow tool calls.
    tools: list of Python functions (will be converted to OpenAI tool schema).
    """
    if model is None:
        model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    client = get_client()

    openai_tools = None
    if tools:
        openai_tools = _functions_to_openai_tools(tools)

    kwargs = {"model": model, "messages": messages}
    if openai_tools:
        kwargs["tools"] = openai_tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)

    msg = response.choices[0].message
    tool_calls = []
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            tool_calls.append(ToolCall(id=tc.id, name=name, arguments=args))

    return ChatResponse(
        content=msg.content or "",
        tool_calls=tool_calls,
    )


def _functions_to_openai_tools(functions: list[Any]) -> list[dict]:
    """Convert Python functions to OpenAI tool schema (function calling format)."""
    import inspect
    tools = []
    for fn in functions:
        sig = inspect.signature(fn)
        params = {}
        for name, param in sig.parameters.items():
            t = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation in (int, float):
                    t = "number"
                elif param.annotation == bool:
                    t = "boolean"
                elif param.annotation in (list, list[str], list[dict]):
                    t = "array"
            params[name] = {"type": t, "description": param.name}
        required = [n for n, p in sig.parameters.items() if p.default == inspect.Parameter.empty]
        tools.append({
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": (fn.__doc__ or "").strip().split("\n")[0],
                "parameters": {
                    "type": "object",
                    "properties": params,
                    **({"required": required} if required else {}),
                },
            },
        })
    return tools
