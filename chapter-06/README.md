# Chapter 6: Agentic-AI-Ready APIs & Tool Design — Source Code

Five examples demonstrating tool design principles, tool surface types,
governance, and discovery patterns for agentic AI systems.

## Setup

See `README.md` for one-time setup (shared venv, `.env`, and dependencies).

## Examples

| Folder | Topic | What it shows |
|---|---|---|
| `01-tool-anatomy/` | Good Tool Design | Four well-designed tools demonstrating naming, descriptions, validation, idempotency, structured errors |
| `02-rest-api-tools/` | REST API as Tools | Flask server + agent that calls HTTP endpoints via tool wrappers |
| `03-mcp-server/` | MCP Server | Model Context Protocol server exposing Acme order tools for any MCP client |
| `04-tool-governance/` | Security & Governance | Input validation, authorization, rate limiting, and audit logging at the tool boundary |
| `05-tool-registry/` | Tool Discovery | Static vs dynamic registries; keyword-based relevance filtering for tool selection at scale |

## Running

All examples run from the repo root:

```bash

# 01 — Tool Anatomy (self-contained)
python chapter-06/01-tool-anatomy/main.py "Where is my order ORD-42?"
python chapter-06/01-tool-anatomy/main.py "Cancel order ORD-43"
python chapter-06/01-tool-anatomy/main.py "Cancel order ORD-42"        # shipped → structured error

# 02 — REST API Tools (two terminals)
# Terminal 1: python chapter-06/02-rest-api-tools/server.py        # runs on localhost:5001
# Terminal 2: python chapter-06/02-rest-api-tools/main.py "Where is my order ORD-42?"

# 03 — MCP Server
npx @modelcontextprotocol/inspector python chapter-06/03-mcp-server/server.py

# 04 — Tool Governance (self-contained)
python chapter-06/04-tool-governance/main.py "Where is my order ORD-42?"      # authorized
python chapter-06/04-tool-governance/main.py "Where is my order ORD-44?"      # BLOCKED
python chapter-06/04-tool-governance/main.py "Cancel order EVIL-1"             # BLOCKED

# 05 — Tool Registry (self-contained)
python chapter-06/05-tool-registry/main.py "Where is my order ORD-42?"
python chapter-06/05-tool-registry/main.py --static "Where is my order ORD-42?"
```

Every example supports `--help`.

## Key Concepts by Example

| Concept (Chapter 6) | Example |
|---|---|
| Six properties of a good tool | 01 — naming, descriptions, validation, idempotency, errors as data |
| REST API wrapping | 02 — thin wrappers over HTTP endpoints |
| MCP (Model Context Protocol) | 03 — `@mcp.tool()` decorator, runtime discovery |
| Input validation | 01 (in-tool) and 04 (governance layer) |
| Authorization | 04 — user ownership check before execution |
| Rate limiting | 04 — per-session call counter |
| Audit logging | 04 — every tool call logged with user, args, result |
| Least privilege | 04 — only user's orders visible |
| Static registry | 05 — fixed tool list at deploy time |
| Dynamic registry | 05 — query-relevant tools from a catalog |
| Tool selection at scale | 05 — keyword-based relevance filter (top-K) |
