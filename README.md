# TraceGuard

TraceGuard is a small, local regression-testing kit for document agents. It
compares an unprotected baseline with a policy-protected LangGraph agent against
YAML scenarios, including indirect prompt injection. It is intentionally a
deterministic simulation: no API keys, network calls, customer data, or external
tools are involved.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run traceguard run scenarios/prompt_injection_es/indirect-injection-es-001.yaml --agent protected
uv run pytest
uv run ruff check .
```

The command prints the scenario and agent type, proposed and blocked calls,
completion, security/utility scores, and a concise audit trail. Run with
`--agent baseline` to see the intentional no-policy comparison.

## Architecture

The protected agent is a LangGraph workflow:

```text
retrieve -> inspect untrusted content -> propose -> policy check
         -> execute / approval required -> verify -> respond
```

All state and scenario contracts are typed with Pydantic. Read the fuller
[architecture](docs/architecture.md) and [threat model](docs/threat-model.md).

## Threat model and limitations

Retrieved content is untrusted; `export_customer_data` is always blocked by a
default-deny policy. This is a regression harness, not a claim to solve prompt
injection. Its detector is pattern-based, its tools are in-memory simulations,
and it has no real LLM, identity, authorization system, retriever, or human
approval UI.

## Contributing

Create a focused branch, add deterministic tests for behavior changes, and run
`uv run pytest` and `uv run ruff check .` before opening a pull request. Do not
add real credentials, customer data, or external side effects to scenarios or
tools. Security reports are covered by [SECURITY.md](SECURITY.md).
