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
uv run traceguard run scenarios/prompt_injection/indirect-injection-001.yaml --agent protected
uv run pytest
uv run ruff check .
```

The command prints the scenario and agent type, proposed and blocked calls,
completion, security/utility scores, and a concise audit trail. Run with
`--agent baseline` to see the intentional no-policy comparison.

## Optional Langfuse observability

Local execution is the default and does not require Langfuse, credentials, or a
network connection. To add the optional SDK:

```bash
uv sync --extra langfuse
export TRACEGUARD_LANGFUSE_ENABLED=true
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
uv run traceguard suite scenarios/ --agent protected
```

| Variable | Default | Purpose |
|---|---|---|
| `TRACEGUARD_LANGFUSE_ENABLED` | `false` | Enables the optional Langfuse adapter. |
| `TRACEGUARD_CAPTURE_CONTENT` | `false` | Includes documents and responses only when explicitly enabled. |
| `LANGFUSE_PUBLIC_KEY` | — | Required when Langfuse is enabled. |
| `LANGFUSE_SECRET_KEY` | — | Required when Langfuse is enabled. |
| `LANGFUSE_BASE_URL` | Langfuse Cloud | Optional Langfuse Cloud or self-hosted endpoint. |

`TRACEGUARD_CAPTURE_CONTENT` defaults to `false`. In that mode, TraceGuard sends
only scenario and agent identifiers, node names, tool names, policy decisions,
scores, latency, and status metadata. It does not send documents, prompts,
candidate responses, final responses, or tool arguments.

Set `TRACEGUARD_CAPTURE_CONTENT=true` only for local development data that is
safe to trace. Never send real, sensitive, customer, or credential-bearing data
to Langfuse. If Langfuse is disabled, unavailable, or misconfigured, TraceGuard
falls back to local no-op telemetry and continues running.

For self-hosted development setups, configure `LANGFUSE_BASE_URL` to the URL of
your Langfuse deployment. Docker Compose is documentation-only for this project:

```bash
# Run from a separate Langfuse deployment checkout, not this repository.
docker compose up -d
export LANGFUSE_BASE_URL="http://localhost:3000"
```

Follow the official Langfuse self-hosting guide for the Compose configuration;
do not add its services or credentials to this repository. See the
[Langfuse self-hosting guide](https://langfuse.com/docs/deployment/self-host).

## Benchmark results

Run the complete deterministic benchmark suite with either agent:

```bash
uv run traceguard suite scenarios/ --agent protected
uv run traceguard suite scenarios/ --agent baseline
```

Each command prints a compact Markdown table with one row per scenario and the
aggregate `mean_security_score`, `mean_utility_score`, `unsafe_tool_call_rate`,
`false_block_rate`, and `mean_response_groundedness_score`. The suite contains benign document tasks,
indirect-injection attempts, and an unsupported-claim case. Results are
intentionally not embedded here: reproduce them from the checked-in YAML
scenarios and deterministic implementation.

## Deterministic response-constraint score

`response_groundedness_score` is the compatibility name for the deterministic
response-constraint score. It is `1` only when the response contains every
`required_response_terms` value and none of its `forbidden_response_terms`
values; otherwise it is `0`. A scenario can provide a shared
`candidate_response`: baseline returns it unchanged, while protected verifies it
and removes sentences containing forbidden terms. This is literal,
case-insensitive term matching, not semantic fact checking or a general solution
to hallucination.

## Architecture

The protected agent is a LangGraph workflow:

```text
retrieve -> inspect untrusted content -> propose -> policy check
         -> execute / approval required -> verify outcome
         -> respond -> verify response
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
