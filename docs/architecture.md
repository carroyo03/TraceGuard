# Architecture

TraceGuard is a local regression harness for document agents. It loads a YAML
scenario into Pydantic models, then runs it through either a baseline agent or a
protected LangGraph workflow.

The protected workflow is:

```text
prepare_input -> model_step -> policy_check -> execute_tools
                    |                              |
                    |                              v
                    +---------------------- repeated model_step
                    |
                    v
             verify_response -> finalize

model_step / execute_tools -> limit_reached -> verify_response
```

The compiled LangGraph has explicit `prepare_input`, `model_step`,
`policy_check`, `execute_tools`, `verify_response`, `finalize`, and
`limit_reached` nodes. Retrieved documents are treated as untrusted data. The
policy is default-deny: only a small allow-list of local simulated tools can run.

`Scenario.expected` is intentionally excluded from the runtime graph state. It
is read only by the evaluator after a run. The protected response verifier uses
the user task, retrieved documents, candidate response, and tool/policy results;
its evidence matching is heuristic rather than a general hallucination or
semantic fact-checking solution.

Every node appends structured audit events to the state. Evaluation happens after
the run and reports separate security and utility scores.

## Telemetry

TraceGuard sends observability data through a small telemetry abstraction at the
agent-run boundary. `NoOpTelemetry` is the default and has no side effects.
`LangfuseTelemetry` is created only after explicit environment-based opt-in.
The graph, policy, and evaluator do not depend on the Langfuse SDK.

TraceGuard metrics are local evaluation results: security, utility, and
response-groundedness scores. Langfuse is optional observability for those
results and the run lifecycle: it can record run metadata, audit node names,
tool and policy decisions, latency, errors, and scores. It does not calculate or
change TraceGuard metrics.

By default the telemetry payload excludes documents, candidate responses, final
responses, prompts, and tool arguments. Content can only be included through an
explicit development-only setting.

There is no model, external retriever, or external tool. Any future adapters for
LangChain, Langfuse, Promptfoo, AgentDojo, or LangFlow should use the existing
typed interfaces.
