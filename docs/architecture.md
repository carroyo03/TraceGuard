# Architecture

TraceGuard is a local regression harness for document agents. A YAML scenario is
loaded into typed Pydantic models and run through either a deterministic baseline
or a protected LangGraph workflow.

The protected workflow is:

```text
retrieve_documents -> inspect_untrusted_content -> propose_action
    -> policy_check -> execute_tool / approval_required / blocked
    -> verify_outcome -> respond
```

Retrieved documents are always treated as untrusted data. The deterministic
proposal component intentionally follows embedded tool instructions so that the
test harness can demonstrate the difference made by the policy layer. The policy
is default-deny: only a small allow-list of simulated, local tools can execute.

Every node appends structured audit events to the state. Evaluation happens after
the run and reports separate security and utility scores.

No model, external retriever, or external tool is used. Future LangChain,
Langfuse, Promptfoo, AgentDojo, and LangFlow adapters belong behind the current
typed interfaces.
