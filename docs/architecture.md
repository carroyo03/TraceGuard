# Architecture

TraceGuard is a local regression harness for document agents. It loads a YAML
scenario into Pydantic models, then runs it through either a baseline agent or a
protected LangGraph workflow.

The protected workflow is:

```text
retrieve_documents -> inspect_untrusted_content -> propose_action
    -> policy_check -> execute_tool / approval_required / blocked
    -> verify_outcome -> respond
```

Retrieved documents are treated as untrusted data. The proposal component follows
embedded tool instructions so the test can show the effect of the policy layer.
The policy is default-deny: only a small allow-list of local simulated tools can
run.

Every node appends structured audit events to the state. Evaluation happens after
the run and reports separate security and utility scores.

There is no model, external retriever, or external tool. Any future adapters for
LangChain, Langfuse, Promptfoo, AgentDojo, or LangFlow should use the existing
typed interfaces.
