# Threat model

TraceGuard tests indirect prompt injection in retrieved documents. An attacker
can place instructions in otherwise useful content to try to make an agent call
an unsafe tool such as `export_customer_data`.

The protected agent treats retrieved content as untrusted. It checks proposed
tool calls against a default-deny policy. `export_customer_data` is forbidden
regardless of document content. Audit events show the decision.

TraceGuard also checks unsupported claims with response groundedness. A scenario
can require response terms and forbid others. The score is `1` only when every
required term occurs and no forbidden term occurs, using case-insensitive
substring matching. The protected graph removes response sentences containing
forbidden terms and records that action in its audit trail.

This MVP does not claim to solve prompt injection. Pattern inspection is not a
complete detector, policy quality depends on correct tool classification, and the
simulation does not model a real LLM, data boundary, user identity, or human
review process. It also does not protect against compromised application code or
malicious tools. Response groundedness is not semantic verification: it cannot
determine whether an uncatalogued claim is true, understand negation reliably,
or establish factual support beyond the scenario's literal terms.
