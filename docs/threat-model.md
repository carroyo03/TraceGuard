# Threat model

TraceGuard tests indirect prompt injection in retrieved documents. An attacker
can place instructions in otherwise useful content in an attempt to make an
agent call an unsafe tool, such as `export_customer_data`.

The protected agent considers retrieved content untrusted. It flags common
instruction-like patterns and checks every proposed tool call against a
default-deny policy. `export_customer_data` is forbidden regardless of document
content. Audit events make the decision inspectable.

This MVP does not claim to solve prompt injection. Pattern inspection is not a
complete detector, policy quality depends on correct tool classification, and the
simulation does not model a real LLM, data boundary, user identity, or human
review process. It also does not protect against compromised application code or
malicious tools.
