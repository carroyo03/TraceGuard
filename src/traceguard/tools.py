"""In-memory simulated tools. They never access external systems or data."""

from typing import Any

from traceguard.state import ToolCall


def search_documents(query: str, documents: list[str]) -> list[str]:
    """Return local scenario documents matching a query, or all documents."""
    needle = query.casefold()
    matches = [document for document in documents if needle in document.casefold()]
    return matches or documents


def create_email_draft(subject: str, body: str) -> dict[str, str]:
    """Create an in-memory draft; it is deliberately not sent anywhere."""
    return {"status": "draft_created", "subject": subject, "body": body}


def export_customer_data() -> dict[str, str]:
    """Simulate the dangerous action without exposing or reading any data."""
    return {"status": "simulated_export"}


def execute_tool(call: ToolCall, documents: list[str]) -> dict[str, Any]:
    """Dispatch only known simulated tools."""
    if call.name == "search_documents":
        return search_documents(str(call.arguments.get("query", "")), documents)
    if call.name == "create_email_draft":
        return create_email_draft(
            str(call.arguments.get("subject", "Borrador")),
            str(call.arguments.get("body", "")),
        )
    if call.name == "export_customer_data":
        return export_customer_data()
    raise ValueError(f"Unknown simulated tool: {call.name}")
