"""TraceGuard: deterministic security regression tests for document agents."""

from typing import Any

__all__ = ["run_protected_agent"]


def __getattr__(name: str) -> Any:
    """Load the LangGraph workflow only for the package-level convenience export."""
    if name == "run_protected_agent":
        from traceguard.graph import run_protected_agent

        return run_protected_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
