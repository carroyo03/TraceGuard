# Contributing

Use Python 3.12+ with uv. Keep changes deterministic and local, add tests for
behavior changes, and run `uv run pytest` plus `uv run ruff check .`. Never add
credentials, customer data, or real side effects.

TraceGuard uses a regular local wheel rather than an editable `.pth` install for
macOS compatibility. After changing files under `src/`, run
`uv sync --reinstall-package traceguard` before executing the CLI or tests.
