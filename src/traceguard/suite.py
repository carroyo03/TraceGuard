"""Deterministic scenario-suite execution and Markdown reporting."""

from pathlib import Path

from traceguard.baseline import run_baseline_agent
from traceguard.evaluator import evaluate_suite, suite_row
from traceguard.graph import run_protected_agent
from traceguard.scenarios import load_scenarios
from traceguard.state import AgentType, SuiteMetrics, SuiteRow


def run_suite(directory: str | Path, agent: AgentType) -> tuple[list[SuiteRow], SuiteMetrics]:
    """Run every scenario with one selected agent and calculate aggregate metrics."""
    scenarios = load_scenarios(directory)
    runner = run_protected_agent if agent == "protected" else run_baseline_agent
    runs = [(scenario, runner(scenario)) for scenario in scenarios]
    return [suite_row(scenario, result) for scenario, result in runs], evaluate_suite(runs)


def render_markdown_table(rows: list[SuiteRow], metrics: SuiteMetrics) -> str:
    """Render a compact, readable Markdown suite report without extra dependencies."""
    lines = [
        "# TraceGuard Suite Report",
        "",
        f"> **Agent:** `{rows[0].agent}` &nbsp;|&nbsp; **Scenarios:** `{len(rows)}`",
        "",
        "| Scenario | Category | Agent | Task | Security | Utility | Proposed tools | "
        "Policy blocks |",
        "|:---|:---|:---|:---:|:---:|:---:|:---|:---|",
    ]
    for row in rows:
        task_status = "✅ Complete" if row.user_task_completed else "❌ Incomplete"
        security_status = "✅ 1.0" if row.security_score else "❌ 0.0"
        utility_status = "✅ 1.0" if row.utility_score else "❌ 0.0"
        lines.append(
            "| "
            f"`{row.scenario_id}` | {row.category.replace('_', ' ')} | `{row.agent}` | "
            f"{task_status} | {security_status} | {utility_status} | "
            f"`{', '.join(row.proposed_tool_calls) or '-'}` | "
            f"{', '.join(row.blocked_tool_calls) or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate metrics",
            "",
            "| Metric | Value |",
            "|:---|---:|",
            f"| Mean security score | **{metrics.mean_security_score:.3f}** |",
            f"| Mean utility score | **{metrics.mean_utility_score:.3f}** |",
            f"| Unsafe tool-call rate | **{metrics.unsafe_tool_call_rate:.3f}** |",
            f"| False-block rate | **{metrics.false_block_rate:.3f}** |",
        ]
    )
    return "\n".join(lines)
