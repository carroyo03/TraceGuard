"""Deterministic scenario-suite execution and report rendering."""

import json
from pathlib import Path

from traceguard.baseline import run_baseline_agent
from traceguard.evaluator import evaluate_suite, suite_row
from traceguard.graph import run_protected_agent
from traceguard.scenarios import load_scenarios
from traceguard.state import AgentType, SuiteMetrics, SuiteReport, SuiteRow
from traceguard.telemetry import Telemetry, get_telemetry_from_env


def run_suite(
    directory: str | Path, agent: AgentType, telemetry: Telemetry | None = None
) -> tuple[list[SuiteRow], SuiteMetrics]:
    """Run every scenario with one selected agent and calculate aggregate metrics."""
    scenarios = load_scenarios(directory)
    runner = run_protected_agent if agent == "protected" else run_baseline_agent
    telemetry_client = telemetry or get_telemetry_from_env()
    runs = [(scenario, runner(scenario, telemetry_client)) for scenario in scenarios]
    return [suite_row(scenario, result) for scenario, result in runs], evaluate_suite(runs)


def render_markdown_table(rows: list[SuiteRow], metrics: SuiteMetrics) -> str:
    """Render the suite results as a Markdown table."""
    lines = [
        "# TraceGuard results",
        "",
        f"Agent: `{rows[0].agent}` | Scenarios: `{len(rows)}`",
        "",
        "| Scenario | Category | Agent | Task complete | Security | Utility | Groundedness | "
        "Proposed tools | Policy blocks |",
        "|:---|:---|:---|:---:|:---:|:---:|:---:|:---|:---|",
    ]
    for row in rows:
        task_status = "yes" if row.user_task_completed else "no"
        lines.append(
            "| "
            f"`{row.scenario_id}` | {row.category.replace('_', ' ')} | `{row.agent}` | "
            f"{task_status} | {row.security_score:.1f} | {row.utility_score:.1f} | "
            f"{row.response_groundedness_score:.1f} | "
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
            "| Mean response groundedness score | "
            f"**{metrics.mean_response_groundedness_score:.3f}** |",
            f"| Unsafe tool-call rate | **{metrics.unsafe_tool_call_rate:.3f}** |",
            f"| False-block rate | **{metrics.false_block_rate:.3f}** |",
        ]
    )
    return "\n".join(lines)


def build_suite_report(rows: list[SuiteRow], metrics: SuiteMetrics) -> SuiteReport:
    """Build the stable JSON contract for one agent's suite result."""
    if not rows:
        raise ValueError("Cannot build a report for an empty scenario suite")
    return SuiteReport(agent=rows[0].agent, scenario_count=len(rows), rows=rows, metrics=metrics)


def render_json_report(rows: list[SuiteRow], metrics: SuiteMetrics) -> str:
    """Render a human-inspectable, machine-readable suite report."""
    return json.dumps(build_suite_report(rows, metrics).model_dump(), indent=2)
