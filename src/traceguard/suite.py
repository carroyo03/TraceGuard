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
    """Render compact, copyable suite output without third-party formatting libraries."""
    lines = [
        "| scenario_id | category | agent | user_task_completed | security_score | utility_score | "
        "proposed_tool_calls | blocked_tool_calls |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.scenario_id} | {row.category} | {row.agent} | "
            f"{str(row.user_task_completed).lower()} | {row.security_score:.1f} | "
            f"{row.utility_score:.1f} | {', '.join(row.proposed_tool_calls) or '-'} | "
            f"{', '.join(row.blocked_tool_calls) or '-'} |"
        )
    lines.extend(
        [
            "",
            "Aggregate metrics:",
            f"- mean_security_score: {metrics.mean_security_score:.3f}",
            f"- mean_utility_score: {metrics.mean_utility_score:.3f}",
            f"- unsafe_tool_call_rate: {metrics.unsafe_tool_call_rate:.3f}",
            f"- false_block_rate: {metrics.false_block_rate:.3f}",
        ]
    )
    return "\n".join(lines)
