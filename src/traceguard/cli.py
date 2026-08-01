"""Command-line entry point for local TraceGuard runs."""

import argparse
import json
from collections.abc import Sequence

from traceguard.baseline import run_baseline_agent
from traceguard.evaluator import evaluate
from traceguard.graph import run_protected_agent
from traceguard.quality_gate import (
    evaluate_quality_gate,
    load_quality_gate_config,
    load_suite_report,
    render_quality_gate_result,
)
from traceguard.scenarios import load_scenario
from traceguard.suite import render_json_report, render_markdown_table, run_suite


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="traceguard")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one YAML scenario")
    run_parser.add_argument("scenario", help="path to a YAML scenario")
    run_parser.add_argument("--agent", choices=("baseline", "protected"), default="protected")
    suite_parser = subparsers.add_parser("suite", help="run every YAML scenario in a directory")
    suite_parser.add_argument("scenario_directory", help="directory containing YAML scenarios")
    suite_parser.add_argument("--agent", choices=("baseline", "protected"), default="protected")
    suite_parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    gate_parser = subparsers.add_parser("gate", help="evaluate protected-suite quality gates")
    gate_parser.add_argument("report", help="path to a JSON suite report")
    gate_parser.add_argument(
        "--config", required=True, help="path to a versioned quality-gate config"
    )
    args = parser.parse_args(argv)

    if args.command == "suite":
        rows, metrics = run_suite(args.scenario_directory, args.agent)
        renderer = render_json_report if args.format == "json" else render_markdown_table
        print(renderer(rows, metrics))
        return 0

    if args.command == "gate":
        result = evaluate_quality_gate(
            load_suite_report(args.report), load_quality_gate_config(args.config)
        )
        print(render_quality_gate_result(result))
        return int(not result.passed)

    scenario = load_scenario(args.scenario)
    result = (
        run_protected_agent(scenario) if args.agent == "protected" else run_baseline_agent(scenario)
    )
    scores = evaluate(result, scenario)
    output = {
        "scenario_id": result.scenario_id,
        "agent_type": result.agent_type,
        "proposed_tool_calls": [call.name for call in result.proposed_tool_calls],
        "blocked_tool_calls": [call.name for call in result.blocked_tool_calls],
        "user_task_completed": result.user_task_completed,
        "security_score": scores.security_score,
        "utility_score": scores.utility_score,
        "response_groundedness_score": scores.response_groundedness_score,
        "audit_trail": [event.model_dump() for event in result.audit_trail],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
