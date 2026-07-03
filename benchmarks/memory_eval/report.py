from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scoring import summarize_results


def load_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _fmt_score(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# OmniMemory benchmark report",
        "",
        f"Cases: {summary['cases']}",
        f"No-memory score: {summary['no_memory_score']:.4f}",
        f"Memory score: {summary['memory_score']:.4f}",
        f"Context score: {_fmt_score(summary['context_score'])}",
        f"Memory lift: {summary['memory_lift']:.4f}",
        f"Privacy violations: {summary['privacy_violations']}",
        f"Write failures: {summary['write_failures']}",
        f"Context failures: {summary['context_failures']}",
        f"Answer failures with context OK: {summary['answer_failures_with_context_ok']}",
        "",
        "## By category",
        "",
        "| Category | Cases | No memory | Memory | Context | Lift |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for category, row in summary["categories"].items():
        lines.append(
            f"| {category} | {row['cases']} | {row['no_memory_score']:.4f} | "
            f"{row['memory_score']:.4f} | {_fmt_score(row['context_score'])} | {row['memory_lift']:.4f} |"
        )

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a markdown report from memory benchmark results.")
    parser.add_argument("results", type=Path, nargs="?", default=Path("benchmark-results/memory_eval/results.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("benchmark-results/memory_eval/report.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.results)
    summary = summarize_results(results)
    markdown = render_markdown(summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
