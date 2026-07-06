from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.dev_agent.agent_graph import Mode, build_dev_agent_graph, load_cases, run_case
from benchmarks.dev_agent.local_llm import build_provider
from benchmarks.dev_agent.scoring import summarize


DEFAULT_CASES = Path(__file__).resolve().parent / "cases.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local coding-agent memory utility benchmark.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--modes", nargs="+", default=["no_memory", "rag_only", "omni_memory"])
    parser.add_argument("--provider", default="fake", help="fake, ollama, openai-compatible, openai")
    parser.add_argument("--model", default=None, help="Local model name, e.g. qwen2.5:7b-instruct")
    parser.add_argument("--base-url", default=None, help="LLM base URL, e.g. http://localhost:11434/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--repo-eval", action="store_true", help="Run sandbox repo pytest checks for cases that define a repo fixture.")
    parser.add_argument("--out", type=Path, default=Path("benchmark-results/dev_agent/results.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("benchmark-results/dev_agent/summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = [_normalize_mode(mode) for mode in args.modes]
    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No dev-agent benchmark cases found.")

    llm = build_provider(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    graph = build_dev_agent_graph()

    rows: list[dict[str, Any]] = []
    for case in cases:
        for mode in modes:
            rows.append(run_case(graph, case, mode=mode, llm=llm, repo_eval=args.repo_eval))

    summary = summarize(rows)
    _write_jsonl(args.out, rows)
    _write_json(args.summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nresults: {args.out}")
    print(f"summary: {args.summary_out}")


def _normalize_mode(value: str) -> Mode:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in {"no_memory", "rag_only", "omni_memory"}:
        raise SystemExit(f"Unsupported mode: {value}")
    return normalized  # type: ignore[return-value]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
