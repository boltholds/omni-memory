from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.builder import build_memory
from app.prompting import PromptRenderer
from domain.llm import LLMResult, Msg
from infra.embeddings.factory import HashEmbedder
from infra.llm.llm_factory import LLMConfig, build_llm

from scoring import score_memory, score_no_memory, summarize_results


class FakeExtractiveLLM:
    """Small deterministic LLM stub for smoke-testing the memory layer.

    It does not try to be smart. It only reads the rendered prompt context and
    returns the most relevant context line. This makes the benchmark runnable in
    CI and on machines without a local LLM provider.
    """

    model = "fake-extractive-llm"

    def generate(self, messages: list[Msg], temperature: float = 0.0) -> LLMResult:
        user_text = messages[-1]["content"] if messages else ""
        question = _extract_question(user_text)
        context = _extract_context(user_text)

        if not context or "(no context)" in context:
            return {
                "text": "Unknown from available context.",
                "model": self.model,
                "finish_reason": "stop",
            }

        selected = _select_context_lines(question, context)
        return {
            "text": " ".join(selected).strip() or "Unknown from available context.",
            "model": self.model,
            "finish_reason": "stop",
        }


def _extract_question(prompt: str) -> str:
    if "Question:" in prompt:
        after = prompt.split("Question:", 1)[1]
        return after.split("Context:", 1)[0].strip()
    return prompt.strip()


def _extract_context(prompt: str) -> str:
    if "Context:" not in prompt:
        return ""
    after = prompt.split("Context:", 1)[1]
    return after.split("Instructions:", 1)[0].strip()


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "is", "are", "what", "where", "which", "does", "do",
        "как", "где", "что", "какой", "какая", "какое", "какие", "мне", "мой", "моя",
        "сейчас", "now", "user", "project",
    }
    out = set()
    for raw in text.casefold().replace("_", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if len(token) >= 3 and token not in stop:
            out.add(token)
    return out


def _select_context_lines(question: str, context: str) -> list[str]:
    lines = [line.strip("- ").strip() for line in context.splitlines() if line.strip()]
    if not lines:
        return []

    current_lines = [line for line in lines if "CURRENT" in line or "status=conflict" in line]
    question_tokens = _tokens(question)

    def score(line: str) -> tuple[int, int, int]:
        line_tokens = _tokens(line)
        overlap = len(question_tokens & line_tokens)
        has_current = 1 if "CURRENT" in line else 0
        has_conflict = 1 if "status=conflict" in line or "conflict" in line.casefold() else 0
        return overlap, has_current, has_conflict

    pool = current_lines or lines
    ranked = sorted(pool, key=score, reverse=True)
    best = ranked[:2]

    if not best or score(best[0]) == (0, 0, 0):
        best = lines[:2]

    return best


def load_cases(paths: list[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.glob("*.jsonl")):
                cases.extend(load_cases([child]))
            continue

        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                try:
                    case = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL in {path}:{lineno}: {exc}") from exc
                case.setdefault("source_file", str(path))
                cases.append(case)
    return cases


def build_provider(args: argparse.Namespace):
    provider = (args.provider or "fake").lower().replace("-", "_")
    if provider in {"fake", "stub", "deterministic"}:
        return FakeExtractiveLLM()

    llm = build_llm(
        LLMConfig(
            provider=provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=args.timeout,
        )
    )
    if llm is None:
        raise RuntimeError(f"Provider {args.provider!r} returned no LLM instance")
    return llm


def generate_without_memory(llm, question: str, *, lang: str, style: str, temperature: float) -> str:
    renderer = PromptRenderer()
    messages = renderer.make_messages(question, [], lang=lang, style=style)
    result = llm.generate(messages, temperature=temperature)
    return str(result.get("text", "")).strip()


def build_fresh_memory(llm, *, reject_conflicts: bool):
    return build_memory(
        use_llm=False,
        reject_conflicts=reject_conflicts,
        llm=llm,
        embedder=HashEmbedder(),
    )


def run_case(case: dict[str, Any], llm, args: argparse.Namespace) -> dict[str, Any]:
    case_id = case["id"]
    question = case["question"]
    lang = case.get("lang", "en")
    style = case.get("style", "concise")
    reject_conflicts = bool(case.get("reject_conflicts", False))

    started = time.perf_counter()
    no_memory_answer = generate_without_memory(
        llm,
        question,
        lang=lang,
        style=style,
        temperature=args.temperature,
    )
    no_memory_latency_ms = (time.perf_counter() - started) * 1000.0

    memory = build_fresh_memory(llm, reject_conflicts=reject_conflicts)
    memory_items = case.get("memory_items", [])
    write_result = memory.write_items_raw(
        memory_items,
        source=f"benchmark:{case_id}",
        meta={"benchmark_case_id": case_id},
    )

    started = time.perf_counter()
    memory_answer = memory.ask(
        question,
        lang=lang,
        style=style,
        temperature=args.temperature,
        include_context=True,
    )
    memory_latency_ms = (time.perf_counter() - started) * 1000.0

    saved_memory_dump = [item.model_dump(mode="json") for item in write_result.saved]
    context_dump = memory_answer.context
    write_summary = {
        "saved": write_result.saved_count,
        "rejected": write_result.rejected_count,
        "errors": write_result.error_count,
        "reasons": write_result.reasons,
        "policy_decisions_count": len(write_result.policy_decisions),
        "operations_count": len(write_result.operations),
    }

    scores = {
        "no_memory": score_no_memory(case, no_memory_answer),
        "memory": score_memory(
            case,
            memory_answer.answer,
            context_dump=context_dump,
            write_summary=write_summary,
            saved_memory_dump=saved_memory_dump,
        ),
    }

    return {
        "id": case_id,
        "category": case.get("category", "unknown"),
        "question": question,
        "source_file": case.get("source_file"),
        "answers": {
            "no_memory": no_memory_answer,
            "memory": memory_answer.answer,
        },
        "latency_ms": {
            "no_memory": round(no_memory_latency_ms, 3),
            "memory": round(memory_latency_ms, 3),
            "overhead": round(memory_latency_ms - no_memory_latency_ms, 3),
        },
        "writeback": write_summary,
        "policy_decisions": [item.model_dump(mode="json") for item in write_result.policy_decisions],
        "memory_operations": [item.model_dump(mode="json") for item in write_result.operations],
        "context": context_dump,
        "advisories": memory_answer.advisories,
        "scores": scores,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OmniMemory A/B benchmark: LLM without memory vs LLM with memory.")
    parser.add_argument(
        "--cases",
        nargs="+",
        type=Path,
        default=[Path(__file__).resolve().parent / "cases"],
        help="Case files or directories with *.jsonl files.",
    )
    parser.add_argument("--out", type=Path, default=Path("benchmark-results/memory_eval/results.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("benchmark-results/memory_eval/summary.json"))
    parser.add_argument("--provider", default="fake", help="fake, openai-compatible, openai, ollama, none-compatible aliases supported by llm_factory.")
    parser.add_argument("--base-url", default=None, help="LLM base URL, e.g. http://localhost:11434/v1")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--api-key", default="local", help="API key for OpenAI-compatible providers")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    if not cases:
        raise SystemExit("No benchmark cases found.")

    llm = build_provider(args)
    results = [run_case(case, llm, args) for case in cases]
    summary = summarize_results(results)

    write_jsonl(args.out, results)
    write_summary(args.summary_out, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nresults: {args.out}")
    print(f"summary: {args.summary_out}")


if __name__ == "__main__":
    main()
