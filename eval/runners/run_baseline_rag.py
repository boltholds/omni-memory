from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from eval.metrics.scoring import load_cases, score_answer, summarize, write_report

DEFAULT_DATASETS = [
    Path("eval/datasets/developer_assistant_cases.jsonl"),
    Path("eval/datasets/conflict_cases.jsonl"),
    Path("eval/datasets/preference_cases.jsonl"),
]

TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_\-]+")


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def item_text(item: dict[str, Any]) -> str:
    if item.get("type") == "fact":
        return f"{item.get('subject', '')} {item.get('predicate', '')} {item.get('object', '')}"
    payload = item.get("payload") or {}
    return str(item.get("text") or item.get("content") or payload.get("text") or payload or item)


def retrieve_simple(memory: list[dict[str, Any]], question: str, k: int = 4) -> list[dict[str, Any]]:
    q_tokens = tokens(question)
    scored = []
    for item in memory:
        text = item_text(item)
        overlap = len(q_tokens & tokens(text))
        scored.append((overlap, text, item))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [item for overlap, _text, item in scored[:k] if overlap > 0]


def answer_from_items(question: str, items: list[dict[str, Any]]) -> str:
    if not items:
        return "I do not have enough memory context to answer."
    facts = [item for item in items if item.get("type") == "fact"]
    notes = [item for item in items if item.get("type") != "fact"]
    parts: list[str] = []
    if facts:
        parts.append("Relevant facts: " + "; ".join(item_text(fact) for fact in facts))
    if notes:
        parts.append("Relevant notes: " + " | ".join(item_text(note) for note in notes))
    return " ".join(parts)


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    memory = list(case.get("setup") or [])
    retrieved = retrieve_simple(memory, str(case.get("question", "")))
    answer = answer_from_items(str(case.get("question", "")), retrieved)
    return {
        "case_id": case.get("id"),
        "case": case,
        "answer": answer,
        "retrieved": retrieved,
        "conflict_detected": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a naive keyword-RAG baseline eval.")
    parser.add_argument("--dataset", action="append", type=Path, dest="datasets")
    parser.add_argument("--out", type=Path, default=Path("eval/reports/baseline_results.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("eval/reports/baseline_latest.md"))
    args = parser.parse_args()

    cases = load_cases(args.datasets or DEFAULT_DATASETS)
    rows = [run_case(case) for case in cases]
    scores = [score_answer(row["case"], row["answer"], conflict_detected=row["conflict_detected"]) for row in rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    write_report(args.report, "baseline_keyword_rag", rows, scores)
    print(json.dumps({"runner": "baseline_keyword_rag", "summary": summarize(scores)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
