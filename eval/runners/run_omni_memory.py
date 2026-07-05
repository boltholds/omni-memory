from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from omni_memory.embeddings import HashEmbedder
from omni_memory.memory import OmniMemory
from omni_memory.infra.repo.episodic_repo import EpisodicRepo
from omni_memory.infra.repo.graph_repo import GraphRepo
from omni_memory.infra.repo.vector_repo import VectorStoreRepo
from eval.metrics.scoring import load_cases, score_answer, summarize, write_report

DEFAULT_DATASETS = [
    Path("eval/datasets/developer_assistant_cases.jsonl"),
    Path("eval/datasets/conflict_cases.jsonl"),
    Path("eval/datasets/preference_cases.jsonl"),
]

TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_\-]+")


def build_eval_memory() -> OmniMemory:
    # HashEmbedder makes eval deterministic and avoids downloading sentence-transformers.
    return OmniMemory(
        use_llm=False,
        vector_repo=VectorStoreRepo(embedder=HashEmbedder()),
        graph_repo=GraphRepo(),
        episodic_repo=EpisodicRepo(db_path=":memory:"),
        reject_conflicts=False,
    )


def normalize_setup_item(item: dict[str, Any]) -> dict[str, Any]:
    item = dict(item)
    if item.get("type") in {"note", "preference"} and "payload" not in item:
        text = item.pop("text", None) or item.pop("content", None)
        item["payload"] = {"text": text or ""}
    return item


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def guess_subject(question: str, facts: list[Any]) -> str | None:
    q = tokens(question)
    subjects = [str(f.subject).lower() for f in facts]
    for subject in subjects:
        if subject in q:
            return subject
    # Small fallback for phrasing like "Where is Alice?" where facts are all about one subject.
    unique = sorted(set(subjects))
    return unique[0] if len(unique) == 1 else None


def format_context_answer(mem: OmniMemory, question: str) -> tuple[str, bool, dict[str, Any]]:
    bundle = mem.retrieve(question, k_sem=5, k_eps=3)
    all_facts = mem.graph_repo.query()
    conflicts = mem.detect_conflicts(question).conflicts
    conflict_detected = bool(conflicts)

    subject = guess_subject(question, all_facts)
    facts = [fact for fact in all_facts if subject is None or fact.subject == subject]
    semantic_chunks = bundle.semantic_chunks

    parts: list[str] = []
    if facts:
        facts_text = "; ".join(f"{f.subject} {f.predicate} {f.object}" for f in facts)
        parts.append("Relevant facts: " + facts_text + ".")

    if conflict_detected:
        conflict_text = "; ".join(f"{c.key}: {', '.join(c.variants)}" for c in conflicts)
        parts.append("Conflicting memory was found: " + conflict_text + ".")

    if semantic_chunks:
        chunk_texts: list[str] = []
        for chunk in semantic_chunks:
            payload = chunk.payload or {}
            chunk_texts.append(str(payload.get("text") or payload.get("content") or payload.get("raw") or payload))
        parts.append("Relevant notes: " + " | ".join(chunk_texts) + ".")

    if not parts:
        parts.append("I do not have enough memory context to answer.")

    context = {
        "facts": [fact.model_dump() for fact in facts],
        "semantic_chunks": [chunk.model_dump() for chunk in semantic_chunks],
        "conflicts": [conflict.model_dump() for conflict in conflicts],
    }
    return " ".join(parts), conflict_detected, context


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    mem = build_eval_memory()
    setup = [normalize_setup_item(item) for item in case.get("setup") or []]
    report = mem.write_items(setup, source=f"eval:{case.get('id')}")
    answer, conflict_detected, context = format_context_answer(mem, str(case.get("question", "")))
    return {
        "case_id": case.get("id"),
        "case": case,
        "write_report": report.model_dump(),
        "answer": answer,
        "context": context,
        "conflict_detected": conflict_detected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OmniMemory eval without external LLM calls.")
    parser.add_argument("--dataset", action="append", type=Path, dest="datasets")
    parser.add_argument("--out", type=Path, default=Path("eval/reports/omni_memory_results.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("eval/reports/omni_memory_latest.md"))
    args = parser.parse_args()

    cases = load_cases(args.datasets or DEFAULT_DATASETS)
    rows = [run_case(case) for case in cases]
    scores = [score_answer(row["case"], row["answer"], conflict_detected=row["conflict_detected"]) for row in rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    write_report(args.report, "omni_memory_no_llm", rows, scores)
    print(json.dumps({"runner": "omni_memory_no_llm", "summary": summarize(scores)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
