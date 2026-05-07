from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    answer_correctness: int
    memory_recall: int
    conflict_awareness: int
    hallucination: int
    total: int
    notes: list[str]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Expected object at {path}:{line_no}")
        cases.append(obj)
    return cases


def load_cases(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        out.extend(load_jsonl(path))
    return out


def _contains(text: str, needle: str) -> bool:
    return re.search(re.escape(needle), text, flags=re.IGNORECASE) is not None


def score_answer(case: dict[str, Any], answer: str, *, conflict_detected: bool = False) -> CaseScore:
    expected = case.get("expected") or {}
    must_include = [str(x) for x in expected.get("must_include", [])]
    must_not_include = [str(x) for x in expected.get("must_not_include", [])]
    expect_conflict = bool(expected.get("conflict", False))

    notes: list[str] = []
    included = [needle for needle in must_include if _contains(answer, needle)]
    missing = [needle for needle in must_include if needle not in included]
    forbidden = [needle for needle in must_not_include if _contains(answer, needle)]

    if missing:
        notes.append("missing: " + ", ".join(missing))
    if forbidden:
        notes.append("forbidden: " + ", ".join(forbidden))

    memory_recall = 2 if not missing else (1 if included else 0)
    hallucination = 2 if not forbidden else 0

    conflict_words = ("conflict", "conflicting", "contradiction", "uncertain", "disagree")
    answer_mentions_conflict = any(_contains(answer, word) for word in conflict_words)
    conflict_ok = (conflict_detected or answer_mentions_conflict) if expect_conflict else not answer_mentions_conflict
    conflict_awareness = 2 if conflict_ok else 0

    if missing or forbidden or not conflict_ok:
        answer_correctness = 0 if len(missing) == len(must_include) or forbidden else 1
    else:
        answer_correctness = 2

    total = answer_correctness + memory_recall + conflict_awareness + hallucination
    return CaseScore(
        case_id=str(case.get("id", "unknown")),
        answer_correctness=answer_correctness,
        memory_recall=memory_recall,
        conflict_awareness=conflict_awareness,
        hallucination=hallucination,
        total=total,
        notes=notes,
    )


def summarize(scores: list[CaseScore]) -> dict[str, Any]:
    if not scores:
        return {"cases": 0, "total": 0, "max_total": 0, "score_pct": 0.0}
    total = sum(s.total for s in scores)
    max_total = len(scores) * 8
    return {
        "cases": len(scores),
        "total": total,
        "max_total": max_total,
        "score_pct": round(total / max_total * 100, 2),
        "answer_correctness_avg": round(sum(s.answer_correctness for s in scores) / len(scores), 2),
        "memory_recall_avg": round(sum(s.memory_recall for s in scores) / len(scores), 2),
        "conflict_awareness_avg": round(sum(s.conflict_awareness for s in scores) / len(scores), 2),
        "hallucination_avg": round(sum(s.hallucination for s in scores) / len(scores), 2),
    }


def write_report(path: Path, runner_name: str, rows: list[dict[str, Any]], scores: list[CaseScore]) -> None:
    summary = summarize(scores)
    lines = [
        f"# Evaluation report: {runner_name}",
        "",
        f"Cases: {summary['cases']}",
        f"Score: {summary['total']}/{summary['max_total']} ({summary['score_pct']}%)",
        "",
        "| Case | Total | Correct | Recall | Conflict | Hallucination | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    by_id = {row["case_id"]: row for row in rows}
    for score in scores:
        row = by_id.get(score.case_id, {})
        notes = "; ".join(score.notes) or "ok"
        lines.append(
            f"| {score.case_id} | {score.total}/8 | {score.answer_correctness} | "
            f"{score.memory_recall} | {score.conflict_awareness} | {score.hallucination} | {notes} |"
        )
        answer = str(row.get("answer", "")).replace("\n", " ")
        if answer:
            lines.append(f"| ↳ answer |  |  |  |  |  | {answer[:240]} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def scores_to_dicts(scores: list[CaseScore]) -> list[dict[str, Any]]:
    return [asdict(score) for score in scores]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a JSONL file produced by an eval runner.")
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = load_jsonl(args.results)
    scores = [
        score_answer(row["case"], row["answer"], conflict_detected=bool(row.get("conflict_detected")))
        for row in rows
    ]
    print(json.dumps({"summary": summarize(scores), "scores": scores_to_dicts(scores)}, ensure_ascii=False, indent=2))
