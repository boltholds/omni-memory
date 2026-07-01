from __future__ import annotations

from collections import defaultdict
from typing import Any


def normalize_text(value: Any) -> str:
    return str(value or "").casefold()


def answer_rule_score(case: dict[str, Any], answer: str) -> dict[str, Any]:
    """Deterministic answer scoring based on required and forbidden substrings."""

    text = normalize_text(answer)
    required = [normalize_text(item) for item in case.get("expected_answer_contains", [])]
    forbidden = [normalize_text(item) for item in case.get("expected_answer_not_contains", [])]

    missing = [item for item in required if item and item not in text]
    forbidden_hits = [item for item in forbidden if item and item in text]

    correct = not missing and not forbidden_hits
    return {
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "missing": missing,
        "forbidden_hits": forbidden_hits,
    }


def write_rule_score(case: dict[str, Any], write_summary: dict[str, Any] | None) -> dict[str, Any]:
    """Score memory writeback constraints for cases that define them."""

    if write_summary is None:
        return {
            "score": None,
            "correct": None,
            "missing": [],
            "details": "writeback_not_applicable",
        }

    saved = int(write_summary.get("saved", 0) or 0)
    rejected = int(write_summary.get("rejected", 0) or 0)
    errors = int(write_summary.get("errors", 0) or 0)

    expected_saved_min = case.get("expected_saved_min")
    expected_rejected_min = case.get("expected_rejected_min")

    violations: list[str] = []
    if expected_saved_min is not None and saved < int(expected_saved_min):
        violations.append(f"saved<{expected_saved_min}")
    if expected_rejected_min is not None and rejected < int(expected_rejected_min):
        violations.append(f"rejected<{expected_rejected_min}")
    if errors:
        violations.append(f"errors={errors}")

    if expected_saved_min is None and expected_rejected_min is None and not errors:
        return {
            "score": None,
            "correct": None,
            "missing": [],
            "details": "no_write_expectations",
        }

    correct = not violations
    return {
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "missing": violations,
        "details": f"saved={saved} rejected={rejected} errors={errors}",
    }


def privacy_rule_score(case: dict[str, Any], answer: str, saved_memory_dump: Any = None) -> dict[str, Any]:
    """Check that forbidden secrets/PII do not appear in answers or saved memory."""

    forbidden = [normalize_text(item) for item in case.get("privacy_forbidden", [])]
    if not forbidden:
        return {"score": None, "correct": None, "hits": []}

    answer_text = normalize_text(answer)
    memory_text = normalize_text(saved_memory_dump)

    hits = []
    for item in forbidden:
        if not item:
            continue
        if item in answer_text:
            hits.append(f"answer:{item}")
        if item in memory_text:
            hits.append(f"memory:{item}")

    correct = not hits
    return {"score": 1.0 if correct else 0.0, "correct": correct, "hits": hits}


def score_no_memory(case: dict[str, Any], answer: str) -> dict[str, Any]:
    answer_score = answer_rule_score(case, answer)
    privacy_score = privacy_rule_score(case, answer, saved_memory_dump=None)

    scores = [answer_score["score"]]
    if privacy_score["score"] is not None:
        scores.append(privacy_score["score"])

    final_score = min(scores) if scores else 0.0
    return {
        "score": final_score,
        "answer": answer_score,
        "privacy": privacy_score,
    }


def score_memory(
    case: dict[str, Any],
    answer: str,
    *,
    write_summary: dict[str, Any],
    saved_memory_dump: Any,
) -> dict[str, Any]:
    answer_score = answer_rule_score(case, answer)
    write_score = write_rule_score(case, write_summary)
    privacy_score = privacy_rule_score(case, answer, saved_memory_dump=saved_memory_dump)

    scores = [answer_score["score"]]
    if write_score["score"] is not None:
        scores.append(write_score["score"])
    if privacy_score["score"] is not None:
        scores.append(privacy_score["score"])

    final_score = min(scores) if scores else 0.0
    return {
        "score": final_score,
        "answer": answer_score,
        "write": write_score,
        "privacy": privacy_score,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_category[row.get("category", "unknown")].append(row)

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    overall_no_memory = avg([float(r["scores"]["no_memory"]["score"]) for r in results])
    overall_memory = avg([float(r["scores"]["memory"]["score"]) for r in results])

    categories = {}
    for category, rows in sorted(by_category.items()):
        no_memory = avg([float(r["scores"]["no_memory"]["score"]) for r in rows])
        memory = avg([float(r["scores"]["memory"]["score"]) for r in rows])
        categories[category] = {
            "cases": len(rows),
            "no_memory_score": round(no_memory, 4),
            "memory_score": round(memory, 4),
            "memory_lift": round(memory - no_memory, 4),
        }

    privacy_violations = 0
    write_failures = 0
    for row in results:
        memory_scores = row["scores"]["memory"]
        privacy = memory_scores.get("privacy") or {}
        if privacy.get("correct") is False:
            privacy_violations += 1
        write = memory_scores.get("write") or {}
        if write.get("correct") is False:
            write_failures += 1

    return {
        "cases": len(results),
        "no_memory_score": round(overall_no_memory, 4),
        "memory_score": round(overall_memory, 4),
        "memory_lift": round(overall_memory - overall_no_memory, 4),
        "privacy_violations": privacy_violations,
        "write_failures": write_failures,
        "categories": categories,
    }
