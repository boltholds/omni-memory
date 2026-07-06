from __future__ import annotations

from collections import defaultdict
from typing import Any

from benchmarks.dev_agent.patch_runner import evaluate_patch_contract
from benchmarks.dev_agent.repo_runner import evaluate_repo_contract


def normalize(value: Any) -> str:
    return str(value or "").casefold()


def parse_agent_json(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = value.removeprefix("```json").removeprefix("```").strip()
        value = value.removesuffix("```").strip()
    try:
        import json

        parsed = json.loads(value)
    except Exception:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end < start:
            return {"action": value}
        try:
            parsed = json.loads(value[start : end + 1])
        except Exception:
            return {"action": value}
    return parsed if isinstance(parsed, dict) else {"action": value}


def judge_action(case: dict[str, Any], mode: str, answer: str, *, context_relevant: bool, repo_eval: bool = False) -> dict[str, Any]:
    parsed = parse_agent_json(answer)
    patch_result = evaluate_patch_contract(case, parsed)
    repo_result = evaluate_repo_contract(case, parsed) if repo_eval else {"applicable": False, "passed": None, "errors": [], "stdout": "", "stderr": ""}
    action_text = normalize(parsed.get("action", answer))
    answer_text = normalize(
        " ".join(
            str(parsed.get(key, ""))
            for key in ("memory_check", "action", "rationale")
        )
    )
    expected_terms = [normalize(item) for item in case.get("expected_terms", [])]
    forbidden_terms = [normalize(item) for item in case.get("forbidden_terms", [])]
    memory_terms = [normalize(item) for item in case.get("memory_terms", [])]

    missing_expected = [item for item in expected_terms if item and item not in answer_text]
    forbidden_hits = [item for item in forbidden_terms if item and item in action_text]
    memory_used = mode != "no_memory" and any(term and term in answer_text for term in memory_terms)
    patch_passed = patch_result["passed"] is not False
    repo_passed = repo_result["passed"] is not False
    tests_passed = not missing_expected and not forbidden_hits and patch_passed and repo_passed
    repeated_old_mistake = not tests_passed
    helped_avoid_repeat = mode == "omni_memory" and context_relevant and tests_passed and not repeated_old_mistake

    return {
        "tests_passed": tests_passed,
        "repeated_old_mistake": repeated_old_mistake,
        "memory_used": memory_used,
        "context_relevant": context_relevant,
        "helped_avoid_repeat": helped_avoid_repeat,
        "missing_expected": missing_expected,
        "forbidden_hits": forbidden_hits,
        "score": 1.0 if tests_passed else 0.0,
        "patch": patch_result,
        "repo": repo_result,
        "parsed": parsed,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)

    modes = {mode: _mode_summary(items) for mode, items in sorted(by_mode.items())}
    no_memory = modes.get("no_memory", {})
    omni = modes.get("omni_memory", {})
    rag = modes.get("rag_only", {})

    baseline_repeat = int(no_memory.get("repeat_failures", 0) or 0)
    omni_repeat = int(omni.get("repeat_failures", 0) or 0)
    rag_repeat = int(rag.get("repeat_failures", 0) or 0)

    return {
        "cases": len({row["case_id"] for row in rows}),
        "runs": len(rows),
        "modes": modes,
        "memory_lift": {
            "pass_rate_delta_vs_no_memory": _delta(omni, no_memory, "pass_rate"),
            "pass_rate_delta_vs_rag_only": _delta(omni, rag, "pass_rate"),
            "repeat_failure_reduction_vs_no_memory": _reduction(baseline_repeat, omni_repeat),
            "repeat_failure_reduction_vs_rag_only": _reduction(rag_repeat, omni_repeat),
        },
    }


def _mode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    passed = sum(1 for row in rows if row["judgement"]["tests_passed"])
    repeat_failures = sum(1 for row in rows if row["judgement"]["repeated_old_mistake"])
    memory_used = sum(1 for row in rows if row["judgement"]["memory_used"])
    relevant = sum(1 for row in rows if row["judgement"]["context_relevant"])
    helped = sum(1 for row in rows if row["judgement"]["helped_avoid_repeat"])
    return {
        "cases": count,
        "tests_passed": passed,
        "pass_rate": round(passed / count, 4) if count else 0.0,
        "repeat_failures": repeat_failures,
        "repeat_failure_rate": round(repeat_failures / count, 4) if count else 0.0,
        "memory_used": memory_used,
        "relevant_memory_used": relevant,
        "helped_avoid_repeat": helped,
        "avg_score": round(sum(float(row["judgement"]["score"]) for row in rows) / count, 4) if count else 0.0,
    }


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    if not left or not right:
        return None
    return round(float(left.get(key, 0.0)) - float(right.get(key, 0.0)), 4)


def _reduction(baseline: int, candidate: int) -> float | None:
    if baseline <= 0:
        return None
    return round((baseline - candidate) / baseline, 4)
