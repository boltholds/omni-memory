from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


class DecisionCandidate(BaseModel):
    """Review-only ADR candidate produced from a development cycle."""

    title: str
    decision: str
    context: str = ""
    consequences: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    refs: dict[str, Any] = Field(default_factory=dict)
    status: str = "proposed"
    confidence: float = 0.7
    reason: str = "architecture_change_detected"
    meta: dict[str, Any] = Field(default_factory=dict)


_ARCHITECTURE_TERMS = {
    "architecture",
    "architectural",
    "adapter",
    "adapters",
    "api",
    "boundary",
    "command",
    "consolidation",
    "contract",
    "decision",
    "domain",
    "evaluator",
    "facade",
    "fastmcp",
    "interface",
    "mcp",
    "policy",
    "protocol",
    "registry",
    "repository",
    "retrieval",
    "retriever",
    "router",
    "schema",
    "scope",
    "sdk",
    "service",
    "workflow",
}

_ACTION_TERMS = {
    "add",
    "added",
    "centralize",
    "centralized",
    "change",
    "changed",
    "choose",
    "decide",
    "decided",
    "extract",
    "extracted",
    "generalize",
    "generalized",
    "introduce",
    "introduced",
    "move",
    "moved",
    "replace",
    "replaced",
    "refactor",
    "refactored",
    "route",
    "routed",
    "split",
    "switched",
    "use",
    "wire",
    "wired",
}

_ARCHITECTURAL_PATH_HINTS = (
    "app/integrations/",
    "app/mcp",
    "app/memory",
    "app/retriever",
    "app/consolidation",
    "app/development_memory_workflow",
    "domain/",
    "infra/repo/",
    "infra/llm/",
    "infra/distillers/",
)


def draft_decision_candidates(request: Any) -> list[DecisionCandidate]:
    """Return review-only decision candidates for meaningful design changes.

    This intentionally never writes to the decision repository. It only surfaces
    proposed ADR-like records next to the development experience so an agent or
    human can explicitly accept, edit, or ignore them.
    """

    if not _looks_like_architectural_change(request):
        return []

    title = _title(request)
    decision = _decision(request)
    if not title or not decision:
        return []

    return [
        DecisionCandidate(
            title=title,
            decision=decision,
            context=_context(request),
            consequences=_consequences(request),
            alternatives=_alternatives(request),
            refs={
                "files": list(getattr(request, "changed_files", []) or []),
                "commands_run": list(getattr(request, "commands_run", []) or []),
                "tests": list(getattr(request, "tests", []) or []),
            },
            status="proposed",
            confidence=_confidence(request),
            reason="development_cycle_contains_architectural_or_api_decision",
            meta={
                "drafted_from": "development_memory_workflow",
                "auto_draft": True,
                "review_required": True,
            },
        )
    ]


def _looks_like_architectural_change(request: Any) -> bool:
    text = _combined_text(request)
    terms = set(_terms(text))
    has_architecture_term = bool(terms.intersection(_ARCHITECTURE_TERMS))
    has_action_term = bool(terms.intersection(_ACTION_TERMS))
    has_decision = bool(getattr(request, "decisions", []) or [])
    has_path_hint = any(_path_is_architectural(path) for path in getattr(request, "changed_files", []) or [])

    if has_decision and (has_architecture_term or has_path_hint):
        return True
    if has_architecture_term and has_action_term and has_path_hint:
        return True
    if has_decision and has_action_term and has_path_hint:
        return True
    return False


def _title(request: Any) -> str:
    decision = _first_nonempty(getattr(request, "decisions", []) or [])
    goal = str(getattr(request, "goal", "") or "").strip()
    summary = str(getattr(request, "summary", "") or "").strip()
    base = decision or goal or summary
    base = _clean_sentence(base)
    if not base:
        return "Proposed development decision"
    if len(base) > 96:
        base = base[:93].rstrip() + "..."
    return base[0].upper() + base[1:]


def _decision(request: Any) -> str:
    decisions = [_clean_sentence(item) for item in getattr(request, "decisions", []) or [] if _clean_sentence(item)]
    if decisions:
        return " ".join(decisions)
    lesson = _clean_sentence(str(getattr(request, "lesson", "") or ""))
    if lesson:
        return lesson
    return _clean_sentence(str(getattr(request, "outcome", "") or ""))


def _context(request: Any) -> str:
    parts: list[str] = []
    for label, value in [
        ("Goal", getattr(request, "goal", "")),
        ("Summary", getattr(request, "summary", "")),
        ("Outcome", getattr(request, "outcome", "")),
        ("Lesson", getattr(request, "lesson", "")),
    ]:
        cleaned = _clean_sentence(str(value or ""))
        if cleaned:
            parts.append(f"{label}: {cleaned}")
    files = list(getattr(request, "changed_files", []) or [])
    if files:
        parts.append("Changed files: " + ", ".join(files[:12]))
    return "\n".join(parts)


def _consequences(request: Any) -> list[str]:
    out: list[str] = []
    files = list(getattr(request, "changed_files", []) or [])
    if files:
        out.append("Affected implementation files: " + ", ".join(files[:8]))
    tests = list(getattr(request, "tests", []) or [])
    if tests:
        out.append("Validated by: " + "; ".join(tests[:4]))
    lesson = _clean_sentence(str(getattr(request, "lesson", "") or ""))
    if lesson:
        out.append("Reusable rationale: " + lesson)
    return _unique(out)


def _alternatives(request: Any) -> list[str]:
    avoid_when = [_clean_sentence(item) for item in getattr(request, "avoid_when", []) or [] if _clean_sentence(item)]
    if avoid_when:
        return avoid_when
    text = _combined_text(request)
    if "instead" in text.casefold() or "replace" in text.casefold() or "replaced" in text.casefold():
        return ["Keep the previous implementation approach."]
    return []


def _confidence(request: Any) -> float:
    confidence = float(getattr(request, "confidence", 0.7) or 0.7)
    bonus = 0.0
    if getattr(request, "decisions", []) or []:
        bonus += 0.08
    if getattr(request, "tests", []) or []:
        bonus += 0.05
    if any(_path_is_architectural(path) for path in getattr(request, "changed_files", []) or []):
        bonus += 0.05
    return min(0.95, max(0.5, confidence + bonus))


def _combined_text(request: Any) -> str:
    values: list[str] = [
        str(getattr(request, "goal", "") or ""),
        str(getattr(request, "summary", "") or ""),
        str(getattr(request, "outcome", "") or ""),
        str(getattr(request, "lesson", "") or ""),
        " ".join(str(item) for item in getattr(request, "decisions", []) or []),
        " ".join(str(item) for item in getattr(request, "changed_files", []) or []),
    ]
    return " ".join(values)


def _path_is_architectural(path: str) -> bool:
    normalized = str(path or "").replace("\\", "/").casefold()
    return any(normalized.startswith(prefix) for prefix in _ARCHITECTURAL_PATH_HINTS)


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-Я0-9_\-]{3,}", str(text or "").casefold())


def _first_nonempty(values: list[Any]) -> str:
    for value in values:
        text = _clean_sentence(str(value or ""))
        if text:
            return text
    return ""


def _clean_sentence(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _clean_sentence(value)
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)
    return out
