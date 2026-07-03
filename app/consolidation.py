from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from domain.models import ExperienceRecord, FailurePatternRecord, Provenance, SkillRecord
from domain.writeback import stable_id

ConsolidationKind = Literal["skill", "failure_pattern"]


@dataclass(frozen=True)
class ConsolidationProposal:
    kind: ConsolidationKind
    title: str
    evidence_ids: list[str]
    confidence: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class ConsolidationResult:
    dry_run: bool
    proposals: list[ConsolidationProposal] = field(default_factory=list)
    saved_skills: list[SkillRecord] = field(default_factory=list)
    saved_failure_patterns: list[FailurePatternRecord] = field(default_factory=list)

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "proposals": [_proposal_to_dict(item) for item in self.proposals],
            "saved_skills": [item.model_dump(mode=mode) for item in self.saved_skills],
            "saved_failure_patterns": [item.model_dump(mode=mode) for item in self.saved_failure_patterns],
        }


class ExperienceConsolidator:
    """Deterministic MVP consolidator.

    It does not run in the write path. It inspects stored experiences and proposes
    reusable SkillRecord and FailurePatternRecord objects. Callers can run it in
    dry-run mode first, then apply the same rules once the proposals look useful.
    """

    def __init__(
        self,
        *,
        experience_repo: Any,
        skill_repo: Any,
        failure_pattern_repo: Any,
    ) -> None:
        self.experience_repo = experience_repo
        self.skill_repo = skill_repo
        self.failure_pattern_repo = failure_pattern_repo

    def consolidate(self, *, dry_run: bool = True, min_confidence: float = 0.85) -> ConsolidationResult:
        experiences = [
            item
            for item in self.experience_repo.list_experiences()
            if item.confidence >= min_confidence
        ]
        proposals = [
            *self._skill_proposals(experiences),
            *self._failure_pattern_proposals(experiences),
        ]

        if dry_run:
            return ConsolidationResult(dry_run=True, proposals=proposals)

        saved_skills: list[SkillRecord] = []
        saved_failure_patterns: list[FailurePatternRecord] = []
        now = time.time()

        for proposal in proposals:
            if proposal.kind == "skill":
                skill = SkillRecord(
                    id=stable_id("skill", proposal.payload),
                    provenance=Provenance(source="consolidation", time=now),
                    **proposal.payload,
                )
                self.skill_repo.save_skill(skill)
                saved_skills.append(skill)
            elif proposal.kind == "failure_pattern":
                pattern = FailurePatternRecord(
                    id=stable_id("failure_pattern", proposal.payload),
                    provenance=Provenance(source="consolidation", time=now),
                    **proposal.payload,
                )
                self.failure_pattern_repo.save_failure_pattern(pattern)
                saved_failure_patterns.append(pattern)

        return ConsolidationResult(
            dry_run=False,
            proposals=proposals,
            saved_skills=saved_skills,
            saved_failure_patterns=saved_failure_patterns,
        )

    def _skill_proposals(self, experiences: list[ExperienceRecord]) -> list[ConsolidationProposal]:
        buckets: dict[str, list[ExperienceRecord]] = {}
        for item in experiences:
            if not item.lesson or not item.reuse_when:
                continue
            key = _topic_key(item)
            buckets.setdefault(key, []).append(item)

        proposals: list[ConsolidationProposal] = []
        for topic, items in buckets.items():
            if len(items) < 2:
                continue
            evidence_ids = [item.id for item in items]
            confidence = min(0.99, sum(item.confidence for item in items) / len(items))
            primary = max(items, key=lambda item: item.confidence)
            payload = {
                "name": _skill_name(topic, primary),
                "problem": primary.context or primary.goal,
                "procedure": _skill_procedure(items),
                "reuse_when": _unique([value for item in items for value in item.reuse_when]),
                "avoid_when": _unique([value for item in items for value in item.avoid_when]),
                "evidence_ids": evidence_ids,
                "confidence": confidence,
                "refs": {"source_experience_ids": evidence_ids},
                "meta": {"promoted_from": "experience_consolidation", "topic": topic},
            }
            proposals.append(
                ConsolidationProposal(
                    kind="skill",
                    title=payload["name"],
                    evidence_ids=evidence_ids,
                    confidence=confidence,
                    payload=payload,
                )
            )
        return proposals

    def _failure_pattern_proposals(self, experiences: list[ExperienceRecord]) -> list[ConsolidationProposal]:
        failed = [item for item in experiences if _looks_failed(item)]
        successful = [item for item in experiences if _looks_successful(item)]
        proposals: list[ConsolidationProposal] = []

        for failure in failed:
            topic = _topic_key(failure)
            fixes = [item for item in successful if _topic_key(item) == topic]
            if not fixes:
                continue
            fix = max(fixes, key=lambda item: item.confidence)
            evidence_ids = [failure.id, fix.id]
            confidence = min(0.95, (failure.confidence + fix.confidence) / 2)
            payload = {
                "symptom": failure.outcome or failure.goal,
                "root_cause": failure.lesson or failure.decision or failure.context,
                "fix": fix.lesson or fix.outcome,
                "detection": _failure_detection(failure),
                "evidence_ids": evidence_ids,
                "confidence": confidence,
                "refs": {"source_experience_ids": evidence_ids},
                "meta": {"promoted_from": "experience_consolidation", "topic": topic},
            }
            proposals.append(
                ConsolidationProposal(
                    kind="failure_pattern",
                    title=payload["symptom"],
                    evidence_ids=evidence_ids,
                    confidence=confidence,
                    payload=payload,
                )
            )
        return proposals


def _proposal_to_dict(item: ConsolidationProposal) -> dict[str, Any]:
    return {
        "kind": item.kind,
        "title": item.title,
        "evidence_ids": item.evidence_ids,
        "confidence": item.confidence,
        "payload": item.payload,
    }


def _topic_key(item: ExperienceRecord) -> str:
    text = " ".join([item.goal, item.context, item.decision, item.lesson, *item.reuse_when])
    terms = [term for term in _terms(text) if term not in _STOP_TERMS]
    return "_".join(terms[:4]) or "general"


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-Я0-9_\-]{4,}", str(text or "").casefold())


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)
    return out


def _skill_name(topic: str, primary: ExperienceRecord) -> str:
    if primary.decision:
        return primary.decision[:120]
    if primary.lesson:
        return primary.lesson[:120]
    return topic.replace("_", " ").title()


def _skill_procedure(items: list[ExperienceRecord]) -> list[str]:
    actions = _unique([action for item in items for action in item.actions])
    if actions:
        return actions[:8]
    lessons = _unique([item.lesson for item in items if item.lesson])
    return lessons[:5]


def _looks_failed(item: ExperienceRecord) -> bool:
    text = " ".join([item.outcome, item.lesson, str(item.evaluation)]).casefold()
    return any(marker in text for marker in ["fail", "failed", "error", "regression", "broken", "ошиб", "пад"])


def _looks_successful(item: ExperienceRecord) -> bool:
    if item.evaluation.get("success") is True:
        return True
    text = " ".join([item.outcome, item.lesson, str(item.evaluation)]).casefold()
    return any(marker in text for marker in ["success", "passed", "fixed", "works", "зел", "успеш"])


def _failure_detection(item: ExperienceRecord) -> str:
    tests = item.evaluation.get("tests")
    if tests:
        return f"Tests: {tests}"
    return "Look for the same symptom in outcome, test output or CI logs."


_STOP_TERMS = {
    "with",
    "when",
    "from",
    "that",
    "this",
    "into",
    "для",
    "если",
    "после",
    "надо",
    "нужно",
}
