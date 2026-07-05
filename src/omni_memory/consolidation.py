from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from omni_memory.domain.experience_evaluator import DomainExperienceEvaluator, EvaluationResult, ExperienceEvaluator
from omni_memory.domain.models import ExperienceRecord, FailurePatternRecord, Provenance, SkillRecord
from omni_memory.domain.writeback import stable_id
from omni_memory.telemetry import span as telemetry_span

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
    reusable SkillRecord and FailurePatternRecord objects. Domain-specific success
    and failure decisions are delegated to an ExperienceEvaluator.
    """

    def __init__(
        self,
        *,
        experience_repo: Any,
        skill_repo: Any,
        failure_pattern_repo: Any,
        evaluator: ExperienceEvaluator | None = None,
    ) -> None:
        self.experience_repo = experience_repo
        self.skill_repo = skill_repo
        self.failure_pattern_repo = failure_pattern_repo
        self.evaluator = evaluator or DomainExperienceEvaluator()

    def consolidate(self, *, dry_run: bool = True, min_confidence: float = 0.85) -> ConsolidationResult:
        with telemetry_span("consolidation.experiences", dry_run=dry_run, min_confidence=min_confidence) as span:
            experiences = [
                item
                for item in self.experience_repo.list_experiences()
                if item.confidence >= min_confidence and _eligible_for_consolidation(item)
            ]
            evaluations = {item.id: self.evaluator.evaluate(item) for item in experiences}
            proposals = [
                *self._skill_proposals(experiences, evaluations),
                *self._failure_pattern_proposals(experiences, evaluations),
            ]
            _set_span_attribute(span, "experience_count", len(experiences))
            _set_span_attribute(span, "proposal_count", len(proposals))

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

    def _skill_proposals(self, experiences: list[ExperienceRecord], evaluations: dict[str, EvaluationResult]) -> list[ConsolidationProposal]:
        buckets: dict[str, list[ExperienceRecord]] = {}
        for item in experiences:
            evaluation = evaluations[item.id]
            if not item.lesson or not item.reuse_when:
                continue
            if evaluation.reuse_potential < 0.5:
                continue
            if not _has_success_signal(item, evaluation):
                continue
            key = _topic_key(item)
            buckets.setdefault(key, []).append(item)

        proposals: list[ConsolidationProposal] = []
        for topic, items in buckets.items():
            items = _distinct_experiences(items)
            if len(items) < 2:
                continue
            evidence_ids = _unique_ids(item.id for item in items)
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
                "meta": {
                    "promoted_from": "experience_consolidation",
                    "topic": topic,
                    "evaluator": "domain_experience_evaluator",
                    **_promotion_scope_meta(items),
                },
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

    def _failure_pattern_proposals(self, experiences: list[ExperienceRecord], evaluations: dict[str, EvaluationResult]) -> list[ConsolidationProposal]:
        failed = [item for item in experiences if _has_failure_signal(item, evaluations[item.id])]
        successful = [item for item in experiences if _has_success_signal(item, evaluations[item.id])]
        proposals: list[ConsolidationProposal] = []

        for failure in failed:
            topic = _topic_key(failure)
            fixes = [
                item
                for item in successful
                if item.id != failure.id and _topic_key(item) == topic and _has_success_signal(item, evaluations[item.id])
            ]
            fixes = _distinct_experiences(fixes)
            if not fixes:
                continue
            fix = max(fixes, key=lambda item: evaluations[item.id].success_score)
            evidence_ids = _unique_ids([failure.id, fix.id])
            if len(evidence_ids) < 2:
                continue
            confidence = min(0.95, (failure.confidence + fix.confidence) / 2)
            payload = {
                "symptom": failure.outcome or failure.goal,
                "root_cause": failure.lesson or failure.decision or failure.context,
                "fix": fix.lesson or fix.outcome,
                "detection": _failure_detection(failure),
                "evidence_ids": evidence_ids,
                "confidence": confidence,
                "refs": {"source_experience_ids": evidence_ids},
                "meta": {
                    "promoted_from": "experience_consolidation",
                    "topic": topic,
                    "evaluator": "domain_experience_evaluator",
                    **_promotion_scope_meta([failure, fix]),
                },
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


def _set_span_attribute(span: Any | None, key: str, value: Any) -> None:
    if span is not None and hasattr(span, "set_attribute"):
        span.set_attribute(key, value)


def _eligible_for_consolidation(item: ExperienceRecord) -> bool:
    meta = dict(item.meta or {})
    scope = _scope(item)
    if bool(meta.get("exclude_from_consolidation") or scope.get("exclude_from_consolidation")):
        return False
    if str(scope.get("durability") or "").lower() in {"ephemeral", "session"}:
        return False
    if str(scope.get("environment") or "").lower() in {"test", "benchmark", "sandbox"}:
        return False
    return True


def _promotion_scope_meta(items: list[ExperienceRecord]) -> dict[str, Any]:
    scopes = [_scope(item) for item in items]
    domain_ids = _unique([domain_id for scope in scopes for domain_id in _scope_domain_ids(scope)])
    environments = {str(scope.get("environment") or "") for scope in scopes if scope.get("environment")}
    visibilities = {str(scope.get("visibility") or "") for scope in scopes if scope.get("visibility")}
    source_domains = _unique([str(item.meta.get("domain")) for item in items if item.meta.get("domain")])
    scope = {
        "tenant_id": next((str(scope.get("tenant_id")) for scope in scopes if scope.get("tenant_id")), "default"),
        "agent_id": next((scope.get("agent_id") for scope in scopes if scope.get("agent_id")), None),
        "domain_ids": domain_ids,
        "environment": environments.pop() if len(environments) == 1 else "dev",
        "durability": "durable",
        "visibility": visibilities.pop() if len(visibilities) == 1 else "private",
        "exclude_from_consolidation": False,
    }
    out: dict[str, Any] = {"scope": scope, "source_experience_scopes": scopes}
    if len(source_domains) == 1:
        out["domain"] = source_domains[0]
    elif source_domains:
        out["domains"] = source_domains
    return out


def _scope(item: ExperienceRecord) -> dict[str, Any]:
    scope = (item.meta or {}).get("scope") or {}
    if hasattr(scope, "model_dump"):
        return scope.model_dump(mode="json")
    return dict(scope) if isinstance(scope, dict) else {}


def _scope_domain_ids(scope: dict[str, Any]) -> list[str]:
    raw = scope.get("domain_ids") or []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw if str(item)]


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


def _unique_ids(values: list[str] | Any) -> list[str]:
    return _unique([str(value) for value in values if str(value)])


def _distinct_experiences(items: list[ExperienceRecord]) -> list[ExperienceRecord]:
    out: list[ExperienceRecord] = []
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            continue
        seen.add(item.id)
        out.append(item)
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


def _has_failure_signal(item: ExperienceRecord, evaluation: EvaluationResult) -> bool:
    return evaluation.failure_score >= 0.6 or evaluation.recommended_memory_type in {"failure_pattern", "both"}


def _has_success_signal(item: ExperienceRecord, evaluation: EvaluationResult) -> bool:
    return evaluation.success_score >= 0.6 or evaluation.recommended_memory_type in {"skill", "both"}


def _failure_detection(item: ExperienceRecord) -> str:
    tests = item.evaluation.get("tests")
    if tests:
        return f"Tests: {tests}"
    validation = item.evaluation.get("validation") or item.meta.get("validation")
    if validation:
        return f"Validation: {validation}"
    return "Look for the same symptom in outcome, validation metrics, test output or CI logs."


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
