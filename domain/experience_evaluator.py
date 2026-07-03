from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from domain.models import ExperienceRecord

RecommendedMemoryType = Literal["skill", "failure_pattern", "both", "none"]


class EvaluationResult(BaseModel):
    success_score: float = 0.5
    failure_score: float = 0.0
    risk_score: float = 0.0
    reuse_potential: float = 0.5
    consolidation_tags: list[str] = Field(default_factory=list)
    recommended_memory_type: RecommendedMemoryType = "none"
    meta: dict[str, Any] = Field(default_factory=dict)


class ExperienceEvaluator(Protocol):
    def evaluate(self, experience: ExperienceRecord) -> EvaluationResult:
        ...


class DevelopmentExperienceEvaluator:
    """Deterministic evaluator for development/coding experience records."""

    def evaluate(self, experience: ExperienceRecord) -> EvaluationResult:
        success_text = _success_text(experience)
        failure_text = _failure_text(experience)
        explicit_success = experience.evaluation.get("success")

        success = _contains_any(success_text, ["success", "passed", "fixed", "works", "green", "зел", "успеш"])
        failure = _contains_any(failure_text, ["fail", "failed", "error", "regression", "broken", "exception", "ошиб", "пад"])
        if explicit_success is True:
            success = True
            failure = False
        elif explicit_success is False:
            success = False
            failure = True

        success_score = 0.9 if success else 0.35
        failure_score = 0.85 if failure else 0.05
        reuse_potential = 0.8 if experience.lesson and experience.reuse_when else 0.35
        recommended: RecommendedMemoryType = "none"
        if success and failure:
            recommended = "both"
        elif success:
            recommended = "skill"
        elif failure:
            recommended = "failure_pattern"

        tags = ["development"]
        if success:
            tags.append("successful")
        if failure:
            tags.append("failure")

        return EvaluationResult(
            success_score=success_score,
            failure_score=failure_score,
            risk_score=0.7 if failure else 0.1,
            reuse_potential=reuse_potential,
            consolidation_tags=tags,
            recommended_memory_type=recommended,
            meta={"deterministic": True, "domain": "development"},
        )


class OpsExperienceEvaluator:
    """Deterministic evaluator for operations and incident-response cycles."""

    def evaluate(self, experience: ExperienceRecord) -> EvaluationResult:
        validation = _validation(experience)
        text = _experience_text(experience)
        restored = bool(validation.get("sla_restored"))
        missed = bool(validation.get("sla_missed"))
        improved_score = _metric_improvement_score(
            _dict_float(validation.get("metrics_before")),
            _dict_float(validation.get("metrics_after")),
        )
        unresolved = _contains_any(text, ["unresolved", "escalated", "still failing", "sla missed", "not restored"])
        resolved = restored or improved_score >= 0.6 or _contains_any(text, ["resolved", "restored", "recovered", "returned to baseline"])
        failed = missed or unresolved or improved_score < -0.2

        success_score = 0.9 if resolved else max(0.2, 0.5 + improved_score / 2)
        failure_score = 0.85 if failed else (0.35 if restored and resolved else 0.1)
        risk_score = 0.8 if failed else (0.5 if restored and resolved else 0.2)
        reuse_potential = 0.85 if experience.lesson and experience.reuse_when else 0.45
        recommended: RecommendedMemoryType = "none"
        if resolved and failed:
            recommended = "both"
        elif resolved:
            recommended = "skill"
        elif failed:
            recommended = "failure_pattern"

        tags = ["ops"]
        service = validation.get("service") or experience.meta.get("service")
        if service:
            tags.append(f"service:{service}")
        if restored:
            tags.append("sla_restored")
        if failed:
            tags.append("incident")

        return EvaluationResult(
            success_score=min(0.99, max(0.0, success_score)),
            failure_score=min(0.99, max(0.0, failure_score)),
            risk_score=min(0.99, max(0.0, risk_score)),
            reuse_potential=reuse_potential,
            consolidation_tags=tags,
            recommended_memory_type=recommended,
            meta={"deterministic": True, "domain": "ops", "metric_improvement_score": improved_score},
        )


class GenericExperienceEvaluator(DevelopmentExperienceEvaluator):
    """Backward-compatible default evaluator using text and explicit success flags."""

    def evaluate(self, experience: ExperienceRecord) -> EvaluationResult:
        result = super().evaluate(experience)
        result.consolidation_tags = [tag for tag in result.consolidation_tags if tag != "development"] or ["generic"]
        result.meta["domain"] = "generic"
        return result


class DomainExperienceEvaluator:
    def __init__(self, evaluators: dict[str, ExperienceEvaluator] | None = None, default: ExperienceEvaluator | None = None) -> None:
        self.evaluators: dict[str, ExperienceEvaluator] = {
            "development": DevelopmentExperienceEvaluator(),
            "ops": OpsExperienceEvaluator(),
        }
        if evaluators:
            self.evaluators.update(evaluators)
        self.default = default or GenericExperienceEvaluator()

    def evaluate(self, experience: ExperienceRecord) -> EvaluationResult:
        domain = _experience_domain(experience)
        evaluator = self.evaluators.get(domain, self.default)
        result = evaluator.evaluate(experience)
        result.meta.setdefault("routed_domain", domain)
        result.meta.setdefault("evaluator", evaluator.__class__.__name__)
        return result


def _experience_domain(experience: ExperienceRecord) -> str:
    meta = dict(experience.meta or {})
    value = meta.get("domain")
    if value:
        return str(value)
    scope = meta.get("scope") or {}
    if isinstance(scope, dict) and scope.get("domain"):
        return str(scope["domain"])
    return "generic"


def _experience_text(experience: ExperienceRecord) -> str:
    return " ".join(
        [
            experience.goal,
            experience.context,
            experience.decision,
            " ".join(experience.actions),
            experience.outcome,
            str(experience.evaluation),
            experience.lesson,
        ]
    ).casefold()


def _success_text(experience: ExperienceRecord) -> str:
    return " ".join(
        [
            experience.decision,
            " ".join(experience.actions),
            experience.outcome,
            str(experience.evaluation),
            experience.lesson,
        ]
    ).casefold()


def _failure_text(experience: ExperienceRecord) -> str:
    return " ".join(
        [
            experience.outcome,
            str(experience.evaluation),
        ]
    ).casefold()


def _contains_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)


def _validation(experience: ExperienceRecord) -> dict[str, Any]:
    raw = experience.evaluation.get("validation") or experience.meta.get("validation") or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _dict_float(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def _metric_improvement_score(before: dict[str, float], after: dict[str, float]) -> float:
    if not before or not after:
        return 0.0
    scores: list[float] = []
    for key, before_value in before.items():
        if key not in after or before_value == 0:
            continue
        after_value = after[key]
        delta = (before_value - after_value) / abs(before_value) if _lower_is_better(key) else (after_value - before_value) / abs(before_value)
        scores.append(max(-1.0, min(1.0, delta)))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _lower_is_better(metric_name: str) -> bool:
    key = metric_name.casefold()
    return any(token in key for token in ["latency", "error", "failure", "fail", "cost", "duration", "queue", "saturation", "cpu", "memory"])
