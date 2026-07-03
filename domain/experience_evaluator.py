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
        text = _experience_text(experience)
        success = _contains_any(text, ["success", "passed", "fixed", "works", "green", "зел", "успеш"])
        failure = _contains_any(text, ["fail", "failed", "error", "regression", "broken", "exception", "ошиб", "пад"])
        if experience.evaluation.get("success") is True:
            success = True
        if experience.evaluation.get("success") is False:
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


class GenericExperienceEvaluator(DevelopmentExperienceEvaluator):
    """Backward-compatible default evaluator using text and explicit success flags."""

    def evaluate(self, experience: ExperienceRecord) -> EvaluationResult:
        result = super().evaluate(experience)
        result.consolidation_tags = [tag for tag in result.consolidation_tags if tag != "development"] or ["generic"]
        result.meta["domain"] = "generic"
        return result


class DomainExperienceEvaluator:
    def __init__(self, evaluators: dict[str, ExperienceEvaluator] | None = None, default: ExperienceEvaluator | None = None) -> None:
        self.evaluators: dict[str, ExperienceEvaluator] = {"development": DevelopmentExperienceEvaluator()}
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


def _contains_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)
