from __future__ import annotations

from omni_memory.consolidation import ExperienceConsolidator
from omni_memory.domain.experience_evaluator import EvaluationResult
from omni_memory.domain.models import ExperienceRecord


class FakeExperienceRepo:
    def __init__(self, experiences: list[ExperienceRecord]) -> None:
        self.experiences = experiences

    def list_experiences(self):
        return list(self.experiences)


class FakeRepo:
    pass


class BothSignalEvaluator:
    def evaluate(self, experience: ExperienceRecord) -> EvaluationResult:
        return EvaluationResult(
            success_score=0.9,
            failure_score=0.9,
            reuse_potential=0.9,
            recommended_memory_type="both",
        )


def _consolidator(experiences: list[ExperienceRecord], evaluator=None) -> ExperienceConsolidator:
    return ExperienceConsolidator(
        experience_repo=FakeExperienceRepo(experiences),
        skill_repo=FakeRepo(),
        failure_pattern_repo=FakeRepo(),
        evaluator=evaluator,
    )


def _experience(
    id_: str,
    *,
    success: bool,
    outcome: str,
    lesson: str,
    confidence: float = 0.91,
) -> ExperienceRecord:
    return ExperienceRecord(
        id=id_,
        goal="Fix CI dependency issue",
        context="CI collection fails because a package import is missing.",
        decision="Remove unnecessary dependency from answer chain" if success else "Add missing dependency blindly",
        actions=["Read failing import", "Change dependency handling"],
        outcome=outcome,
        evaluation={"success": success, "tests": "passed" if success else "failed"},
        lesson=lesson,
        reuse_when=["CI fails during collection", "missing package import"],
        confidence=confidence,
        meta={"domain_ids": ["domain:project:omni-memory"]},
    )


def test_successful_experiences_do_not_create_failure_pattern():
    experiences = [
        _experience("exp-success-1", success=True, outcome="pytest passed after fix one", lesson="Prefer removing unnecessary dependency before adding a heavy package."),
        _experience("exp-success-2", success=True, outcome="pytest passed after fix two", lesson="Prefer removing unnecessary dependency before adding a heavy package."),
    ]

    result = _consolidator(experiences).consolidate(dry_run=True, min_confidence=0.85)

    assert any(proposal.kind == "skill" for proposal in result.proposals)
    assert not any(proposal.kind == "failure_pattern" for proposal in result.proposals)


def test_failure_pattern_requires_distinct_failure_and_fix_experiences():
    mixed_single = _experience(
        "exp-both",
        success=True,
        outcome="pytest failed first, then passed after fix",
        lesson="Remove unnecessary dependency before adding packages.",
    )

    result = _consolidator([mixed_single], evaluator=BothSignalEvaluator()).consolidate(dry_run=True, min_confidence=0.85)

    assert not any(proposal.kind == "failure_pattern" for proposal in result.proposals)


def test_failure_pattern_evidence_ids_are_unique_and_never_self_paired():
    success = _experience(
        "exp-success",
        success=True,
        outcome="pytest passed after fix",
        lesson="Prefer removing unnecessary dependency before adding a heavy package.",
    )
    failure = _experience(
        "exp-failure",
        success=False,
        outcome="pytest failed with regression during collection",
        lesson="Blindly adding dependencies can create CI regressions.",
    )

    result = _consolidator([success, failure]).consolidate(dry_run=True, min_confidence=0.85)
    failure_patterns = [proposal for proposal in result.proposals if proposal.kind == "failure_pattern"]

    assert failure_patterns
    for proposal in failure_patterns:
        assert len(proposal.evidence_ids) == len(set(proposal.evidence_ids))
        assert len(proposal.evidence_ids) >= 2
        assert proposal.evidence_ids == proposal.payload["evidence_ids"]
        assert proposal.evidence_ids == proposal.payload["refs"]["source_experience_ids"]
