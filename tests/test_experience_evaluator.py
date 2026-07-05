from __future__ import annotations

from omni_memory.builder import build_memory
from omni_memory.domain.experience_evaluator import DomainExperienceEvaluator, EvaluationResult
from omni_memory.domain.models import ExperienceRecord
from omni_memory.infra.embeddings.factory import HashEmbedder


class OpsLikeEvaluator:
    def evaluate(self, experience: ExperienceRecord) -> EvaluationResult:
        validation = experience.evaluation.get("validation") or {}
        restored = bool(validation.get("sla_restored"))
        return EvaluationResult(
            success_score=0.9 if restored else 0.2,
            failure_score=0.1 if restored else 0.85,
            risk_score=0.2 if restored else 0.8,
            reuse_potential=0.8,
            consolidation_tags=["ops", "sla"],
            recommended_memory_type="skill" if restored else "failure_pattern",
        )


def test_domain_experience_evaluator_routes_by_experience_domain():
    evaluator = DomainExperienceEvaluator(evaluators={"ops": OpsLikeEvaluator()})
    experience = ExperienceRecord(
        id="exp-ops",
        goal="Restore API latency SLA",
        outcome="Latency returned to baseline",
        evaluation={"validation": {"sla_restored": True}},
        lesson="Rollback bad cache config when p95 spikes after deploy.",
        reuse_when=["p95 latency spike after deploy"],
        confidence=0.9,
        meta={"domain": "ops"},
    )

    result = evaluator.evaluate(experience)

    assert result.success_score == 0.9
    assert result.recommended_memory_type == "skill"
    assert result.meta["routed_domain"] == "ops"
    assert result.meta["evaluator"] == "OpsLikeEvaluator"
    assert "ops" in result.consolidation_tags


def test_domain_experience_evaluator_registers_development_by_default():
    evaluator = DomainExperienceEvaluator()
    experience = ExperienceRecord(
        id="exp-dev",
        goal="Fix CI dependency issue",
        outcome="pytest passed after fix",
        evaluation={"tests": "passed"},
        lesson="Remove unnecessary dependency before adding packages.",
        reuse_when=["CI dependency issue"],
        confidence=0.9,
        meta={"domain": "development"},
    )

    result = evaluator.evaluate(experience)

    assert result.recommended_memory_type == "skill"
    assert result.meta["routed_domain"] == "development"
    assert result.meta["evaluator"] == "DevelopmentExperienceEvaluator"
    assert "development" in result.consolidation_tags


def test_consolidation_uses_injected_evaluator_from_memory_builder():
    memory = build_memory(
        use_llm=False,
        embedder=HashEmbedder(),
        experience_evaluator=DomainExperienceEvaluator(evaluators={"ops": OpsLikeEvaluator()}),
    )

    for suffix in ["one", "two"]:
        memory.record_experience(
            goal="Restore API latency SLA",
            context="p95 latency increased after cache deployment",
            actions=["Inspect traces", "Rollback cache config"],
            outcome=f"Incident resolved {suffix}",
            evaluation={"validation": {"sla_restored": True, "latency_p95_after_ms": 280}},
            lesson="Rollback cache configuration when p95 latency spikes after deploy.",
            reuse_when=["p95 latency spike after deploy"],
            confidence=0.91,
            source="codex-dev",
            meta={"domain": "ops", "domain_ids": ["domain:ops:api"]},
        )

    result = memory.consolidate_experiences(dry_run=True, min_confidence=0.85)

    assert any(proposal.kind == "skill" for proposal in result.proposals)
    assert result.proposals[0].payload["meta"]["domain"] == "ops"
