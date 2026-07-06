from __future__ import annotations

from omni_memory import (
    RecordExperienceRequest,
    WriteDecisionRequest,
    WriteFailurePatternRequest,
    WriteSkillRequest,
    build_memory,
)
from omni_memory.infra.embeddings.factory import HashEmbedder


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


def test_decision_request_records_a_searchable_decision():
    memory = _memory()

    report = memory.write_decision(
        WriteDecisionRequest(
            title="Use request dataclasses",
            decision="Represent wide write calls as request objects.",
            consequences=["Memory API is easier for agents to compose."],
            source="test",
        )
    )

    assert report.saved == 1
    decisions = memory.list_decisions()
    assert len(decisions) == 1
    assert decisions[0].title == "Use request dataclasses"
    assert decisions[0].consequences == ["Memory API is easier for agents to compose."]


def test_experience_request_records_retrievable_lesson():
    memory = _memory()

    report = memory.record_experience(
        RecordExperienceRequest(
            goal="Reduce parameter soup",
            decision="Keep kwargs facade and add request dataclasses.",
            actions=["Added request object API"],
            outcome="Callers can pass one domain object.",
            lesson="Use request dataclasses for wide memory operations.",
            reuse_when=["adding semantic write operations"],
            confidence=0.9,
            source="test",
        )
    )

    assert report.saved == 1
    results = memory.search_experiences("semantic write operations", k=3)
    assert [item.lesson for item in results] == ["Use request dataclasses for wide memory operations."]


def test_skill_and_failure_pattern_requests_save_cognitive_records():
    memory = _memory()

    skill_report = memory.write_skill(
        WriteSkillRequest(
            name="Request-object write API",
            problem="Wide function signatures make agent calls brittle.",
            procedure=["Create a dataclass", "Pass it to a request method"],
            reuse_when=["function has many cohesive parameters"],
            confidence=0.85,
            source="test",
        )
    )
    pattern_report = memory.write_failure_pattern(
        WriteFailurePatternRequest(
            symptom="A write method keeps gaining keyword-only parameters.",
            root_cause="The method is modeling a request without naming it.",
            fix="Introduce a request dataclass and delegate old kwargs API to it.",
            detection="Signature contains several cohesive optional fields.",
            confidence=0.8,
            source="test",
        )
    )

    assert skill_report.saved == 1
    assert pattern_report.saved == 1
    assert memory.repositories.skill.search("wide signatures", k=1)[0].name == "Request-object write API"
    assert memory.repositories.failure_pattern.search("keyword-only parameters", k=1)[0].fix == "Introduce a request dataclass and delegate old kwargs API to it."


def test_short_write_api_accepts_request_object_only():
    memory = _memory()

    report = memory.write_skill(
        WriteSkillRequest(
            name="Request-only skill write",
            problem="Callers pass one cohesive request object.",
            procedure=["Construct request", "Call write_skill(request)"],
            source="test",
        )
    )

    assert report.saved == 1
    assert memory.repositories.skill.search("cohesive request object", k=1)[0].name == "Request-only skill write"
