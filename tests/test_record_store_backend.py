from __future__ import annotations

from pathlib import Path

from omni_memory import build_memory
from omni_memory.domain.models import (
    DecisionRecord,
    ExperienceRecord,
    FailurePatternRecord,
    Provenance,
    ReviewItem,
    SkillRecord,
)
from omni_memory.domain.requests import RecordExperienceRequest, WriteDecisionRequest
from omni_memory.infra.record_store import InMemoryRecordStoreBackend, JsonRecordStoreBackend, RecordStoreBackends
from omni_memory.infra.repo.cognitive_repo import FailurePatternRepo, PersistentSkillRepo, SkillRepo
from omni_memory.infra.repo.decision_repo import DecisionRepo, PersistentDecisionRepo
from omni_memory.infra.repo.experience_repo import ExperienceRepo
from omni_memory.infra.repo.review_repo import ReviewQueueRepo
from omni_memory.memory_repositories import build_memory_repositories


class RecordingRecordStoreBackend(InMemoryRecordStoreBackend):
    def __init__(self) -> None:
        super().__init__()
        self.saved_ids: list[str] = []
        self.clear_calls = 0

    def save(self, record_id: str, record) -> None:
        self.saved_ids.append(record_id)
        super().save(record_id, record)

    def clear(self) -> int:
        self.clear_calls += 1
        return super().clear()


def _decision(record_id: str, title: str, *, status: str = "accepted", t: float = 1.0) -> DecisionRecord:
    return DecisionRecord(id=record_id, title=title, status=status, decision=title, provenance=Provenance(time=t))


def _experience(record_id: str, goal: str, *, t: float = 1.0) -> ExperienceRecord:
    return ExperienceRecord(id=record_id, goal=goal, lesson=goal, provenance=Provenance(time=t))


def _skill(record_id: str, name: str, *, t: float = 1.0) -> SkillRecord:
    return SkillRecord(id=record_id, name=name, procedure=[name], provenance=Provenance(time=t))


def _pattern(record_id: str, symptom: str, *, t: float = 1.0) -> FailurePatternRecord:
    return FailurePatternRecord(id=record_id, symptom=symptom, fix=symptom, provenance=Provenance(time=t))


def _review(record_id: str, kind: str, title: str, *, status: str = "proposed", t: float = 1.0) -> ReviewItem:
    return ReviewItem(id=record_id, kind=kind, title=title, payload={"title": title}, status=status, provenance=Provenance(time=t))


def test_decision_repo_uses_injected_record_store_backend():
    backend = RecordingRecordStoreBackend()
    repo = DecisionRepo(backend=backend)

    repo.save_decision(_decision("d1", "Use FastMCP", t=1.0))
    repo.save_decision(_decision("d2", "Use backend facade", status="proposed", t=2.0))

    assert backend.saved_ids == ["d1", "d2"]
    assert repo.get_decision("d1").title == "Use FastMCP"
    assert [item.id for item in repo.list_decisions()] == ["d2", "d1"]
    assert [item.id for item in repo.list_decisions(status="accepted")] == ["d1"]
    assert [item.id for item in repo.search("backend facade", k=2)] == ["d2"]


def test_experience_skill_failure_and_review_repos_use_record_backends():
    exp_backend = RecordingRecordStoreBackend()
    skill_backend = RecordingRecordStoreBackend()
    pattern_backend = RecordingRecordStoreBackend()
    review_backend = RecordingRecordStoreBackend()

    exp_repo = ExperienceRepo(backend=exp_backend)
    skill_repo = SkillRepo(backend=skill_backend)
    pattern_repo = FailurePatternRepo(backend=pattern_backend)
    review_repo = ReviewQueueRepo(backend=review_backend)

    exp_repo.save_experience(_experience("e1", "Refactor storage"))
    skill_repo.save_skill(_skill("s1", "Use repository ports"))
    pattern_repo.save_failure_pattern(_pattern("p1", "Missing adapter boundary"))
    review_repo.save_review_item(_review("r1", "decision", "Review storage decision"))

    assert exp_backend.saved_ids == ["e1"]
    assert skill_backend.saved_ids == ["s1"]
    assert pattern_backend.saved_ids == ["p1"]
    assert review_backend.saved_ids == ["r1"]
    assert exp_repo.search("storage", k=1)[0].id == "e1"
    assert skill_repo.search("repository", k=1)[0].id == "s1"
    assert pattern_repo.search("adapter", k=1)[0].id == "p1"
    assert review_repo.list_review_items(kind="decision")[0].id == "r1"


def test_json_record_store_backend_roundtrips_typed_records(tmp_path: Path):
    path = tmp_path / "decisions.json"
    repo = DecisionRepo(backend=JsonRecordStoreBackend(path, DecisionRecord))

    repo.save_decision(_decision("d1", "Persist records"))

    restored = DecisionRepo(backend=JsonRecordStoreBackend(path, DecisionRecord))
    assert restored.get_decision("d1").title == "Persist records"
    assert restored.count() == 1

    restored.clear()
    assert DecisionRepo(backend=JsonRecordStoreBackend(path, DecisionRecord)).count() == 0


def test_persistent_repo_wrappers_use_json_record_store_backend(tmp_path: Path):
    inner_decisions = DecisionRepo()
    inner_decisions.save_decision(_decision("d1", "Persist decision"))
    persistent_decisions = PersistentDecisionRepo(inner_decisions, tmp_path / "decisions.json")

    assert persistent_decisions.get_decision("d1").title == "Persist decision"
    assert DecisionRepo(backend=JsonRecordStoreBackend(tmp_path / "decisions.json", DecisionRecord)).count() == 1

    inner_skills = SkillRepo()
    inner_skills.save_skill(_skill("s1", "Persist skill"))
    persistent_skills = PersistentSkillRepo(inner_skills, tmp_path / "skills.json")

    assert persistent_skills.get_skill("s1").name == "Persist skill"
    assert SkillRepo(backend=JsonRecordStoreBackend(tmp_path / "skills.json", SkillRecord)).count() == 1


def test_repository_builder_accepts_record_store_backend_bundle():
    decision_backend = RecordingRecordStoreBackend()
    experience_backend = RecordingRecordStoreBackend()
    repos = build_memory_repositories(
        record_store_backends=RecordStoreBackends(
            decision=decision_backend,
            experience=experience_backend,
        )
    )

    repos.decision.save_decision(_decision("d1", "Builder decision backend"))
    repos.experience.save_experience(_experience("e1", "Builder experience backend"))

    assert decision_backend.saved_ids == ["d1"]
    assert experience_backend.saved_ids == ["e1"]
    assert repos.decision.count() == 1
    assert repos.experience.count() == 1


def test_public_memory_builder_accepts_record_store_backend_bundle():
    decision_backend = RecordingRecordStoreBackend()
    experience_backend = RecordingRecordStoreBackend()
    memory = build_memory(
        record_store_backends=RecordStoreBackends(
            decision=decision_backend,
            experience=experience_backend,
        )
    )

    memory.write_decision(WriteDecisionRequest(title="Builder accepts record backend", decision="Expose record_store_backends on build_memory.", source="test"))
    memory.record_experience(RecordExperienceRequest(goal="Use record backend", lesson="Builder should wire typed stores.", source="test"))

    assert decision_backend.saved_ids
    assert experience_backend.saved_ids
    assert memory.repository_stats()["decisions"] == 1
    assert memory.repository_stats()["experiences"] == 1
