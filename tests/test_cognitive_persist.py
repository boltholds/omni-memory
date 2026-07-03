from __future__ import annotations

from domain.models import FailurePatternRecord, Provenance, SkillRecord
from infra.repo.cognitive_repo import (
    FailurePatternRepo,
    PersistentFailurePatternRepo,
    PersistentSkillRepo,
    SkillRepo,
)


def test_persistent_skill_repo_reloads_saved_skills(tmp_path):
    path = tmp_path / "skills.json"
    repo = PersistentSkillRepo(SkillRepo(), path)

    repo.save_skill(
        SkillRecord(
            id="skill-1",
            name="Use writeback pipeline",
            problem="A memory tool writes directly into a repository.",
            procedure=["Route through OmniMemory facade", "Verify policy behavior"],
            reuse_when=["adding memory tools"],
            confidence=0.9,
            provenance=Provenance(source="test", time=100.0),
        )
    )

    reloaded = PersistentSkillRepo(SkillRepo(), path)

    assert reloaded.count() == 1
    assert reloaded.get_skill("skill-1").name == "Use writeback pipeline"
    assert reloaded.search("writeback memory tool")[0].id == "skill-1"


def test_persistent_failure_pattern_repo_reloads_and_clears(tmp_path):
    path = tmp_path / "failure_patterns.json"
    repo = PersistentFailurePatternRepo(FailurePatternRepo(), path)

    repo.save_failure_pattern(
        FailurePatternRecord(
            id="pattern-1",
            symptom="Tests fail during collection",
            root_cause="A dependency import is unnecessary.",
            fix="Remove the dependency path or route through local code.",
            detection="pytest collection fails",
            confidence=0.9,
            provenance=Provenance(source="test", time=100.0),
        )
    )

    reloaded = PersistentFailurePatternRepo(FailurePatternRepo(), path)
    assert reloaded.count() == 1
    assert reloaded.get_failure_pattern("pattern-1").fix.startswith("Remove")
    assert reloaded.search("pytest dependency")[0].id == "pattern-1"

    assert reloaded.clear() == 1
    assert PersistentFailurePatternRepo(FailurePatternRepo(), path).count() == 0
