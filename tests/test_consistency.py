# tests/test_consistency.py
from domain.models import Fact, Provenance
from infra.consistency import SimpleConsistencyEngine, score_trust_recent_first


def F(fid: str, s: str, p: str, o: str, t: float = 0.0, src: str = "user") -> Fact:
    return Fact(
        id=fid,
        subject=s,
        predicate=p,
        object=o,
        provenance=Provenance(source=src, time=t),
        meta={},
    )


def test_detect_conflict_on_different_objects():
    facts = [
        F("f1", "Alice", "at", "lighthouse", t=1.0),
        F("f2", "Alice", "at", "bridge", t=2.0),
        F("f3", "Bob",   "at", "lighthouse", t=3.0),
    ]
    eng = SimpleConsistencyEngine()
    report = eng.detect_conflicts(facts)

    assert len(report.conflicts) == 1
    c = report.conflicts[0]
    assert c.key == "Alice::at"
    assert set(c.variants) == {"bridge", "lighthouse"}


def test_no_conflict_when_same_object_repeats():
    facts = [
        F("a1", "Alice", "near", "sea"),
        F("a2", "Alice", "near", "sea"),
    ]
    eng = SimpleConsistencyEngine()
    report = eng.detect_conflicts(facts)
    assert report.conflicts == []


def test_score_trust_recent_first_heuristic():
    facts = [
        F("x1", "A", "p", "o1", t=10.0, src="user"),
        F("x2", "A", "p", "o2", t=20.0, src="verified"),
    ]
    scores = score_trust_recent_first(facts)
    assert "x1" in scores and "x2" in scores
    # verified + новее должен иметь больший скор
    assert scores["x2"] > scores["x1"]
