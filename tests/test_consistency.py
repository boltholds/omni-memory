# tests/test_consistency.py
from domain.models import Fact, Provenance
from infra.consistency import SimpleConsistencyEngine, build_fact_beliefs, score_trust_recent_first


def F(fid: str, s: str, p: str, o: str, t: float = 0.0, src: str = "user") -> Fact:
    return Fact(
        id=fid,
        subject=s,
        predicate=p,
        object=o,
        provenance=Provenance(source=src, time=t),
        meta={},
    )


def Fm(
    fid: str,
    s: str,
    p: str,
    o: str,
    meta: dict,
    t: float = 0.0,
    src: str = "user",
) -> Fact:
    fact = F(fid, s, p, o, t=t, src=src)
    fact.meta = meta
    return fact


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


def test_build_fact_beliefs_selects_current_and_keeps_history():
    facts = [
        F("old", "Alice", "at", "lighthouse", t=10.0, src="user"),
        F("new", "alice", "at", "bridge", t=20.0, src="verified"),
    ]

    beliefs = build_fact_beliefs(facts, as_of=30.0)

    assert len(beliefs) == 1
    belief = beliefs[0]
    assert belief.key == "alice::at"
    assert belief.current is not None
    assert belief.current.id == "new"
    assert belief.status == "conflict"
    assert [f.id for f in belief.alternatives] == ["old"]


def test_build_fact_beliefs_ignores_expired_fact_for_current():
    facts = [
        Fact(
            id="expired",
            subject="alice",
            predicate="at",
            object="lighthouse",
            provenance=Provenance(source="verified", time=100.0),
            meta={"valid_to": 150.0},
        ),
        Fact(
            id="active",
            subject="alice",
            predicate="at",
            object="bridge",
            provenance=Provenance(source="user", time=10.0),
            meta={},
        ),
    ]

    beliefs = build_fact_beliefs(facts, as_of=200.0)

    assert beliefs[0].current is not None
    assert beliefs[0].current.id == "active"
    assert [f.id for f in beliefs[0].historical] == ["expired"]


def test_status_metadata_controls_current_belief_and_conflicts():
    facts = [
        Fm(
            "old",
            "mcp_server",
            "implemented_with",
            "minimal json-rpc",
            {"status": "historical", "superseded_by": "new"},
            t=10.0,
        ),
        Fm(
            "new",
            "mcp_server",
            "implemented_with",
            "official MCP SDK FastMCP",
            {"status": "current"},
            t=20.0,
            src="verified",
        ),
        Fm(
            "bad",
            "mcp_server",
            "implemented_with",
            "wrong implementation",
            {"status": "retracted"},
            t=30.0,
        ),
    ]

    conflicts = SimpleConsistencyEngine().detect_conflicts(facts)
    beliefs = build_fact_beliefs(facts, as_of=40.0)

    assert conflicts.conflicts == []
    assert beliefs[0].current is not None
    assert beliefs[0].current.id == "new"
    assert [fact.id for fact in beliefs[0].historical] == ["old"]
