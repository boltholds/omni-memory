from __future__ import annotations

from app.builder import build_memory
from app.fact_mining import StaticFactExtractor
from infra.embeddings.factory import HashEmbedder


TEXT = "OmniMemory uses FastMCP for the MCP server. OmniMemory requires explicit memory scope for durable facts."


def _memory_with_candidates(candidates):
    return build_memory(
        use_llm=False,
        embedder=HashEmbedder(),
        fact_extractor=StaticFactExtractor(candidates),
    )


def test_fact_mining_dry_run_returns_policy_accepted_candidates_without_writing():
    memory = _memory_with_candidates(
        [
            {
                "subject": "OmniMemory",
                "predicate": "uses",
                "object": "FastMCP",
                "confidence": 0.91,
                "evidence_quote": "OmniMemory uses FastMCP for the MCP server.",
                "reason": "Explicit implementation fact.",
                "temporal_scope": "current",
            }
        ]
    )

    result = memory.mine_facts(
        TEXT,
        source="test-fact-mining",
        dry_run=True,
        domain_ids=["domain:project:omni-memory"],
    )

    assert result.dry_run is True
    assert result.candidate_count == 1
    assert result.accepted_count == 1
    assert result.saved_count == 0
    assert result.candidates[0].status == "policy_accepted"
    assert result.candidates[0].writeback_memory_id is not None
    assert memory.repository_stats()["facts"] == 0
    assert result.writeback.saved_count == 1
    assert result.writeback.operations[0].status == "accepted"


def test_fact_mining_apply_saves_policy_accepted_candidates():
    memory = _memory_with_candidates(
        [
            {
                "subject": "OmniMemory",
                "predicate": "requires",
                "object": "explicit memory scope",
                "confidence": 0.88,
                "evidence_quote": "OmniMemory requires explicit memory scope for durable facts.",
                "reason": "Explicit architecture constraint.",
                "temporal_scope": "current",
            }
        ]
    )

    result = memory.mine_facts(TEXT, source="test-fact-mining", dry_run=False)

    assert result.dry_run is False
    assert result.saved_count == 1
    assert result.candidates[0].status == "saved"
    assert memory.repository_stats()["facts"] == 1
    retrieved = memory.retrieve("OmniMemory requires scope", intent="answer_question")
    assert any(fact.predicate == "requires" for fact in retrieved.facts)


def test_fact_mining_rejects_ungrounded_evidence_before_writeback():
    memory = _memory_with_candidates(
        [
            {
                "subject": "OmniMemory",
                "predicate": "uses",
                "object": "Kubernetes",
                "confidence": 0.95,
                "evidence_quote": "OmniMemory uses Kubernetes.",
                "reason": "Extractor hallucinated unsupported evidence.",
                "temporal_scope": "current",
            }
        ]
    )

    result = memory.mine_facts(TEXT, source="test-fact-mining", dry_run=False)

    assert result.candidate_count == 1
    assert result.accepted_count == 0
    assert result.candidates[0].status == "validation_rejected"
    assert "evidence_quote_not_found" in result.candidates[0].validation_reasons
    assert result.writeback.total_count == 0
    assert memory.repository_stats()["facts"] == 0


def test_fact_mining_conflicting_candidate_requires_review_in_review_mode():
    memory = _memory_with_candidates(
        [
            {
                "subject": "OmniMemory",
                "predicate": "uses",
                "object": "handwritten JSON-RPC server",
                "confidence": 0.9,
                "evidence_quote": "OmniMemory uses FastMCP for the MCP server.",
                "reason": "Conflicting extracted object should be reviewed.",
                "temporal_scope": "current",
            }
        ]
    )
    memory.write_items_raw(
        [
            {
                "id": "fact-existing-fastmcp",
                "type": "fact",
                "subject": "omnimemory",
                "predicate": "uses",
                "object": "FastMCP",
                "meta": {"confidence": 1.0},
            }
        ],
        source="codex-dev",
    )

    result = memory.mine_facts(TEXT, source="test-fact-mining", dry_run=False, policy_mode="review")

    assert result.review_count == 1
    assert result.saved_count == 0
    assert result.candidates[0].status == "requires_review"
    assert "requires_review" in result.candidates[0].policy_reasons
    assert memory.repository_stats()["facts"] == 1


def test_fact_mining_rejects_pii_and_low_confidence_candidates():
    text = "OmniMemory uses FastMCP. Contact root@example.com for access."
    memory = _memory_with_candidates(
        [
            {
                "subject": "OmniMemory",
                "predicate": "uses",
                "object": "FastMCP",
                "confidence": 0.2,
                "evidence_quote": "OmniMemory uses FastMCP.",
                "reason": "Too weak confidence.",
                "temporal_scope": "current",
            },
            {
                "subject": "access",
                "predicate": "contact",
                "object": "root@example.com",
                "confidence": 0.95,
                "evidence_quote": "Contact root@example.com for access.",
                "reason": "PII should not become a fact.",
                "temporal_scope": "current",
            },
        ]
    )

    result = memory.mine_facts(text, source="test-fact-mining", dry_run=False, min_confidence=0.75)

    assert result.accepted_count == 0
    assert {reason for candidate in result.candidates for reason in candidate.validation_reasons} >= {
        "confidence_too_low",
        "pii_detected_in_evidence",
    }
    assert memory.repository_stats()["facts"] == 0
