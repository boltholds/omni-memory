from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from domain.distiller import MemoryCandidate, SessionDistillationResult, SessionTurn
from domain.writeback import WritebackRawItem

_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
]

_ALLOWED_KINDS = {"fact", "preference", "episode", "note"}


@dataclass(frozen=True)
class CandidateValidation:
    accepted: bool
    reason: str


class ConservativeCandidateValidator:
    """Filters session-level memory candidates before they can reach writeback.

    This validator intentionally prefers false negatives over false positives.
    Bad memory is usually worse than missing memory.
    """

    def __init__(self, *, min_confidence: float = 0.75) -> None:
        self.min_confidence = min_confidence

    def validate(self, candidate: MemoryCandidate, *, transcript: str) -> CandidateValidation:
        if not candidate.should_write:
            return CandidateValidation(False, "candidate_marked_as_reject")

        if candidate.kind not in _ALLOWED_KINDS:
            return CandidateValidation(False, "unsupported_candidate_kind")

        if candidate.confidence < self.min_confidence:
            return CandidateValidation(False, "candidate_confidence_too_low")

        evidence = candidate.evidence_quote.strip()
        if not evidence:
            return CandidateValidation(False, "missing_evidence_quote")

        if evidence not in transcript:
            return CandidateValidation(False, "evidence_quote_not_found_in_transcript")

        if _looks_like_secret(evidence) or _payload_has_secret(candidate.payload):
            return CandidateValidation(False, "secret_or_credential_detected")

        if candidate.kind == "fact" and not _valid_fact_payload(candidate.payload):
            return CandidateValidation(False, "invalid_fact_payload")

        if candidate.kind == "episode" and not str(candidate.payload.get("summary", "")).strip():
            return CandidateValidation(False, "invalid_episode_payload")

        if candidate.kind in {"note", "preference"} and not _candidate_text(candidate).strip():
            return CandidateValidation(False, "missing_text_payload")

        return CandidateValidation(True, "accepted")


def build_transcript(turns: list[SessionTurn]) -> str:
    return "\n".join(f"{turn.role}: {turn.content}" for turn in turns)


def accepted_candidates(
    result: SessionDistillationResult,
    *,
    transcript: str,
    validator: ConservativeCandidateValidator | None = None,
) -> tuple[list[MemoryCandidate], list[str]]:
    validator = validator or ConservativeCandidateValidator()
    accepted: list[MemoryCandidate] = []
    rejected: list[str] = list(result.rejected)

    for candidate in result.candidates:
        decision = validator.validate(candidate, transcript=transcript)
        if decision.accepted:
            accepted.append(candidate)
        else:
            rejected.append(f"{decision.reason}: {candidate.reason or candidate.kind}")

    return accepted, rejected


def candidates_to_writeback_items(
    candidates: list[MemoryCandidate],
    *,
    source: str,
    meta: dict[str, Any] | None = None,
) -> list[WritebackRawItem]:
    items: list[WritebackRawItem] = []
    now = time.time()
    base_meta = meta or {}

    for candidate in candidates:
        item_meta = {
            **base_meta,
            "confidence": candidate.confidence,
            "evidence": candidate.evidence_quote,
            "distiller_reason": candidate.reason,
            "temporal_scope": candidate.temporal_scope,
            "distilled_from": "session",
        }
        provenance = {
            "source": source,
            "time": now,
            "meta": {"evidence": candidate.evidence_quote},
        }

        if candidate.kind == "fact":
            payload = candidate.payload
            items.append(
                WritebackRawItem(
                    id=f"fact-{uuid.uuid4().hex}",
                    type="fact",
                    subject=str(payload["subject"]).strip().lower(),
                    predicate=str(payload["predicate"]).strip().lower(),
                    object=str(payload["object"]).strip(),
                    provenance=provenance,
                    meta=item_meta,
                )
            )
            continue

        if candidate.kind == "episode":
            summary = str(candidate.payload.get("summary", "")).strip()
            participants = candidate.payload.get("participants", []) or []
            items.append(
                WritebackRawItem(
                    id=f"episode-{uuid.uuid4().hex}",
                    type="episode",
                    summary=summary,
                    participants=list(participants),
                    events=[{"event_type": "session_summary", "summary": summary}],
                    provenance=provenance,
                    meta=item_meta,
                )
            )
            continue

        text = _candidate_text(candidate)
        items.append(
            WritebackRawItem(
                id=f"{candidate.kind}-{uuid.uuid4().hex}",
                type=candidate.kind,
                payload={"text": text, "kind": candidate.kind},
                provenance=provenance,
                meta=item_meta,
            )
        )

    return items


def _candidate_text(candidate: MemoryCandidate) -> str:
    return str(candidate.payload.get("text") or candidate.payload.get("summary") or candidate.reason or "")


def _valid_fact_payload(payload: dict[str, Any]) -> bool:
    return all(str(payload.get(key, "")).strip() for key in ("subject", "predicate", "object"))


def _looks_like_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def _payload_has_secret(payload: dict[str, Any]) -> bool:
    return _looks_like_secret(str(payload))
