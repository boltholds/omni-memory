from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from domain.distiller import MemoryCandidate
from domain.llm import ILLMProvider, Msg
from domain.writeback import WritebackRawItem, WritebackRequest, WritebackResult, stable_id


FactCandidateStatus = Literal[
    "extracted",
    "validation_rejected",
    "policy_accepted",
    "policy_rejected",
    "requires_review",
    "saved",
    "error",
]


class FactMiningInput(BaseModel):
    text: str
    source: str = "fact-mining"
    dry_run: bool = True
    min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    policy_mode: Literal["permissive", "strict", "review"] = "review"
    domain_ids: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class FactMiningCandidate(BaseModel):
    id: str = ""
    subject: str
    predicate: str
    object: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quote: str
    reason: str = ""
    temporal_scope: Literal["current", "past", "future", "unknown"] = "unknown"
    domain_ids: list[str] = Field(default_factory=list)
    source_span: dict[str, int | None] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    status: FactCandidateStatus = "extracted"
    validation_reasons: list[str] = Field(default_factory=list)
    policy_reasons: list[str] = Field(default_factory=list)
    writeback_memory_id: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            payload = {
                "subject": self.subject,
                "predicate": self.predicate,
                "object": self.object,
                "evidence_quote": self.evidence_quote,
            }
            self.id = stable_id("fact_candidate", payload)

    @property
    def triple_key(self) -> tuple[str, str, str]:
        return (_canon(self.subject), _canon(self.predicate), _canon(self.object))


class FactMiningResult(BaseModel):
    dry_run: bool
    source: str
    policy_mode: str
    candidates: list[FactMiningCandidate] = Field(default_factory=list)
    writeback: WritebackResult = Field(default_factory=WritebackResult)
    extractor: str = "unknown"
    errors: list[str] = Field(default_factory=list)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def accepted_count(self) -> int:
        return sum(1 for item in self.candidates if item.status in {"policy_accepted", "saved"})

    @property
    def review_count(self) -> int:
        return sum(1 for item in self.candidates if item.status == "requires_review")

    @property
    def rejected_count(self) -> int:
        return sum(1 for item in self.candidates if item.status in {"validation_rejected", "policy_rejected", "error"})

    @property
    def saved_count(self) -> int:
        return 0 if self.dry_run else self.writeback.saved_count

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        data = super().model_dump(*args, **kwargs)
        data.update(
            {
                "candidate_count": self.candidate_count,
                "accepted_count": self.accepted_count,
                "review_count": self.review_count,
                "rejected_count": self.rejected_count,
                "saved_count": self.saved_count,
            }
        )
        return data


class FactExtractor(Protocol):
    name: str

    def extract(self, request: FactMiningInput) -> list[FactMiningCandidate]:
        ...


class LLMFactExtractor:
    name = "llm_json_fact_extractor"

    def __init__(self, llm: ILLMProvider, *, temperature: float = 0.0) -> None:
        self.llm = llm
        self.temperature = temperature

    def extract(self, request: FactMiningInput) -> list[FactMiningCandidate]:
        response = self.llm.generate(_fact_mining_messages(request), temperature=self.temperature)
        payload = _parse_json_object(response.get("text", ""))
        raw_candidates = payload.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raise ValueError("fact_mining_invalid_candidates_json")
        return [FactMiningCandidate.model_validate(item) for item in raw_candidates]


class StaticFactExtractor:
    """Test/helper extractor. Production callers should pass an LLMFactExtractor."""

    name = "static_fact_extractor"

    def __init__(self, candidates: list[FactMiningCandidate | dict[str, Any]]) -> None:
        self.candidates = [
            candidate if isinstance(candidate, FactMiningCandidate) else FactMiningCandidate.model_validate(candidate)
            for candidate in candidates
        ]

    def extract(self, request: FactMiningInput) -> list[FactMiningCandidate]:
        return [candidate.model_copy(deep=True) for candidate in self.candidates]


class FactMiningService:
    """Production-like fact mining pipeline.

    Pipeline stages:
    1. extractor proposes structured fact candidates;
    2. validator checks schema, confidence, duplicate triples and grounded evidence;
    3. accepted candidates are converted to WritebackRawItem objects;
    4. WriteBackService runs policies in dry-run or apply mode;
    5. candidates are annotated with policy/save/review status.
    """

    def __init__(
        self,
        *,
        writeback_service: Any,
        extractor: FactExtractor | None = None,
        llm: ILLMProvider | None = None,
    ) -> None:
        self.writeback_service = writeback_service
        self.extractor = extractor or (LLMFactExtractor(llm) if llm is not None else None)

    def mine_text(
        self,
        text: str,
        *,
        source: str = "fact-mining",
        dry_run: bool = True,
        min_confidence: float = 0.75,
        policy_mode: Literal["permissive", "strict", "review"] = "review",
        domain_ids: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        extractor: FactExtractor | None = None,
    ) -> FactMiningResult:
        request = FactMiningInput(
            text=text,
            source=source,
            dry_run=dry_run,
            min_confidence=min_confidence,
            policy_mode=policy_mode,
            domain_ids=domain_ids or [],
            meta=meta or {},
        )
        selected_extractor = extractor or self.extractor
        if selected_extractor is None:
            return FactMiningResult(
                dry_run=dry_run,
                source=source,
                policy_mode=policy_mode,
                extractor="none",
                errors=["fact_extractor_not_configured"],
            )

        try:
            candidates = selected_extractor.extract(request)
        except Exception as exc:
            return FactMiningResult(
                dry_run=dry_run,
                source=source,
                policy_mode=policy_mode,
                extractor=getattr(selected_extractor, "name", type(selected_extractor).__name__),
                errors=[f"extractor_error:{type(exc).__name__}:{exc}"],
            )

        candidates = _dedup_candidates(candidates)
        for candidate in candidates:
            _validate_candidate(candidate, request)

        write_items = [_candidate_to_raw_item(candidate, request) for candidate in candidates if candidate.status != "validation_rejected"]
        writeback = self.writeback_service.write(
            WritebackRequest(
                items=write_items,
                source=source,
                dry_run=dry_run,
                meta={
                    **request.meta,
                    "policy_mode": policy_mode,
                    "mined_by": "fact_mining",
                    "fact_mining_dry_run": dry_run,
                    "domain_ids": request.domain_ids,
                },
            )
        )
        _annotate_candidates_from_writeback(candidates, writeback, dry_run=dry_run)
        return FactMiningResult(
            dry_run=dry_run,
            source=source,
            policy_mode=policy_mode,
            candidates=candidates,
            writeback=writeback,
            extractor=getattr(selected_extractor, "name", type(selected_extractor).__name__),
        )


def _fact_mining_messages(request: FactMiningInput) -> list[Msg]:
    system = """You extract only explicit, evidence-grounded facts for a long-term memory system.
Return a single JSON object and no prose.

Rules:
- Extract only facts directly supported by the text.
- Do not infer hidden intent, private attributes, medical/legal conclusions, or guesses.
- Every candidate must include an exact evidence_quote copied from the input text.
- Prefer stable product/project facts, decisions, capabilities, constraints, APIs, dependencies and preferences.
- Skip secrets, credentials, emails, phone numbers and sensitive personal data.
- Use short normalized predicates such as uses, requires, supports, decided, configured_as, owned_by, prefers, located_in.
- Set temporal_scope to current, past, future or unknown.
- Confidence must reflect evidence quality.

Schema:
{
  "candidates": [
    {
      "subject": "string",
      "predicate": "string",
      "object": "string",
      "confidence": 0.0,
      "evidence_quote": "exact quote from text",
      "reason": "why this should become memory",
      "temporal_scope": "current|past|future|unknown",
      "domain_ids": ["optional domain ids"],
      "source_span": {"start": 0, "end": 0},
      "raw": {}
    }
  ]
}
"""
    user = f"""Extract fact candidates from this text.

Default domain_ids: {json.dumps(request.domain_ids, ensure_ascii=False)}
Minimum useful confidence: {request.min_confidence}

TEXT:
{request.text}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _candidate_to_raw_item(candidate: FactMiningCandidate, request: FactMiningInput) -> WritebackRawItem:
    domain_ids = candidate.domain_ids or request.domain_ids
    meta = {
        **request.meta,
        "confidence": candidate.confidence,
        "evidence": candidate.evidence_quote,
        "distiller_reason": candidate.reason,
        "temporal_scope": candidate.temporal_scope,
        "mined_by": "fact_mining",
        "candidate_id": candidate.id,
        "domain_ids": domain_ids,
        "source_span": candidate.source_span,
    }
    payload = {
        "subject": candidate.subject.strip().lower(),
        "predicate": candidate.predicate.strip().lower(),
        "object": candidate.object.strip(),
    }
    return WritebackRawItem(
        id=stable_id("fact", payload),
        type="fact",
        subject=payload["subject"],
        predicate=payload["predicate"],
        object=payload["object"],
        meta=meta,
    )


def _validate_candidate(candidate: FactMiningCandidate, request: FactMiningInput) -> None:
    reasons: list[str] = []
    if not candidate.subject.strip() or not candidate.predicate.strip() or not candidate.object.strip():
        reasons.append("empty_fact_field")
    if candidate.confidence < request.min_confidence:
        reasons.append("confidence_too_low")
    if not candidate.evidence_quote.strip():
        reasons.append("missing_evidence_quote")
    elif not _quote_is_grounded(candidate.evidence_quote, request.text):
        reasons.append("evidence_quote_not_found")
    if _looks_like_secret(candidate.evidence_quote) or _looks_like_secret(str(candidate.raw)):
        reasons.append("secret_or_credential_detected")
    if _looks_like_pii(candidate.evidence_quote):
        reasons.append("pii_detected_in_evidence")
    if _looks_like_negated_or_uncertain(candidate.evidence_quote):
        reasons.append("uncertain_or_negated_evidence")

    if reasons:
        candidate.status = "validation_rejected"
        candidate.validation_reasons = reasons


def _dedup_candidates(candidates: list[FactMiningCandidate]) -> list[FactMiningCandidate]:
    out: list[FactMiningCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        key = candidate.triple_key
        if key in seen:
            duplicate = candidate.model_copy(update={"status": "validation_rejected", "validation_reasons": ["duplicate_candidate"]})
            out.append(duplicate)
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _annotate_candidates_from_writeback(candidates: list[FactMiningCandidate], writeback: WritebackResult, *, dry_run: bool) -> None:
    by_candidate_id: dict[str, FactMiningCandidate] = {candidate.id: candidate for candidate in candidates}

    for memory_object in writeback.saved:
        candidate_id = (getattr(memory_object, "meta", {}) or {}).get("candidate_id")
        candidate = by_candidate_id.get(str(candidate_id))
        if candidate is None:
            continue
        candidate.status = "policy_accepted" if dry_run else "saved"
        candidate.writeback_memory_id = getattr(memory_object, "id", None)

    for decision in [*writeback.rejected, *writeback.errors]:
        memory_object = decision.memory_object
        candidate_id = (getattr(memory_object, "meta", {}) or {}).get("candidate_id") if memory_object is not None else None
        candidate = by_candidate_id.get(str(candidate_id))
        if candidate is None:
            continue
        if decision.reason == "requires_review":
            candidate.status = "requires_review"
        elif decision in writeback.errors:
            candidate.status = "error"
        else:
            candidate.status = "policy_rejected"
        candidate.policy_reasons.append(decision.reason or "unknown")
        candidate.writeback_memory_id = getattr(memory_object, "id", None)


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("fact_mining_expected_json_object")
    return data


def _quote_is_grounded(quote: str, text: str) -> bool:
    if quote in text:
        return True
    return _normalize_ws(quote) in _normalize_ws(text)


def _normalize_ws(value: str) -> str:
    return " ".join(str(value or "").split())


def _canon(value: str) -> str:
    return _normalize_ws(value).strip().casefold()


_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
]

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s()\-]{7,}\d)(?!\d)")
_UNCERTAIN_RE = re.compile(r"(?i)\b(maybe|might|probably|possibly|not sure|unclear|guess|думаю|возможно|кажется|не уверен|не уверена)\b")
_NEGATION_RE = re.compile(r"(?i)\b(does not|doesn't|do not|don't|not|never|не использует|не поддерживает|не является)\b")


def _looks_like_secret(text: str) -> bool:
    return any(pattern.search(str(text or "")) for pattern in _SECRET_PATTERNS)


def _looks_like_pii(text: str) -> bool:
    raw = str(text or "")
    return bool(_EMAIL_RE.search(raw) or _PHONE_RE.search(raw))


def _looks_like_negated_or_uncertain(text: str) -> bool:
    raw = str(text or "")
    return bool(_UNCERTAIN_RE.search(raw) or _NEGATION_RE.search(raw))


def candidate_to_memory_candidate(candidate: FactMiningCandidate) -> MemoryCandidate:
    return MemoryCandidate(
        kind="fact",
        should_write=candidate.status != "validation_rejected",
        confidence=candidate.confidence,
        reason=candidate.reason,
        evidence_quote=candidate.evidence_quote,
        temporal_scope=candidate.temporal_scope,
        payload={
            "subject": candidate.subject,
            "predicate": candidate.predicate,
            "object": candidate.object,
        },
    )
