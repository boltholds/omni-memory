from __future__ import annotations

import json
import re

from omni_memory.domain.distiller import MemoryCandidate, SessionDistillationResult, SessionTurn
from omni_memory.domain.llm import ILLMProvider, Msg

_SYSTEM_PROMPT = """You are a conservative memory distiller for an AI assistant.
Extract only durable, useful memories from the whole dialogue session.
Prefer rejecting over writing uncertain or noisy memory.

Rules:
- Do not write small talk.
- Do not write secrets, API keys, tokens, passwords, or credentials.
- Do not turn plans, guesses, questions, rejected ideas, or considered alternatives into facts.
- Do not turn negated statements into positive facts.
- Every writable candidate must include an exact evidence_quote copied from the transcript.
- Use kind=fact only for stable structured facts with subject, predicate, object.
- Use kind=preference for durable user preferences.
- Use kind=episode for useful debugging/project/session summaries.
- Use kind=note only for useful durable free-form memory that is not a fact/preference/episode.
- Use should_write=false when uncertain.

Return strict JSON only with this shape:
{
  "candidates": [
    {
      "kind": "fact|preference|episode|note|reject",
      "should_write": true,
      "confidence": 0.0,
      "reason": "why this should or should not be written",
      "evidence_quote": "exact quote from transcript",
      "temporal_scope": "current|past|future|unknown",
      "payload": {}
    }
  ],
  "rejected": []
}

For fact payload use: {"subject": "...", "predicate": "...", "object": "..."}.
For preference/note payload use: {"text": "..."}.
For episode payload use: {"summary": "...", "participants": []}.
"""


class ConservativeLLMSessionDistiller:
    """Session-level LLM distiller.

    It only proposes candidates. Validation and writeback are handled elsewhere.
    """

    def __init__(self, llm: ILLMProvider, *, temperature: float = 0.0) -> None:
        self.llm = llm
        self.temperature = temperature

    def distill_session(self, turns: list[SessionTurn]) -> SessionDistillationResult:
        transcript = "\n".join(f"{turn.role}: {turn.content}" for turn in turns)
        messages: list[Msg] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ]
        result = self.llm.generate(messages, temperature=self.temperature)
        text = str(result.get("text", "")).strip()
        data = _extract_json_object(text)
        if data is None:
            return SessionDistillationResult(rejected=["distiller_returned_invalid_json"])

        try:
            return SessionDistillationResult.model_validate(data)
        except Exception as exc:
            return SessionDistillationResult(rejected=[f"distiller_schema_validation_failed: {exc}"])


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None

    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            loaded = json.loads(fenced.group(1))
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            return None

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        loaded = json.loads(text[start : end + 1])
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        return None
