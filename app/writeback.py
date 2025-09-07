from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple, Union

from domain.models import (
    MemoryObject,
    Fact,
    Episode,
    WriteReport,
    Provenance,
)
from domain.policy import MemoryPolicy
from domain.ports import IMemoryWriteRepository, IGraphRepository, IEpisodicRepository
from infra.consistency import score_trust_recent_first


EmailRe = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ApiKeyLikeRe = re.compile(r"(api[_-]?key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{12,}", re.I)


def contains_pii(text: str) -> bool:
    return bool(EmailRe.search(text) or ApiKeyLikeRe.search(text))


def obj_kind(obj: Dict[str, Any]) -> str:
    """
    Грубая маршрутизация входа:
    - если есть subject/predicate/object -> Fact
    - если есть participants/events -> Episode
    - иначе -> MemoryObject (note)
    """
    if all(k in obj for k in ("subject", "predicate", "object")):
        return "fact"
    if "participants" in obj or "events" in obj:
        return "episode"
    return "note"


class WriteBackService:
    """
    Фильтрация и запись новых объектов памяти.
    Политики:
      - PII: выкинуть заметки с email/секретами
      - Confidence: факты ниже порога не писать
    """

    def __init__(
        self,
        vector_repo: IMemoryWriteRepository,
        graph_repo: IGraphRepository,
        episodic_repo: IEpisodicRepository,
        policy: MemoryPolicy | None = None,
    ) -> None:
        self._vector = vector_repo
        self._graph = graph_repo
        self._episodic = episodic_repo
        self._policy = policy or MemoryPolicy()

    def _parse(self, raw: Dict[str, Any]) -> Tuple[str, Union[MemoryObject, Fact, Episode]]:
        kind = obj_kind(raw)
        if kind == "fact":
            # гарантируем наличие provenance
            raw.setdefault("provenance", Provenance().model_dump())
            return "fact", Fact.model_validate(raw)
        if kind == "episode":
            raw.setdefault("provenance", Provenance().model_dump())
            return "episode", Episode.model_validate(raw)
        # note / generic memory object
        # нормализуем в MemoryObject(id, type="note", payload={"text": ...})
        payload = raw.get("payload") or {}
        if "text" not in payload and "content" in raw:
            payload["text"] = raw["content"]
        if "text" not in payload and "text" in raw:
            payload["text"] = raw["text"]
        norm = {
            "id": raw.get("id") or raw.get("uuid") or raw.get("hash") or "",
            "type": raw.get("type") or "note",
            "payload": payload or {"raw": raw},
            "provenance": raw.get("provenance") or Provenance().model_dump(),
            "meta": raw.get("meta") or {},
        }
        return "note", MemoryObject.model_validate(norm)

    def write(self, items: List[Dict[str, Any]]) -> WriteReport:
        saved = 0
        rejected = 0
        reasons: List[str] = []

        # --- 1) Парсим и группируем ---
        notes: List[MemoryObject] = []
        facts: List[Fact] = []
        episodes: List[Episode] = []

        parsed: List[Tuple[str, Union[MemoryObject, Fact, Episode]]] = []
        for raw in items:
            try:
                parsed.append(self._parse(raw))
            except Exception as e:
                rejected += 1
                reasons.append(f"parse_error: {e}")
        for kind, model in parsed:
            if kind == "fact":
                facts.append(model)  # type: ignore[arg-type]
            elif kind == "episode":
                episodes.append(model)  # type: ignore[arg-type]
            else:
                notes.append(model)  # type: ignore[arg-type]

        # --- 2) Фильтрация PII для заметок ---
        filtered_notes: List[MemoryObject] = []
        for n in notes:
            text = str((n.payload or {}).get("text", ""))[:5000]
            if contains_pii(text):
                rejected += 1
                reasons.append("pii_blocked_note")
                continue
            filtered_notes.append(n)
        notes = filtered_notes

        # --- 3) Оценка confidence для фактов ---
        fact_scores = score_trust_recent_first(facts) if facts else {}
        accept_th = self._policy.confidence.accept
        filtered_facts: List[Fact] = []
        for f in facts:
            sc = fact_scores.get(f.id, 0.5)
            if sc < accept_th:
                rejected += 1
                reasons.append(f"low_confidence_fact:{f.id}:{sc:.2f}")
                continue
            filtered_facts.append(f)
        facts = filtered_facts

        # --- 4) Запись по репозиториям ---
        for n in notes:
            try:
                self._vector.save_object(n)
                saved += 1
            except Exception as e:
                rejected += 1
                reasons.append(f"note_save_error:{n.id}:{e}")

        for f in facts:
            try:
                self._graph.save_fact(f)
                saved += 1
            except Exception as e:
                rejected += 1
                reasons.append(f"fact_save_error:{f.id}:{e}")

        for e in episodes:
            try:
                self._episodic.save_episode(e)
                saved += 1
            except Exception as ex:
                rejected += 1
                reasons.append(f"episode_save_error:{e.id}:{ex}")

        return WriteReport(saved=saved, rejected=rejected, reasons=reasons)
