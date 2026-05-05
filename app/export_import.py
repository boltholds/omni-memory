# app/export_import.py
from __future__ import annotations
from typing import Any, Dict, List
import json
import sqlite3

from domain.models import MemoryObject, Fact, Episode, EpisodeEvent
from app.writeback import WriteBackService


def export_memory(vrepo, grepo, erepo) -> Dict[str, Any]:
    """
    Собирает архив памяти из трёх репозиториев в структуру:
    {
      "notes":    [MemoryObject... as dict],
      "facts":    [Fact... as dict],
      "episodes": [Episode... as dict]
    }
    """
    # --- Notes из VectorStoreRepo ---
    notes: List[Dict[str, Any]] = [o.model_dump() for o in vr_repo_iter_objects(vrepo)]

    # --- Facts из GraphRepo ---
    facts: List[Dict[str, Any]] = [f.model_dump() for f in gr_repo_iter_facts(grepo)]

    # --- Episodes из EpisodicRepo ---
    episodes: List[Dict[str, Any]] = [e.model_dump() for e in ep_repo_iter_all(erepo)]

    return {"notes": notes, "facts": facts, "episodes": episodes}


def import_memory(writeback: WriteBackService, archive: Dict[str, Any]):
    """
    Загрузка архива через WriteBackService (применяются политики/PII/т.д.).
    Возвращает итоговый WriteReport.
    """
    items: List[Dict[str, Any]] = []
    items += list(archive.get("facts") or [])
    items += list(archive.get("episodes") or [])
    # Преобразуем сохранённые MemoryObject к унифицированному "note"-входу
    for n in archive.get("notes") or []:
        if isinstance(n, dict):
            payload = (n.get("payload") or {})
            text = payload.get("text")
            if text is None and payload:
                # если не было text — сохраним весь payload в raw
                items.append({"id": n.get("id") or "", "type": "note", "payload": payload})
            else:
                items.append({"id": n.get("id") or "", "type": "note", "text": text or ""})

    return writeback.write(items)


# ----------------- Итераторы (внутренняя кухня репозиториев) -----------------

def vr_repo_iter_objects(vrepo) -> List[MemoryObject]:
    # VectorStoreRepo хранит оригинальные MemoryObject в _store
    return list(vrepo._store.values())  # type: ignore[attr-defined]

def gr_repo_iter_facts(grepo) -> List[Fact]:
    # GraphRepo хранит MultiDiGraph в _g с edges (s, o, key, data)
    g = grepo._g  # type: ignore[attr-defined]
    out: List[Fact] = []
    for s, o, k, data in g.edges(keys=True, data=True):
        out.append(Fact(
            id=str(k),
            subject=str(s),
            predicate=str(data.get("predicate", "")),
            object=str(o),
            provenance=data.get("provenance") or {},  # pydantic приведёт dict -> модель
            meta=data.get("meta") or {},
        ))
    return out

def ep_repo_iter_all(erepo) -> List[Episode]:
    # Вычитываем всё из SQLite
    conn: sqlite3.Connection = erepo._conn  # type: ignore[attr-defined]
    conn.row_factory = sqlite3.Row
    eps = conn.execute("SELECT id, participants, summary, provenance, meta FROM episodes").fetchall()
    out: List[Episode] = []
    for r in eps:
        evrows = conn.execute(
            "SELECT t, event_type, summary, refs FROM events WHERE ep_id=? ORDER BY COALESCE(t,0.0)",
            (r["id"],),
        ).fetchall()
        evs = [
            EpisodeEvent(
                t=er["t"], event_type=er["event_type"], summary=er["summary"],
                refs=json.loads(er["refs"] or "{}")
            )
            for er in evrows
        ]
        out.append(
            Episode(
                id=r["id"],
                participants=json.loads(r["participants"] or "[]"),
                summary=r["summary"] or "",
                events=evs,
                provenance=json.loads(r["provenance"] or "{}"),
                meta=json.loads(r["meta"] or "{}"),
            )
        )
    return out
