# infra/episodic_repo.py
from __future__ import annotations

import json
import sqlite3
from typing import List, Optional, Tuple
import time

from domain.models import Episode, EpisodeEvent
from domain.ports import IEpisodicRepository
from app.profiling import timed
from app.metrics import EPISODES
from infra.exceptions import SchemaInitError,PersistenceError,DataIntegrityError

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    created_at REAL,
    participants TEXT NOT NULL, -- JSON list[str]
    summary TEXT,
    provenance TEXT,            -- JSON
    meta TEXT                   -- JSON
);
CREATE TABLE IF NOT EXISTS events (
    ep_id TEXT NOT NULL,
    t REAL,
    event_type TEXT,
    summary TEXT,
    refs TEXT,                  -- JSON
    FOREIGN KEY(ep_id) REFERENCES episodes(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_events_ep_id ON events(ep_id);
CREATE INDEX IF NOT EXISTS idx_episodes_summary ON episodes(summary);
CREATE INDEX IF NOT EXISTS idx_events_summary ON events(summary);
"""


def _jdump(x) -> str:
    return json.dumps(x, ensure_ascii=False)

def _jload(s: Optional[str], context: str) -> dict:
    if s is None:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise DataIntegrityError(f"Bad JSON in {context}") from e


class EpisodicRepo(IEpisodicRepository):
    """
    SQLite-репозиторий эпизодов.
    - Таблица episodes: метаданные эпизода + participants (JSON)
    - Таблица events: события с временными метками и refs (JSON)
    Поиск: по участнику (participants) и ключевым словам в summary/событиях.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys=ON;")
        # строковые результаты по умолчанию -> str
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        try:
            with self._conn:
                self._conn.executescript(_SCHEMA)
        except sqlite3.Error as e:
            raise SchemaInitError("Failed to init episodic schema") from e

    # ---- IEpisodicRepository ----
    def save_episode(self, episode: Episode) -> None:
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO episodes (id, created_at, participants, summary, provenance, meta)
                    VALUES (?, COALESCE(?, strftime('%s','now')), ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        summary=excluded.summary,
                        participants=excluded.participants,
                        provenance=excluded.provenance,
                        meta=excluded.meta
                    """,
                    (
                        episode.id,
                        getattr(episode.provenance, "time", None),
                        _jdump(episode.participants),
                        episode.summary,
                        _jdump(episode.provenance.model_dump()),
                        _jdump(episode.meta),
                    ),
                )
                # Сначала удалим прежние события для upsert-логики по id
                self._conn.execute("DELETE FROM events WHERE ep_id = ?", (episode.id,))
                self._conn.executemany(
                    """
                    INSERT INTO events (ep_id, t, event_type, summary, refs)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            episode.id,
                            ev.t,
                            ev.event_type,
                            ev.summary,
                            _jdump(ev.refs),
                        )
                        for ev in episode.events
                    ],
                )
    
                try:
                    EPISODES.set(self.count())
                except Exception:
                    # не ломаем бизнес-операцию, если метрики недоступны
                    pass
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to save episode {episode.id}") from e
    
    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()
        return int(row["n"] if isinstance(row, sqlite3.Row) else row[0])
    
    @timed("retriever.retrieve", slow_ms=100)
    def search(self, user: str | None, entities: list[str], k: int = 5) -> List[Episode]:
        """
        Простой поиск:
        1) user ∈ participants (если задан)
        2) Любая из entities встречается в episodes.summary или events.summary
        Отдаём top-k по эвристическому скору: (совпадения в events)*2 + (совпадения в summary)*1
        """
        if k <= 0:
            raise ValueError("k must be > 0")
        if not isinstance(entities, list):
            raise ValueError("entities must be a list[str]")
    
        ents = [e.strip() for e in entities if e and e.strip()]
        # Подготовим LIKE-паттерны
        like_terms = [f"%{e}%" for e in ents]

        # 1) Кандидаты по участнику (или все эпизоды, если user=None)
        q_base = "SELECT id, created_at, participants, summary, provenance, meta FROM episodes"
        params: Tuple = ()
        if user:
            q_base += " WHERE participants LIKE ?"
            params = (f"%{json.dumps(user)[1:-1]}%",)  # грубо ищем подстроку имени в JSON-строке

        try:
            rows = list(self._conn.execute(q_base, params))

            # 2) Посчитаем скор по текстовым совпадениям
            scored: List[Tuple[float, sqlite3.Row]] = []
            for row in rows:
                ep_id = row["id"]
                ep_sum = row["summary"] or ""
                events = list(self._conn.execute("SELECT summary FROM events WHERE ep_id=?", (ep_id,)))
                ev_texts = [r["summary"] or "" for r in events]

                score = 0.0
                if like_terms:
                    # совпадения в summary эпизода
                    score += sum(1 for t in like_terms if t.strip("%").lower() in ep_sum.lower())
                    # совпадения по событиям (вес 2)
                    ev_hits = 0
                    for evs in ev_texts:
                        ev_hits += sum(1 for t in like_terms if t.strip("%").lower() in evs.lower())
                    score += 2.0 * ev_hits

                # Если ни user, ни entities не заданы — вернём всё со score=0 (ограничим k позже)
                scored.append((score, row))

            # 3) Сортировка по score (desc) и time (desc) как tie-breaker
            scored.sort(key=lambda x: (x[0], x[1]["created_at"] or 0.0), reverse=True)
            top = scored[: max(1, k)]

            # 4) Восстановим модели Episode
            episodes: List[Episode] = []
            for _, r in top:
                ep_id = r["id"]
                events_rows = self._conn.execute(
                    "SELECT t, event_type, summary, refs FROM events WHERE ep_id=? ORDER BY COALESCE(t, 0.0) ASC",
                    (ep_id,),
                ).fetchall()
                evs = [
                    EpisodeEvent(
                        t=er["t"],
                        event_type=er["event_type"],
                        summary=er["summary"],
                        refs=_jload(er["refs"], f"events.refs ep_id={ep_id}"),
                    )
                    for er in events_rows
                ]
                episodes.append(
                    Episode(
                        id=ep_id,
                        participants=json.loads(r["participants"]),
                        summary=r["summary"] or "",
                        events=evs,
                        provenance=_jload(r["provenance"], f"episodes.provenance id={ep_id}"),
                        meta=_jload(r["meta"], f"episodes.meta id={ep_id}"),
                    )
                )
            return episodes
        
        except sqlite3.Error as e:
            raise PersistenceError("Search failed") from e

    
    def gc_expired(self, now: float | None = None) -> int:
        now = time.time() if now is None else float(now)
        try:
            with self._conn:
                rows = self._conn.execute("SELECT id, meta FROM episodes").fetchall()
                dead_ids = []
                for r in rows:
                    meta = _jload(r["meta"], f"episodes.meta id={r['id']}")
                    exp = meta.get("expire_at")
                    if exp is not None and float(exp) < now:
                        dead_ids.append(r["id"])
                for eid in dead_ids:
                    self._conn.execute("DELETE FROM episodes WHERE id=?", (eid,))
                removed = len(dead_ids)
        except sqlite3.Error as e:
            raise PersistenceError("gc_expired failed") from e
        # Обновляем Gauge после удаления
        try:
            EPISODES.set(self.count())
        except Exception:
            pass
        return removed
