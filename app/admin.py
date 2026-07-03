from __future__ import annotations

import cProfile
import io
import logging
import pstats
import time
from typing import Any, Dict, Protocol

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.export_import import export_memory, import_memory
from app.security import admin_api_key_guard
from infra.repo.episodic_repo import _jload


class WritebackFacade(Protocol):
    def write(self, items: list[dict[str, Any]]):
        ...


class LogLevelIn(BaseModel):
    level: str  # DEBUG|INFO|WARNING|ERROR


class VectorPathIn(BaseModel):
    dir: str


class GCRequest(BaseModel):
    dry_run: bool = False
    now: float | None = None


class ProfileReq(BaseModel):
    seconds: int = 5  # лимит
    target: str = "answer"  # "retrieve"|"context"|"answer"


router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(admin_api_key_guard)])


def attach_repos(vrepo, grepo, erepo, writeback: WritebackFacade | None = None):
    router.state = {"vrepo": vrepo, "grepo": grepo, "erepo": erepo, "writeback": writeback}  # type: ignore[attr-defined]


@router.post("/reset")
def reset():
    st = router.state  # type: ignore[attr-defined]
    # просто пересоздадим in-memory структуры
    st["vrepo"]._index = type(st["vrepo"]._index)(st["vrepo"]._dim)  # IndexFlatIP
    st["vrepo"]._ids.clear()
    st["vrepo"]._store.clear()

    st["grepo"]._g.clear()
    # episodic проще пересоздать схему
    st["erepo"]._conn.close()
    st["erepo"].__init__(db_path=":memory:")

    return {"status": "reset"}


@router.get("/export")
def export_all():
    st = router.state  # type: ignore[attr-defined]
    data = export_memory(st["vrepo"], st["grepo"], st["erepo"])
    return data


@router.post("/import")
def import_all(archive: Dict[str, Any]):
    st = router.state  # type: ignore[attr-defined]
    wb = st.get("writeback")
    if wb is None:
        return {"saved": 0, "rejected": 0, "reasons": ["writeback_not_attached"]}
    rep = import_memory(wb, archive)
    return rep


@router.post("/vector/save")
def vector_save(inp: VectorPathIn):
    st = router.state  # type: ignore[attr-defined]
    st["vrepo"].save(inp.dir)
    return {"status": "ok", "dir": inp.dir}


@router.post("/vector/load")
def vector_load(inp: VectorPathIn):
    st = router.state  # type: ignore[attr-defined]
    st["vrepo"].load(inp.dir)
    return {"status": "ok", "dir": inp.dir}


@router.post("/log-level")
def set_log_level(inp: LogLevelIn):
    level = getattr(logging, inp.level.upper(), None)
    if level is None:
        return {"status": "error", "message": f"Unknown level: {inp.level}"}
    logging.getLogger().setLevel(level)
    for name in ("app.http", "app.llm", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).setLevel(level)
    return {"status": "ok", "level": inp.level.upper()}


@router.post("/profile/once")
def profile_once(inp: ProfileReq):
    """
    Синхронно: прогоняет типовой сценарий (напр. answer на фиксированный вопрос)
    и возвращает топ функций по времени.
    """
    import orchestrator
    pr = cProfile.Profile()
    pr.enable()
    # --- прогон демо ---
    if inp.target == "retrieve":
        orchestrator.plan_retrieval("Where is Alice?")
    elif inp.target == "context":
        b = orchestrator.plan_retrieval("Where is Alice?")
        orchestrator.assemble_context(b)
    else:
        from app.main import llm_provider
        b = orchestrator.plan_retrieval("Where is Alice?")
        pack = orchestrator.assemble_context(b)
        if llm_provider:
            from app.prompting import PromptRenderer
            prnd = PromptRenderer()
            msgs = prnd.make_messages("Where is Alice?", [f"{s.title}:\n{s.body}" for s in pack.sections])
            llm_provider.generate(msgs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(40)  # топ-40
    return {"report": s.getvalue()}


@router.post("/gc")
def gc(inp: GCRequest):
    st = router.state  # type: ignore[attr-defined]
    now = time.time() if inp.now is None else float(inp.now)

    # dry-run: просто посчитаем
    if inp.dry_run:
        # «подсчёт» на глаз из внутренностей (минимально — прогоним логику без удаления)
        v_dead = sum(1 for obj in st["vrepo"]._store.values() if (obj.meta or {}).get("expire_at", now + 1) < now)  # type: ignore[attr-defined]
        g_dead = 0
        for s, o, k, data in st["grepo"]._g.edges(keys=True, data=True):  # type: ignore[attr-defined]
            exp = (data.get("meta") or {}).get("expire_at")
            if exp is not None and float(exp) < now:
                g_dead += 1
        e_dead = 0
        for r in st["erepo"]._conn.execute("SELECT meta FROM episodes"):  # type: ignore[attr-defined]
            meta = _jload(r[0]) or {}
            exp = meta.get("expire_at")
            if exp is not None and float(exp) < now:
                e_dead += 1
        return {"dry_run": True, "vector": v_dead, "graph": g_dead, "episodes": e_dead}

    # реальный GC
    v = st["vrepo"].gc_expired(now)
    g = st["grepo"].gc_expired(now)
    e = st["erepo"].gc_expired(now)
    return {"removed": {"vector": v, "graph": g, "episodes": e}}
