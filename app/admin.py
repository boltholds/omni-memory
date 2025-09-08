from fastapi import APIRouter
from infra.vector_repo import VectorStoreRepo
from infra.graph_repo import GraphRepo
from infra.episodic_repo import EpisodicRepo,_jload
from app.export_import import export_memory, import_memory
from typing import Any, Dict
from app.writeback import WriteBackService
from pydantic import BaseModel
import time
import logging

class LogLevelIn(BaseModel):
    level: str  # DEBUG|INFO|WARNING|ERROR

class VectorPathIn(BaseModel):
    dir: str

class GCRequest(BaseModel):
    dry_run: bool = False
    now: float | None = None

router = APIRouter(prefix="/admin", tags=["admin"])

def attach_repos(vrepo, grepo, erepo, writeback: WriteBackService | None = None):
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

@router.post("/gc")
def gc(inp: GCRequest):
    st = router.state  # type: ignore[attr-defined]
    now = time.time() if inp.now is None else float(inp.now)

    # dry-run: просто посчитаем
    if inp.dry_run:
        # «подсчёт» на глаз из внутренностей (минимально — прогоним логику без удаления)
        v_dead = sum(1 for obj in st["vrepo"]._store.values() if (obj.meta or {}).get("expire_at", now+1) < now)  # type: ignore[attr-defined]
        g_dead = 0
        for s,o,k,data in st["grepo"]._g.edges(keys=True, data=True):  # type: ignore[attr-defined]
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