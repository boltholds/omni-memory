from fastapi import APIRouter
from infra.vector_repo import VectorStoreRepo
from infra.graph_repo import GraphRepo
from infra.episodic_repo import EpisodicRepo
from app.export_import import export_memory, import_memory
from typing import Any, Dict
from app.writeback import WriteBackService
from pydantic import BaseModel

class VectorPathIn(BaseModel):
    dir: str



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