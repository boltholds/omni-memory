from fastapi import APIRouter
from infra.vector_repo import VectorStoreRepo
from infra.graph_repo import GraphRepo
from infra.episodic_repo import EpisodicRepo

router = APIRouter(prefix="/admin", tags=["admin"])

def attach_repos(v: VectorStoreRepo, g: GraphRepo, e: EpisodicRepo):
    router.state = {"vrepo": v, "grepo": g, "erepo": e}  # type: ignore[attr-defined]

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
