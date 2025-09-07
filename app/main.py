from typing import Optional,List, Dict, Any

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from domain.models import RetrievalBundle, WriteReport, ConflictReport, ContextPack, MemoryObject, Fact
from domain.policy import MemoryPolicy

from infra.vector_repo import VectorStoreRepo
from infra.graph_repo import GraphRepo
from infra.episodic_repo import EpisodicRepo
from infra.consistency import SimpleConsistencyEngine
from infra.metrics import metrics, timeit_request

from app.retriever import Retriever
from app.orchestrator import Orchestrator
from app.writeback import WriteBackService


class RetrieveIn(BaseModel):
    q: str
    k_sem: int = 5
    k_eps: int = 3

def create_app() -> FastAPI:
    app = FastAPI(title="omni-memory", version="0.1.0")

    # CORS на будущее (например, для веб-клиента)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.middleware("http")
    async def metrics_mw(request, call_next):
        metrics.inc("requests_total", 1)
        with timeit_request():
            response = await call_next(request)
        return response

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "stats": metrics.snapshot()}


    vrepo = VectorStoreRepo()
    grepo = GraphRepo()
    erepo = EpisodicRepo()
    retriever = Retriever(vrepo, grepo, erepo)
    consistency = SimpleConsistencyEngine()
    orchestrator = Orchestrator(retriever, SimpleConsistencyEngine())
    writeback_svc = WriteBackService(vrepo, grepo, erepo, MemoryPolicy())

    @app.post("/retrieve", response_model=RetrievalBundle)
    def retrieve(inp: RetrieveIn):
        out = retriever.retrieve(inp.q, inp.k_sem, inp.k_eps)
        metrics.inc("retrieve_calls", 1)
        return out

    @app.post("/writeback", response_model=WriteReport)
    def writeback(objs: list[dict]):
        rep = writeback_svc.write(objs)
        metrics.inc("writeback_saved", rep.saved)
        metrics.inc("writeback_rejected", rep.rejected)
        return rep

    @app.post("/conflicts", response_model=ConflictReport)
    def conflicts(items: List[Dict[str, Any]]):
        facts: List[Fact] = []
        for it in items:
            if all(k in it for k in ("subject", "predicate", "object")):
                try:
                    facts.append(Fact.model_validate(it))
                except Exception:
                    pass  # пропускаем сломанные
        return consistency.detect_conflicts(facts)



    @app.post("/context", response_model=ContextPack)
    def context(inp: Optional[dict] = None):
        q = "" if inp is None else str(inp.get("q", ""))
        bundle = orchestrator.plan_retrieval(q)
        return orchestrator.assemble_context(bundle)
    
    
    @app.get("/")
    async def redirect_to_docs():
        return RedirectResponse(url="/docs")

    return app

app = create_app()
