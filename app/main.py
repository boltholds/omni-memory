from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from domain.models import RetrievalBundle, WriteReport, ConflictReport, ContextPack, MemoryObject, Fact
from infra.vector_repo import VectorStoreRepo
from infra.graph_repo import GraphRepo
from infra.episodic_repo import EpisodicRepo
from infra.consistency import SimpleConsistencyEngine
from app.retriever import Retriever
from app.orchestrator import Orchestrator


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

    @app.get("/healthz")
    def healthz():
        return {"status": "ok","version":"is_mock"}


    vrepo = VectorStoreRepo()
    grepo = GraphRepo()
    erepo = EpisodicRepo()
    retriever = Retriever(vrepo, grepo, erepo)
    orchestrator = Orchestrator(retriever, SimpleConsistencyEngine())

    @app.post("/retrieve", response_model=RetrievalBundle)
    def retrieve(inp: RetrieveIn):
        return retriever.retrieve(inp.q, inp.k_sem, inp.k_eps)

    @app.post("/writeback", response_model=WriteReport)
    def writeback(objs: list[MemoryObject]):
        return WriteReport(saved=len(objs), rejected=0, reasons=[])

    @app.post("/conflicts", response_model=ConflictReport)
    def conflicts(facts: list[Fact]):
        return ConflictReport(conflicts=[])

    @app.post("/context", response_model=ContextPack)
    def context(inp: dict):
        bundle = orchestrator.plan_retrieval(inp.get("q", ""))
        return orchestrator.assemble_context(bundle)
    
    
    @app.get("/")
    async def redirect_to_docs():
        return RedirectResponse(url="/docs")

    return app

app = create_app()
