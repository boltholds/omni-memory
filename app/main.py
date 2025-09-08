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
from infra.llm_factory import build_llm

from app.config import settings
from app.admin import router as admin_router, attach_repos
from app.retriever import Retriever
from app.orchestrator import Orchestrator
from app.writeback import WriteBackService
from app.embeddings import build_embedder
from app.prompting import make_messages


class ContextIn(BaseModel):
    q: str = ""
    max_tokens: Optional[int] = None
    draft: Optional[bool] = False



class RetrieveIn(BaseModel):
    q: str
    k_sem: int = 5
    k_eps: int = 3


class GenerateIn(BaseModel):
    q: str
    context_sections: Optional[List[str]] = None
    temperature: Optional[float] = None

class GenerateOut(BaseModel):
    text: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="omni-memory", version="0.1.0")

    # TODO: CORS for future server
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


    embedder = build_embedder(settings.embedding_backend, settings.embedding_model)
    vrepo = VectorStoreRepo(embedder=embedder)
    grepo = GraphRepo()
    erepo = EpisodicRepo(db_path=settings.sqlite_path)

    
    retriever = Retriever(vrepo, grepo, erepo)
    consistency = SimpleConsistencyEngine()
    orchestrator = Orchestrator(retriever, SimpleConsistencyEngine())
    writeback_svc = WriteBackService(vrepo, grepo, erepo, MemoryPolicy())
    
    attach_repos(vrepo, grepo, erepo, writeback_svc)
    app.include_router(admin_router)

    llm_provider = build_llm()  # может быть и None

    @app.post("/generate", response_model=GenerateOut)
    def generate(inp: GenerateIn):
        if llm_provider is None:
            return GenerateOut(text="LLM provider is not configured (LLM_PROVIDER=none).")
        msgs = make_messages(inp.q, inp.context_sections or [])
        res = llm_provider.generate(msgs, temperature=inp.temperature or settings.llm_temperature)
        return GenerateOut(**res)


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
    def context(inp: Optional[ContextIn] = None):
        q = "" if inp is None else inp.q
        bundle = orchestrator.plan_retrieval(q)

        # временно «подменим» бюджет из запроса
        if inp and inp.max_tokens:
            from app.config import settings as _settings
            old = _settings.context_max_tokens
            try:
                _settings.context_max_tokens = int(inp.max_tokens)
                pack = orchestrator.assemble_context(bundle)
            finally:
                _settings.context_max_tokens = old
        else:
            pack = orchestrator.assemble_context(bundle)

        # черновик ответа при наличии провайдера
        if inp and inp.draft and llm_provider is not None:
            secs = [f"{s.title}:\n{s.body}" for s in pack.sections]
            msgs = make_messages(q, secs)
            res = llm_provider.generate(msgs, temperature=settings.llm_temperature)
            pack.advisories = list(dict.fromkeys(pack.advisories + [f"DRAFT: {res.get('text','').strip()}"]))
        return pack
    
    
    @app.get("/")
    async def redirect_to_docs():
        return RedirectResponse(url="/docs")

    return app

app = create_app()
