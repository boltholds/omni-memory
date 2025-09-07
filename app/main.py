from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from domain.models import RetrievalBundle, WriteReport, ConflictReport, ContextPack, MemoryObject, Fact

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
        return {"status": "ok"}

    @app.post("/retrieve", response_model=RetrievalBundle)
    def retrieve(query: dict):
        # query = {"q": "..."}
        return RetrievalBundle()

    @app.post("/writeback", response_model=WriteReport)
    def writeback(objs: list[MemoryObject]):
        return WriteReport(saved=len(objs), rejected=0, reasons=[])

    @app.post("/conflicts", response_model=ConflictReport)
    def conflicts(facts: list[Fact]):
        return ConflictReport(conflicts=[])

    @app.post("/context", response_model=ContextPack)
    def context():
        return ContextPack(sections=[], advisories=[])
    
    
    @app.get("/")
    async def redirect_to_docs():
        return RedirectResponse(url="/docs")

    return app

app = create_app()
