from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest
from fastapi import APIRouter, Response, Request

# Глобальный реестр (или оставь дефолтный, если не нужно мульти-процесс)
REG = CollectorRegistry(auto_describe=True)

HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests", ["method", "route", "status"], registry=REG
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "route", "status"],
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 13),
    registry=REG,
)

LLM_CALLS = Counter("llm_calls_total", "LLM calls", ["model", "status"], registry=REG)
LLM_LATENCY = Histogram(
    "llm_call_duration_seconds", "LLM call duration", ["model", "status"], 
    buckets=(1, 10, 100, 1000, 2000, 3000, 4000, 6000), registry=REG
)

VECTOR_SIZE = Gauge("vector_store_objects", "Objects in vector store", registry=REG)
GRAPH_FACTS = Gauge("graph_facts_total", "Facts in graph store", registry=REG)
EPISODES = Gauge("episodes_total", "Episodes stored", registry=REG)

router = APIRouter()

@router.get("/metrics")
def metrics(_: Request) -> Response:
    data = generate_latest(REG)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
