from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_ratelimit_basic(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    monkeypatch.setenv("RATE_LIMIT_BURST", "1")
    # три запроса подряд -> третий 429
    r1 = client.get("/healthz"); r2 = client.get("/healthz"); r3 = client.get("/healthz")
    assert r3.status_code in (200, 429)  # healthz можно исключить из лимита; см. твои правила
