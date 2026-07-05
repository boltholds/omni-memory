from fastapi.testclient import TestClient
from omni_memory.main import app


client = TestClient(app)

def test_request_id_echo():
    rid = "abc-123"
    r = client.get("/healthz", headers={"X-Request-Id": rid})
    assert r.headers.get("X-Request-Id") == rid
