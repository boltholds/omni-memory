from fastapi.testclient import TestClient
from omni_memory.main import app

def test_request_id_header_present():
    c = TestClient(app)
    r = c.get("/healthz", headers={"X-Request-ID":"test-req-123", "X-API-Key": "CHANGE_ME"})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == "test-req-123"

def test_admin_log_level():
    c = TestClient(app)
    r = c.post("/admin/log-level", json={"level":"DEBUG"},headers={"X-API-Key": "CHANGE_ME"})
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
