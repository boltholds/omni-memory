from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)

def test_admin_requires_api_key():
    r = client.post("/admin/reset")
    assert r.status_code in (401, 404)  # если admin выключен, будет 404

def test_admin_ok_with_key(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "k1")
    settings.admin_api_key = "k1"
    r = client.post("/admin/reset", headers={"X-API-Key": "k1"})
    assert r.status_code in (200, 204)  # зависит от реализации
