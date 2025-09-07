from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_retrieve_smoke():
    r = client.post("/retrieve", json={"q": "test"})
    assert r.status_code == 200
    assert "semantic_chunks" in r.json()

def test_writeback_smoke():
    r = client.post("/writeback", json=[{"id":"1","type":"note","payload":{"text":"hi"}}])
    assert r.status_code == 200
    assert r.json()["saved"] == 1

def test_conflicts_smoke():
    r = client.post("/conflicts", json=[])
    assert r.status_code == 200
    assert "conflicts" in r.json()

def test_context_smoke():
    r = client.post("/context")
    assert r.status_code == 200
    assert "sections" in r.json()
