# tests/test_metrics_smoke.py
from fastapi.testclient import TestClient
from omni_memory.main import app

client = TestClient(app)

def test_metrics_increment_on_endpoints():
    # стартовая метрика
    h1 = client.get("/healthz").json()["stats"]
    req0 = h1["requests_total"]
    saved0 = h1["writeback_saved"]
    retr0 = h1["retrieve_calls"]

    # 1) writeback: сохраним одну простую заметку без PII
    wb = client.post("/writeback", json=[{"id":"n1","type":"note","text":"stone bridge by river"}])
    assert wb.status_code == 200
    # 2) retrieve: дернем пустой поиск
    rv = client.post("/retrieve", json={"q":"bridge","k_sem":3,"k_eps":2})
    assert rv.status_code == 200

    # проверим приращения
    h2 = client.get("/healthz").json()["stats"]
    assert h2["requests_total"] >= req0 + 3  # /healthz + /writeback + /retrieve + /healthz
    assert h2["writeback_saved"] >= saved0 + 1
    assert h2["retrieve_calls"] >= retr0 + 1
