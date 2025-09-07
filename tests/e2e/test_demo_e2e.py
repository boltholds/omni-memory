# tests/e2e/test_demo_e2e.py
from __future__ import annotations
from typing import Any, Dict, List
import json
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="function", autouse=True)
def _reset_between_tests(client: TestClient):
    """
    Сбрасываем состояние, если доступен /admin/reset (dev-окружение).
    Если эндпоинта нет (например, в минимальной сборке) — тесты всё равно пойдут.
    """
    resp = client.post("/admin/reset")
    # 404/405 игнорируем — значит, /admin нет или другой метод
    assert resp.status_code in (200, 404, 405)
    yield


def _writeback(client: TestClient, items: List[Dict[str, Any]]):
    r = client.post("/writeback", json=items)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rejected"] == 0, body
    return body


def test_full_demo_flow(client: TestClient):
    # 1) Заливаем два факта и один эпизод
    demo = [
        {"id": "f1", "subject": "alice", "predicate": "at", "object": "lighthouse"},
        {"id": "f2", "subject": "alice", "predicate": "at", "object": "bridge"},
        {
            "id": "ep1",
            "participants": ["Alice", "Nikolai"],
            "summary": "Evening near the lighthouse",
            "events": [
                {"t": 1.0, "event_type": "seen", "summary": "Alice met fisherman Nikolai", "refs": {}}
            ],
            "provenance": {"source": "test"},
        },
    ]
    wb = _writeback(client, demo)
    assert wb["saved"] == 3

    # 2) /retrieve — должен вернуть хотя бы один факт про Alice и эпизод ep1
    rv = client.post("/retrieve", json={"q": "Where is Alice?"})
    assert rv.status_code == 200, rv.text
    bundle = rv.json()
    fact_ids = {f["id"] for f in bundle["facts"]}
    epi_ids = {e["id"] for e in bundle["episodes"]}
    assert {"f1", "f2"} & fact_ids  # хотя бы один факт из двух
    assert "ep1" in epi_ids

    # 3) /conflicts — можно слать смешанный массив, эндпоинт сам вытащит только факты
    cf = client.post("/conflicts", json=demo)
    assert cf.status_code == 200, cf.text
    cbody = cf.json()
    # ожидаем ключ alice::at с вариантами bridge/lighthouse (порядок не гарантирован)
    keys = {c["key"] for c in cbody["conflicts"]}
    assert "alice::at" in keys
    variants = next(c["variants"] for c in cbody["conflicts"] if c["key"] == "alice::at")
    assert set(variants) == {"lighthouse", "bridge"}

    # 4) /context — должен содержать Facts/Conflicts/Episodes секции
    cx = client.post("/context", json={"q": "Alice at lighthouse"})
    assert cx.status_code == 200, cx.text
    ctx = cx.json()
    titles = [s["title"] for s in ctx["sections"]]
    assert "Facts" in titles
    assert "Conflicts" in titles
    assert "Episodes" in titles

    # 5) /healthz — метрики должны увеличиться
    h1 = client.get("/healthz")
    assert h1.status_code == 200
    stats = h1.json().get("stats", {})
    # Минимальные sanity-проверки
    assert stats.get("requests_total", 0) >= 5
    assert stats.get("writeback_saved", 0) >= 3
    assert stats.get("retrieve_calls", 0) >= 1
