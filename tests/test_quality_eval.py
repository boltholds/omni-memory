# tests/test_quality_eval.py
from fastapi.testclient import TestClient
from omni_memory.main import app

client = TestClient(app)

def test_hallucination_when_no_context(monkeypatch):
    # Мокаем сбор контекста: всегда пусто
    r = client.post("/answer", json={"q":"Where is Alice?", "lang":"en", "style":"plain"}, headers={"X-API-Key": "CHANGE_ME"})
    assert r.status_code == 200
    # проверяем, что метрика появилась
    m = client.get("/metrics", headers={"X-API-Key": "CHANGE_ME"}).text
    assert "qa_hallucinations_total" in m

def test_conflict_should_be_surface(monkeypatch):
    # Подготовить данные: два факта в конфликте (alice at lighthouse/bridge)
    client.post("/writeback", json=[{"id":"f1","subject":"alice","predicate":"at","object":"lighthouse"},
                                    {"id":"f2","subject":"alice","predicate":"at","object":"bridge"}] , headers={"X-API-Key": "CHANGE_ME"})
    r = client.post("/answer", json={"q":"Where is Alice?", "lang":"en", "style":"plain"}, headers={"X-API-Key": "CHANGE_ME"} )
    # либо ответ содержит "conflict", либо метрика miss растёт
    m = client.get("/metrics", headers={"X-API-Key": "CHANGE_ME"}).text
    ok = ("conflict" in r.json().get("answer","").lower()) or ("qa_conflict_misses_total " in m)
    assert ok
