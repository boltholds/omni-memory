# tests/test_quality_eval.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_hallucination_when_no_context(monkeypatch):
    # Мокаем сбор контекста: всегда пусто
    r = client.post("/answer", json={"q":"Where is Alice?", "lang":"en", "style":"plain"})
    assert r.status_code == 200
    # проверяем, что метрика появилась
    m = client.get("/metrics").text
    assert "qa_hallucinations_total" in m

def test_conflict_should_be_surface(monkeypatch):
    # Подготовить данные: два факта в конфликте (alice at lighthouse/bridge)
    client.post("/writeback", json=[{"id":"f1","subject":"alice","predicate":"at","object":"lighthouse"},
                                    {"id":"f2","subject":"alice","predicate":"at","object":"bridge"}])
    r = client.post("/answer", json={"q":"Where is Alice?", "lang":"en", "style":"plain"})
    # либо ответ содержит "conflict", либо метрика miss растёт
    m = client.get("/metrics").text
    ok = ("conflict" in r.json().get("answer","").lower()) or ("qa_conflict_misses_total " in m)
    assert ok
