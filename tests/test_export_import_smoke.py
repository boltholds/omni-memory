from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_export_import_roundtrip_smoke():
    demo = [
        {"id":"f1","subject":"alice","predicate":"at","object":"lighthouse"},
        {"id":"ep1","participants":["Alice"],"summary":"near lighthouse","events":[{"t":1.0,"event_type":"seen","summary":"ok","refs":{}}]},
        {"id":"n1","type":"note","text":"stone bridge over river"}
    ]
    # импорт
    r_imp = client.post("/admin/import", json={"facts":[demo[0]], "episodes":[demo[1]], "notes":[demo[2]]}, headers={"X-API-Key": "CHANGE_ME"})
    assert r_imp.status_code == 200
    # экспорт
    r_exp = client.get("/admin/export", headers={"X-API-Key": "CHANGE_ME"})
    assert r_exp.status_code == 200
    body = r_exp.json()
    assert body["facts"] and body["episodes"] and body["notes"]
