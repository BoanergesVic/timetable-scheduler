from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_timetable_before_generate_is_404_or_ok():
    r = client.get("/api/timetable")
    assert r.status_code in (200, 404)

def test_generate_view_filter_export():
    r = client.post("/api/generate", json={"population": 40, "seed": 3, "courses": 60, "lecturers": 21})
    assert r.status_code == 200
    meta = r.json()
    assert meta["hard_violations"] == 0 and meta["independent_check_conflicts"] == 0
    rows = client.get("/api/timetable").json()["rows"]
    assert len(rows) == meta["sessions"]
    day = client.get("/api/timetable", params={"day": "Sat"}).json()["rows"]
    assert day and all(x["day"] == "Sat" for x in day)
    csv = client.get("/api/export.csv")
    assert csv.status_code == 200 and csv.text.startswith("course,")
    assert client.get("/api/history").json()

def test_rejects_invalid_population():
    assert client.post("/api/generate", json={"population": 5}).status_code == 422
