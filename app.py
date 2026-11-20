"""FastAPI web application: generate, view, filter and export timetables.  Run: uvicorn app:app --reload"""
import csv, io, json, sqlite3, time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scheduler.data import generate
from scheduler.engine import Model, run_ga, to_rows, validate

DB = Path(__file__).with_name("timetables.db")
app = FastAPI(title="Automated Timetable Scheduler for Distance Learning Programmes")
STATE = {"problem": None, "model": None, "rows": None, "meta": None}


def db():
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, pop INTEGER,"
                " seed INTEGER, courses INTEGER, hard INTEGER, penalty INTEGER, seconds REAL, rows TEXT)")
    return con


class GenerateRequest(BaseModel):
    population: int = Field(150, ge=20, le=400)
    seed: int = Field(1, ge=0)
    courses: int = Field(120, ge=20, le=150)
    lecturers: int = Field(42, ge=10, le=60)


@app.post("/api/generate")
def generate_timetable(req: GenerateRequest):
    try:
        problem = generate(seed=req.seed, n_courses=req.courses, n_lecturers=req.lecturers)
    except (ValueError, IndexError):
        raise HTTPException(400, "Dataset too large for the available lecturers/cohorts; reduce courses or add lecturers.")
    model = Model(problem)
    res = run_ga(problem, pop_size=req.population, seed=req.seed, model=model)
    rows = to_rows(model, res["genes"])
    issues = validate(model, res["genes"])
    meta = dict(population=req.population, seed=req.seed, courses=req.courses, sessions=model.N,
                hard_violations=res["hard"], penalty=res["penalty"], fitness=round(res["fitness"], 5),
                generations=res["generations"], seconds=round(res["seconds"], 2),
                soft=res["breakdown"], independent_check_conflicts=len(issues))
    STATE.update(problem=problem, model=model, rows=rows, meta=meta)
    con = db(); con.execute("INSERT INTO runs (created,pop,seed,courses,hard,penalty,seconds,rows) VALUES (?,?,?,?,?,?,?,?)",
                            (time.time(), req.population, req.seed, req.courses, res["hard"], res["penalty"],
                             res["seconds"], json.dumps(rows))); con.commit(); con.close()
    return meta


def _need():
    if STATE["rows"] is None:
        raise HTTPException(404, "No timetable yet. POST /api/generate first.")


@app.get("/api/timetable")
def timetable(cohort: str = Query(None), lecturer: str = Query(None), venue: str = Query(None), day: str = Query(None)):
    _need()
    rows = STATE["rows"]
    if cohort: rows = [r for r in rows if cohort.lower() in r["cohorts"].lower()]
    if lecturer: rows = [r for r in rows if lecturer.lower() == r["lecturer"].lower()]
    if venue: rows = [r for r in rows if venue.lower() == r["venue"].lower()]
    if day: rows = [r for r in rows if r["day"].lower() == day.lower()]
    return {"meta": STATE["meta"], "rows": rows}


@app.get("/api/lists")
def lists():
    _need()
    p = STATE["problem"]
    return dict(cohorts=[c.name for c in p.cohorts], lecturers=[l.name for l in p.lecturers], venues=[v.name for v in p.venues])


@app.get("/api/history")
def history():
    con = db()
    out = [dict(id=r[0], created=r[1], population=r[2], seed=r[3], courses=r[4], hard=r[5], penalty=r[6], seconds=r[7])
           for r in con.execute("SELECT id,created,pop,seed,courses,hard,penalty,seconds FROM runs ORDER BY id DESC LIMIT 20")]
    con.close(); return out


@app.get("/api/export.csv")
def export_csv():
    _need()
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(STATE["rows"][0].keys())); w.writeheader(); w.writerows(STATE["rows"])
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=timetable.csv"})


app.mount("/static", StaticFiles(directory=Path(__file__).with_name("web")), name="static")


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("web") / "index.html")
