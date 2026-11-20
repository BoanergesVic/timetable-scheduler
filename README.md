# Automated Timetable Scheduler for Distance Learning Programmes

CSC 476 semester project, Group 18, Department of Computer Science, University of Ibadan.

A web application that generates conflict-free weekend timetables for distance learning programmes.
It models timetabling as a Constraint Satisfaction Problem (CSP) and searches for a good solution with an
elitist Genetic Algorithm (GA).

## What it does
- Builds a weekend timetable (Friday evening, Saturday, Sunday: 10 two-hour blocks).
- **Hard constraints** (never violated): no lecturer, venue or student cohort in two places at once; venue capacity respected.
- **Soft constraints** (penalised): lecturer availability, intensive courses not in late Sunday slots,
  hybrid courses placed in hybrid-capable halls, no cohort with three back-to-back blocks.
- Supports physical halls, hybrid halls and virtual rooms.
- Web page to generate a timetable, filter it by cohort, lecturer, venue or day, and export it as CSV.
- Stores each run in a local SQLite database.

## How it works
1. **CSP preprocessing** (`scheduler/engine.py`, class `Model`): builds, for every session, the list of
   (time slot, venue) options that satisfy capacity and delivery mode, and seeds half of the starting
   population with conflict-aware constructive timetables.
2. **Genetic Algorithm** (`run_ga`): fitness = 1 / (1 + penalty), with
   `penalty = 1000*hard + 50*unavailable + 20*pedagogical + 20*modality + 10*consecutive`;
   tournament selection, two-point crossover, adaptive violation-guided mutation, elitism.
3. An independent checker (`validate`) re-checks the final timetable for clashes.

## Run it
    pip install -r requirements.txt
    uvicorn app:app --reload        # then open http://127.0.0.1:8000
    pytest                          # 10 automated tests
    python run_benchmark.py         # reproduces every result reported in Chapter Four

## Project layout
    app.py                FastAPI application (REST API + web page)
    web/index.html        Browser interface
    scheduler/data.py     Problem model and synthetic dataset generator
    scheduler/engine.py   CSP preprocessing, genetic algorithm, validator
    run_benchmark.py      Experiments (population size, ablation, baseline, scalability)
    results/              Benchmark output (CSV and JSON)
    tests/                PyTest tests for the engine and the API

## Limitations
- The dataset is **synthetic**: generated to resemble a distance learning centre (120 courses, 42 lecturers,
  28 physical halls, 6 virtual rooms, 8 programmes x 5 levels). It is not real institutional data.
- No login or user roles yet; data is generated, not entered through forms.
- SQLite is used for storage. PostgreSQL, a React interface, background workers and PDF/Excel/iCal export are future work.

## Team
Group 18. See the team participation page in the project report.
