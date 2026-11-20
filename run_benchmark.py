"""Reproduces every number reported in Chapter Four. Run:  python run_benchmark.py"""
import csv, json, time, statistics as st
import numpy as np
from scheduler.data import generate
from scheduler.engine import Model, run_ga, validate

problem = generate(seed=1)            # 120 courses, 42 lecturers, 28 physical + 6 virtual venues, 5 levels
model = Model(problem)
out = {"dataset": dict(courses=len(problem.courses), sessions=model.N, lecturers=len(problem.lecturers),
                       physical_venues=sum(v.kind != "virtual" for v in problem.venues),
                       virtual_rooms=sum(v.kind == "virtual" for v in problem.venues),
                       cohorts=len(problem.cohorts), slots=len(problem.slots))}

# ---- Experiment 1: population size ----
rows = []
rid = 0
for N in (50, 100, 150, 200):
    for seed in (1, 2, 3):
        rid += 1
        r = run_ga(problem, pop_size=N, seed=seed, model=model)
        assert validate(model, r["genes"]) == []
        rows.append(dict(run=f"R-{rid:02d}", pop=N, seed=seed, gen_hard0=r["gen_hard0"], gen_best=r["gen_opt"],
                         generations=r["generations"], hard=r["hard"], penalty=r["penalty"],
                         fitness=round(r["fitness"], 5), seconds=round(r["seconds"], 2), **r["breakdown"]))
        print(rows[-1])
with open("results/benchmark_population.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
out["population_runs"] = rows

# ---- Experiment 2: ablation (N = 150, 3 seeds) ----
abl = []
for seed in (1, 2, 3):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    seeds = np.array([model.greedy(rng) for _ in range(150)])
    ev = model.evaluate(seeds); b = int(np.argmin(ev["penalty"]))
    abl.append(dict(method="CSP-style greedy seeding only (best of 150)", seed=seed, hard=int(ev["hard"][b]),
                    penalty=int(ev["penalty"][b]), seconds=round(time.perf_counter() - t0, 2)))
    r = run_ga(problem, pop_size=150, seed=seed, greedy_frac=0.0, model=model)
    abl.append(dict(method="GA only (random initial population)", seed=seed, hard=r["hard"], penalty=r["penalty"], seconds=round(r["seconds"], 2)))
    r = run_ga(problem, pop_size=150, seed=seed, model=model)
    abl.append(dict(method="Hybrid (CSP seeding + GA)", seed=seed, hard=r["hard"], penalty=r["penalty"], seconds=round(r["seconds"], 2)))
for a in abl: print(a)
out["ablation"] = abl

# ---- Experiment 3: simulated manual-style baseline (first-fit, rooms checked, nothing else) ----
base = []
for seed in (1, 2, 3):
    rng = np.random.default_rng(seed)
    genes = np.zeros(model.N, dtype=np.int64)
    used = set()
    order = rng.permutation(model.N)
    for i in order:                       # one person, no lecturer/cohort checking, only avoids double-booked rooms
        for g in range(model.dom_len[i]):
            s, v = int(model.dom_slot[i, g]), int(model.dom_ven[i, g])
            if (v, s) not in used:
                used.add((v, s)); genes[i] = g; break
    ev = model.evaluate(genes[None, :])
    base.append(dict(seed=seed, lecturer_clashes=int(ev["lect"][0]), cohort_clashes=int(ev["cohort"][0]),
                     venue_clashes=int(ev["venue"][0]), unavailable=int(ev["unavail"][0]), penalty=int(ev["penalty"][0])))
for b in base: print(b)
out["baseline"] = base

# ---- Experiment 4: scalability ----
sc = []
for nc, nl in ((60, 21), (120, 42), (150, 53)):
    try:
        p = generate(seed=1, n_courses=nc, n_lecturers=nl)
    except Exception as e:
        print("skip", nc, e); continue
    m = Model(p)
    r = run_ga(p, pop_size=150, seed=1, model=m)
    assert validate(m, r["genes"]) == []
    sc.append(dict(courses=nc, sessions=m.N, hard=r["hard"], penalty=r["penalty"], seconds=round(r["seconds"], 2)))
    print(sc[-1])
out["scalability"] = sc
json.dump(out, open("results/benchmark.json", "w"), indent=1)
print("saved results/benchmark.json")
