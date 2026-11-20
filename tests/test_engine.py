import numpy as np, pytest
from scheduler.data import generate
from scheduler.engine import Model, run_ga, validate, W_HARD

@pytest.fixture(scope="module")
def setup():
    p = generate(seed=1); return p, Model(p)

def test_dataset_size(setup):
    p, m = setup
    assert len(p.courses) == 120 and len(p.lecturers) == 42
    assert sum(v.kind != "virtual" for v in p.venues) == 28 and sum(v.kind == "virtual" for v in p.venues) == 6

def test_domains_respect_capacity_and_modality(setup):
    p, m = setup
    for i, (c, _) in enumerate(m.sessions):
        for v in m.dom_venues[i]:
            ven = p.venues[v]
            assert ven.capacity >= m.s_size[i]
            assert (ven.kind == "virtual") == (c.modality == "virtual")

def test_ga_produces_conflict_free_timetable(setup):
    p, m = setup
    r = run_ga(p, pop_size=60, seed=5, model=m, max_gen=120)
    assert r["hard"] == 0
    assert validate(m, r["genes"]) == []

def test_ga_improves_on_seed_solution(setup):
    p, m = setup
    rng = np.random.default_rng(7)
    seed_pen = m.evaluate(m.greedy(rng)[None, :])["penalty"][0]
    r = run_ga(p, pop_size=60, seed=7, model=m, max_gen=150)
    assert r["penalty"] < seed_pen

def test_validator_detects_lecturer_clash(setup):
    p, m = setup
    r = run_ga(p, pop_size=40, seed=2, model=m, max_gen=60)
    genes = r["genes"].copy()
    # force two sessions of the same lecturer into the same slot
    by_lect = {}
    for i, (c, _) in enumerate(m.sessions): by_lect.setdefault(c.lecturer, []).append(i)
    a, b = next(v[:2] for v in by_lect.values() if len(v) >= 2)
    sa = m.dom_slot[a, genes[a]]
    cand = [g for g in range(m.dom_len[b]) if m.dom_slot[b, g] == sa]
    genes[b] = cand[0]
    assert any(x[0] == "lecturer" for x in validate(m, genes))
    assert m.evaluate(genes[None, :])["hard"][0] >= 1

def test_fitness_formula(setup):
    p, m = setup
    g = m.greedy(np.random.default_rng(1))[None, :]
    e = m.evaluate(g)
    assert e["fitness"][0] == pytest.approx(1.0 / (1.0 + e["penalty"][0]))
    assert e["penalty"][0] >= W_HARD * e["hard"][0]

def test_same_seed_is_reproducible(setup):
    p, m = setup
    a = run_ga(p, pop_size=30, seed=11, model=m, max_gen=40)
    b = run_ga(p, pop_size=30, seed=11, model=m, max_gen=40)
    assert a["penalty"] == b["penalty"] and (a["genes"] == b["genes"]).all()
