"""Problem model and synthetic dataset generator.

The weekend contact period is divided into 10 two-hour blocks:
Friday evening (1), Saturday (5) and Sunday (4).
"""
from dataclasses import dataclass, field
import random

SLOTS = [
    ("Fri", "17:00-19:00"),
    ("Sat", "08:00-10:00"), ("Sat", "10:00-12:00"), ("Sat", "12:00-14:00"),
    ("Sat", "14:00-16:00"), ("Sat", "16:00-18:00"),
    ("Sun", "08:00-10:00"), ("Sun", "10:00-12:00"), ("Sun", "12:00-14:00"),
    ("Sun", "14:00-16:00"),
]
DAY_OF_SLOT = [0, 1, 1, 1, 1, 1, 2, 2, 2, 2]
LATE_SLOTS = {8, 9}            # late Sunday blocks (fatigue)
MODALITIES = ("in-person", "virtual", "hybrid")


@dataclass
class Venue:
    id: int
    name: str
    kind: str            # "physical", "hybrid" or "virtual"
    capacity: int


@dataclass
class Lecturer:
    id: int
    name: str
    available: list      # list[bool] per slot


@dataclass
class Cohort:
    id: int
    name: str            # e.g. "P1-300L"
    size: int


@dataclass
class Course:
    id: int
    code: str
    title: str
    lecturer: int
    cohorts: list        # cohort ids
    sessions: int        # number of 2-hour sessions per weekend
    modality: str
    intensive: bool


@dataclass
class Problem:
    venues: list
    lecturers: list
    cohorts: list
    courses: list
    slots: list = field(default_factory=lambda: list(SLOTS))


def generate(seed=1, n_courses=120, n_lecturers=42, n_physical=28, n_virtual=6,
             n_programmes=8, n_levels=5):
    rng = random.Random(seed)
    venues = []
    vid = 0
    for i in range(n_physical):
        if i < 6:
            cap = rng.choice([200, 250, 300, 400, 500, 600])
        elif i < 16:
            cap = rng.choice([80, 100, 120, 150])
        else:
            cap = rng.choice([30, 40, 50, 60])
        kind = "hybrid" if i % 3 == 0 else "physical"
        venues.append(Venue(vid, f"Hall {i+1}", kind, cap)); vid += 1
    for i in range(n_virtual):
        venues.append(Venue(vid, f"Virtual Room {i+1}", "virtual", rng.choice([300, 500, 800, 1000]))); vid += 1

    lecturers = []
    for i in range(n_lecturers):
        avail = [rng.random() < 0.72 for _ in SLOTS]
        if sum(avail) < 5:
            for j in rng.sample(range(len(SLOTS)), 5): avail[j] = True
        lecturers.append(Lecturer(i, f"Lecturer {i+1}", avail))

    cohorts = []
    cid = 0
    for p in range(n_programmes):
        for lv in range(1, n_levels + 1):
            cohorts.append(Cohort(cid, f"P{p+1}-{lv}00L", rng.randint(15, 180))); cid += 1

    load = {l.id: 0 for l in lecturers}
    cohort_load = {c.id: 0 for c in cohorts}
    courses = []
    for i in range(n_courses):
        sessions = 2 if rng.random() < 0.2 else 1
        cands = [l for l in lecturers if load[l.id] + sessions <= 5]
        lec = rng.choice(cands)
        load[lec.id] += sessions
        k = 2 if rng.random() < 0.25 else 1
        pool = [c for c in cohorts if cohort_load[c.id] + sessions <= 6]
        chosen = rng.sample(pool, k)
        for c in chosen: cohort_load[c.id] += sessions
        modality = rng.choices(MODALITIES, weights=[55, 25, 20])[0]
        courses.append(Course(i, f"CRS{100+i}", f"Course {i+1}", lec.id,
                              [c.id for c in chosen], sessions, modality,
                              rng.random() < 0.3))
    return Problem(venues, lecturers, cohorts, courses)
