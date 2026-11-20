"""CSP preprocessing + elitist Genetic Algorithm timetable engine (NumPy-vectorised)."""
import time
import numpy as np
from .data import SLOTS, DAY_OF_SLOT, LATE_SLOTS

W_HARD, W_AVAIL, W_PED, W_MOD, W_CONSEC = 1000, 50, 20, 20, 10
S = len(SLOTS)


class Model:
    """Flattens a Problem into arrays; builds CSP domains (preprocessing step)."""

    def __init__(self, problem):
        self.p = problem
        V = len(problem.venues)
        self.V = V
        self.v_cap = np.array([v.capacity for v in problem.venues])
        self.v_kind = [v.kind for v in problem.venues]
        sess = []
        for c in problem.courses:
            for k in range(c.sessions):
                sess.append((c, k))
        self.sessions = sess
        self.N = N = len(sess)
        self.s_lect = np.array([c.lecturer for c, _ in sess])
        self.s_intensive = np.array([c.intensive for c, _ in sess])
        self.s_size = np.array([sum(problem.cohorts[i].size for i in c.cohorts) for c, _ in sess])
        pairs = [(i, ch) for i, (c, _) in enumerate(sess) for ch in c.cohorts]
        self.pair_s = np.array([a for a, _ in pairs]); self.pair_c = np.array([b for _, b in pairs])
        self.C = len(problem.cohorts)
        self.avail = np.array([l.available for l in problem.lecturers], dtype=bool)
        self.L = len(problem.lecturers)
        self.late = np.zeros(S, dtype=bool); self.late[list(LATE_SLOTS)] = True
        self.day = np.array(DAY_OF_SLOT)
        # --- CSP preprocessing: domain of allowed venues per session ---
        self.dom_venues = []
        for i, (c, _) in enumerate(sess):
            ok = []
            for vi, v in enumerate(problem.venues):
                if v.capacity < self.s_size[i]: continue
                if c.modality == "virtual" and v.kind != "virtual": continue
                if c.modality in ("in-person", "hybrid") and v.kind == "virtual": continue
                ok.append(vi)
            if not ok:
                ok = [int(np.argmax(self.v_cap))]
            self.dom_venues.append(np.array(ok))
        self.dom_len = np.array([len(d) * S for d in self.dom_venues])
        D = self.dom_len.max()
        self.dom_slot = np.zeros((N, D), dtype=np.int16)
        self.dom_ven = np.zeros((N, D), dtype=np.int16)
        for i, d in enumerate(self.dom_venues):
            slots = np.repeat(np.arange(S), len(d)); vens = np.tile(d, S)
            n = len(slots)
            self.dom_slot[i, :n] = slots; self.dom_ven[i, :n] = vens
        self.s_needs_hybrid = np.array([c.modality == "hybrid" for c, _ in sess])
        self.v_is_hybrid = np.array([k == "hybrid" for k in self.v_kind])

    def decode(self, genes):
        idx = np.arange(self.N)
        return self.dom_slot[idx, genes], self.dom_ven[idx, genes]

    def evaluate(self, pop):
        """pop: (P, N) gene matrix -> dict of per-chromosome penalties + conflict mask."""
        P = pop.shape[0]
        idx = np.arange(self.N)
        slot = self.dom_slot[idx, pop].astype(np.int64)
        ven = self.dom_ven[idx, pop].astype(np.int64)
        row = np.arange(P)[:, None]
        key = (self.s_lect[None, :] * S + slot) + row * (self.L * S)
        cnt = np.bincount(key.ravel(), minlength=P * self.L * S)
        lect_c = cnt[key]
        lect = (np.maximum(cnt - 1, 0)).reshape(P, -1).sum(1)
        key2 = (ven * S + slot) + row * (self.V * S)
        cnt2 = np.bincount(key2.ravel(), minlength=P * self.V * S)
        ven_c = cnt2[key2]
        venue = (np.maximum(cnt2 - 1, 0)).reshape(P, -1).sum(1)
        ps, pc = self.pair_s, self.pair_c
        slot_p = slot[:, ps]
        key3 = (pc[None, :] * S + slot_p) + row * (self.C * S)
        cnt3 = np.bincount(key3.ravel(), minlength=P * self.C * S)
        coh_c_pair = cnt3[key3]
        cohort = (np.maximum(cnt3 - 1, 0)).reshape(P, -1).sum(1)
        coh_c = np.zeros((P, self.N), dtype=np.int64)
        np.maximum.at(coh_c, (np.repeat(np.arange(P), len(ps)), np.tile(ps, P)), coh_c_pair.ravel())
        cap = (self.v_cap[ven] < self.s_size[None, :]).sum(1)
        hard = lect + venue + cohort + cap
        ua_g = ~self.avail[self.s_lect[None, :], slot]
        ped_g = self.s_intensive[None, :] & self.late[slot]
        mod_g = self.s_needs_hybrid[None, :] & ~self.v_is_hybrid[ven]
        unavail = ua_g.sum(1); ped = ped_g.sum(1); mod = mod_g.sum(1)
        occ = (cnt3.reshape(P, self.C, S) > 0)
        consec = np.zeros(P, dtype=np.int64)
        for d in range(3):
            cols = np.where(self.day == d)[0]
            o = occ[:, :, cols]
            for a in range(len(cols) - 2):
                consec += (o[:, :, a] & o[:, :, a + 1] & o[:, :, a + 2]).sum(1)
        penalty = W_HARD * hard + W_AVAIL * unavail + W_PED * ped + W_MOD * mod + W_CONSEC * consec
        conflict = (lect_c > 1) | (ven_c > 1) | (coh_c > 1)
        bad = conflict | ua_g | ped_g | mod_g
        return dict(bad=bad, hard=hard, lect=lect, venue=venue, cohort=cohort, cap=cap, unavail=unavail,
                    ped=ped, mod=mod, consec=consec, penalty=penalty,
                    fitness=1.0 / (1.0 + penalty), conflict=conflict)

    def greedy(self, rng):
        """Conflict-aware constructive seed (randomised, largest sessions first)."""
        order = np.argsort(-(self.s_size + rng.random(self.N) * 40))
        lec_used, ven_used, coh_used = set(), set(), set()
        genes = np.zeros(self.N, dtype=np.int64)
        for i in order:
            cohs = self.pair_c[self.pair_s == i]
            n = self.dom_len[i]
            cand = rng.permutation(n)[:60]
            best = None
            for g in cand:
                s = int(self.dom_slot[i, g]); v = int(self.dom_ven[i, g])
                bad = ((self.s_lect[i], s) in lec_used) or ((v, s) in ven_used) or any((c, s) in coh_used for c in cohs)
                if not bad and self.avail[self.s_lect[i], s]:
                    best = g; break
                if not bad and best is None: best = g
            if best is None: best = cand[0]
            genes[i] = best
            s = int(self.dom_slot[i, best]); v = int(self.dom_ven[i, best])
            lec_used.add((self.s_lect[i], s)); ven_used.add((v, s))
            for c in cohs: coh_used.add((c, s))
        return genes


def run_ga(problem, pop_size=150, max_gen=1000, pc=0.85, pm_base=0.02, pm_max=0.05,
           elite=2, tournament=3, seed=0, patience=150, greedy_frac=0.5, trials=8, model=None, progress=None):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    m = model or Model(problem)
    N = m.N
    pop = np.empty((pop_size, N), dtype=np.int64)
    ng = int(pop_size * greedy_frac)
    for k in range(pop_size):
        pop[k] = m.greedy(rng) if k < ng else (rng.random(N) * m.dom_len).astype(np.int64)
    ev = m.evaluate(pop)
    gen_hard0 = None; best_pen = ev["penalty"].min(); stall = 0; gen_opt = 0
    history = []
    pm = pm_base
    for g in range(1, max_gen + 1):
        order = np.argsort(ev["penalty"])
        best = order[0]
        if ev["hard"][best] == 0 and gen_hard0 is None: gen_hard0 = g - 1
        history.append((g - 1, float(ev["fitness"][best]), int(ev["hard"][best])))
        if progress: progress(g, float(ev["fitness"][best]), int(ev["hard"][best]))
        if ev["penalty"][best] == 0: break
        if stall >= patience and gen_hard0 is not None: break
        cand = rng.integers(0, pop_size, size=(pop_size, tournament))
        win = cand[np.arange(pop_size), np.argmin(ev["penalty"][cand], axis=1)]
        parents = pop[win]
        kids = parents.copy()
        for k in range(0, pop_size - 1, 2):
            if rng.random() < pc:
                a, b = sorted(rng.integers(0, N, 2))
                kids[k, a:b], kids[k + 1, a:b] = parents[k + 1, a:b].copy(), parents[k, a:b].copy()
        # adaptive, violation-guided mutation with best-of-M trials (memetic step):
        # each trial moves 1..3 sessions (more when the search stagnates), biased to violated genes
        k_ev = m.evaluate(kids)
        w = 1.0 + 9.0 * k_ev["bad"]
        cs = np.cumsum(w, axis=1)
        trial = np.repeat(kids[:, None, :], trials + 1, axis=1)          # P x (M+1) x N
        n_mut = 1 + min(2, stall // 40)
        for _ in range(n_mut):
            r = rng.random((pop_size, trials)) * cs[:, -1:]
            gi = (cs[:, None, :] < r[:, :, None]).sum(axis=2)            # P x M gene index
            gi = np.minimum(gi, N - 1)
            newv = (rng.random((pop_size, trials)) * m.dom_len[gi]).astype(np.int64)
            rows = np.arange(pop_size)[:, None]; cols = np.arange(1, trials + 1)[None, :]
            trial[rows, cols, gi] = newv
        flat = trial.reshape(pop_size * (trials + 1), N)
        t_ev = m.evaluate(flat)
        pen = t_ev["penalty"].reshape(pop_size, trials + 1)
        pick = np.argmin(pen, axis=1)
        kids = trial[np.arange(pop_size), pick]
        kids[:elite] = pop[order[:elite]]
        pop = kids
        ev = m.evaluate(pop)
        cur = ev["penalty"].min()
        if cur < best_pen: best_pen = cur; stall = 0; gen_opt = g
        else: stall += 1
        pm = min(pm_max, pm_base + 0.03 * min(1.0, stall / 50.0))
    order = np.argsort(ev["penalty"]); best = order[0]
    res = dict(genes=pop[best].copy(), fitness=float(ev["fitness"][best]), hard=int(ev["hard"][best]),
               penalty=int(ev["penalty"][best]), generations=len(history), gen_hard0=gen_hard0,
               gen_opt=gen_opt, seconds=time.perf_counter() - t0, history=history,
               breakdown={k: int(ev[k][best]) for k in ("lect", "venue", "cohort", "cap", "unavail", "ped", "mod", "consec")})
    res["model"] = m
    return res


def to_rows(model, genes):
    slot, ven = model.decode(np.asarray(genes))
    rows = []
    for i, (c, k) in enumerate(model.sessions):
        d, t = model.p.slots[int(slot[i])]
        v = model.p.venues[int(ven[i])]
        rows.append(dict(course=c.code, title=c.title, session=k + 1,
                         lecturer=model.p.lecturers[c.lecturer].name,
                         cohorts=", ".join(model.p.cohorts[x].name for x in c.cohorts),
                         day=d, time=t, slot=int(slot[i]), venue=v.name, venue_kind=v.kind,
                         modality=c.modality))
    rows.sort(key=lambda r: (r["slot"], r["venue"]))
    return rows


def validate(model, genes):
    """Independent (non-vectorised) conflict checker used by tests and the API."""
    slot, ven = model.decode(np.asarray(genes))
    issues = []
    seen_l, seen_v, seen_c = {}, {}, {}
    for i, (c, k) in enumerate(model.sessions):
        s, v = int(slot[i]), int(ven[i])
        for key, store, label in (((c.lecturer, s), seen_l, "lecturer"), ((v, s), seen_v, "venue")):
            if key in store: issues.append((label, c.code, store[key]))
            store[key] = c.code
        for ch in c.cohorts:
            key = (ch, s)
            if key in seen_c: issues.append(("cohort", c.code, seen_c[key]))
            seen_c[key] = c.code
        if model.v_cap[v] < model.s_size[i]: issues.append(("capacity", c.code, None))
    return issues
