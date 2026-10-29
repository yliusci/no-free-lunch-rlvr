"""
T* -- the no-free-lunch theorem, CORRECTED.

A free lunch at an interior policy w is a tangent dw (1^T dw = 0) with Q dw >= 0
and Q dw != 0 (a Pareto improvement of per-prompt success; since
f^T dw = (1/N) 1^T Q dw, "mean up" is equivalent to "some prompt up"). This does
NOT depend on w.

CORRECT characterization (Stiemke's lemma). Exactly one holds:
  (I) exists dw with Q dw >= 0, Q dw != 0, 1^T dw = 0        [free lunch]
  (II) exists y > 0 (strict) with Q^T y = mu * 1             [no free lunch]
i.e. no free lunch iff there is a strictly positive reweighting of prompts under
which all strategies have equal weighted fitness.

SUFFICIENT structural condition: EXCLUSIVE specialization -- each strategy j has a
prompt set X_j^excl of positive mass solved ONLY by j (q_{xj}>0, q_{xj'}=0 for
j'!=j on X_j^excl). Then any dw with dw_j<0 strictly lowers X_j^excl, so no
Pareto improvement exists.

"Unique best" (each strategy is the top scorer on some prompts, others nonzero)
is NOT sufficient: the quasi-generalist counterexample below has a free lunch.

This script: verifies the Stiemke primal/dual equivalence, confirms exclusive
specialization => no free lunch on 1000 instances AND on adversarial generators,
and shows unique-best / quasi-generalist / overlapping generators DO admit free
lunches. Writes results/tstar.json.
"""
import json, os
import numpy as np
from scipy.optimize import linprog
from common import rng, tilt, p_of, delta_k, passk_curve, count_sign_changes, cdf_diff_sign_changes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
KS = np.unique(np.round(np.logspace(0, 5, 500)).astype(int)); KS = KS[KS >= 1]


# ---------------- primal / dual ----------------
def freelunch_gain(Q):
    """max f^T dw s.t. Q dw >= 0, 1^T dw = 0, |dw|<=1. >tol => free lunch exists."""
    M = Q.shape[1]; f = Q.mean(axis=0)
    res = linprog(c=-f, A_ub=-Q, b_ub=np.zeros(Q.shape[0]),
                  A_eq=np.ones((1, M)), b_eq=[0.0],
                  bounds=[(-1, 1)] * M, method="highs")
    return float(-res.fun) if res.success else None


def stiemke_feasible(Q, eps=1.0):
    """Is there y >= eps (strict y>0) with Q^T y = mu*1 ? LP feasibility.
    Variables: y (N) >= eps, mu free (split). Equalities Q^T y - mu 1 = 0 (M)."""
    N, M = Q.shape
    # vars: y (N), mu+ , mu-  ; Q^T y - (mu+ - mu-) 1 = 0
    Aeq = np.hstack([Q.T, -np.ones((M, 1)), np.ones((M, 1))])
    beq = np.zeros(M)
    c = np.zeros(N + 2)
    bounds = [(eps, None)] * N + [(0, None), (0, None)]
    res = linprog(c=c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    return bool(res.success)


# ---------------- generators ----------------
def exclusive_specialized(gen, n_strat=10, pps=30, floor=0.0, off=0.0):
    """Each strategy has an EXCLUSIVE block solved only by it (others = 0 there).
    Plus optional shared prompts anyone can attempt."""
    N = n_strat * pps; M = n_strat
    Q = np.full((N, M), off)
    for j in range(n_strat):
        pr = slice(j * pps, (j + 1) * pps)
        Q[pr, :] = 0.0                                   # only j solves these
        Q[pr, j] = gen.uniform(0.3, 0.95, size=pps)
    return np.clip(Q, 0.0, 0.99)


def unique_best(gen, n_strat=10, pps=30):
    """Each strategy is the UNIQUE top scorer on its block, but other strategies
    also solve those prompts (nonzero). This is the insufficient old A1."""
    N = n_strat * pps; M = n_strat
    Q = gen.uniform(0.3, 0.95, size=(N, M))              # everyone decent everywhere
    for j in range(n_strat):
        pr = slice(j * pps, (j + 1) * pps)
        Q[pr, j] = np.maximum(Q[pr, :].max(axis=1) + gen.uniform(0.02, 0.1, pps), Q[pr, j])
    return np.clip(Q, 0.01, 0.99)


def quasi_generalist(gen, n_strat=10, pps=30):
    """Adversarial: near-generalist strategies (high everywhere) that are unique
    best on a block but easily replaced -> free lunches abound."""
    N = n_strat * pps; M = n_strat
    base = gen.uniform(0.7, 0.95, size=(N, M))
    for j in range(n_strat):
        pr = slice(j * pps, (j + 1) * pps)
        base[pr, j] = np.minimum(0.99, base[pr, :].max(axis=1) + 0.02)
    return np.clip(base, 0.4, 0.99)


def overlapping(gen, n_strat=10, pps=30, overlap=2):
    """Each prompt block is solvable by a strategy AND its `overlap` neighbors."""
    N = n_strat * pps; M = n_strat
    Q = np.full((N, M), 0.05)
    for j in range(n_strat):
        pr = slice(j * pps, (j + 1) * pps)
        for d in range(overlap + 1):
            Q[pr, (j + d) % M] = gen.uniform(0.4, 0.95, size=pps) * (1 - 0.2 * d)
    return np.clip(Q, 0.01, 0.99)


def dominant_specialized(gen, n_strat=10, pps=30, d=0.7, eps=None):
    """
    Dominant specialization (realistic, no exact zeros): for each j there are
    prompts of positive mass with q_xj >= d and q_xj' <= eps for j'!=j, with
    (M-1)*eps < d/2. Sufficient for no free lunch (Neumann-series dual construction).
    """
    M = n_strat
    if eps is None:
        eps = 0.4 * d / max(1, M - 1)          # ensures (M-1)eps = 0.4 d < d/2
    N = n_strat * pps
    Q = gen.uniform(0.0, eps, size=(N, M))      # everyone weak by default (<= eps)
    for j in range(n_strat):
        pr = slice(j * pps, (j + 1) * pps)
        Q[pr, :] = gen.uniform(0.0, eps, size=(pps, M))
        Q[pr, j] = gen.uniform(d, min(0.99, d + 0.25), size=pps)   # j strong (>= d)
    return np.clip(Q, 0.0, 0.99)


def threshold_counterexample():
    """f_j < R(tau) is NOT the right drop threshold; Lambda(tau)/tau is.
    f=(0.9,0.75,0.5,0.1), uniform w0, tau=5: strategy f=0.75 has f < R(tau) yet
    its weight INCREASES because 0.75 > Lambda(tau)/tau."""
    f = np.array([0.9, 0.75, 0.5, 0.1]); w0 = np.full(4, 0.25); tau = 5.0
    R0 = float(w0 @ f)
    wt = tilt(w0, f, tau); R = float(wt @ f)
    Lam = np.log(w0 @ np.exp(tau * f)); Lam_over_tau = float(Lam / tau)
    ratio = float(wt[1] / w0[1])                 # strategy f=0.75
    return {"R0_base_mean": R0, "R_tilted_mean": R, "Lambda_over_tau": Lam_over_tau,
            "strategy_f": 0.75, "weight_ratio": ratio,
            "f_below_R_tau": bool(0.75 < R), "f_below_Lambda_over_tau": bool(0.75 < Lam_over_tau),
            "weight_increased": bool(ratio > 1.0),
            "note": "0.75 < R(tau) but 0.75 > Lambda/tau, so weight rises: R(tau) is the wrong threshold; R0 (Jensen) is safe"}


def near_generalist(gen, n_strat=10, pps=30):
    """Adversarial: one near-generalist strategy good on ALL prompts makes the
    specialists redundant, so concentrating on it Pareto-improves -> free lunch.
    This is the soft form of the dominance witness (Thm 2c) and the structure of
    the hand counterexample."""
    N = n_strat * pps; M = n_strat
    Q = np.full((N, M), 0.05)
    for j in range(1, n_strat):                          # specialists 1..M-1
        pr = slice(j * pps, (j + 1) * pps)
        Q[pr, j] = gen.uniform(0.5, 0.9, size=pps)
    # strategy 0 is a near-generalist: covers every block almost as well as its
    # specialist, and owns block 0
    Q[:, 0] = gen.uniform(0.80, 0.97, size=N)
    Q[0 * pps:1 * pps, 0] = gen.uniform(0.9, 0.99, size=pps)
    return np.clip(Q, 0.05, 0.99)


COUNTEREXAMPLE = np.array([[1.00, 0.90, 0.96],
                           [0.90, 1.00, 0.96],
                           [0.50, 0.50, 1.00]])


# ---------------- tests ----------------
def scan(genfn, gen, n, want_no_fl):
    nofl = 0; dual = 0; agree = 0
    for _ in range(n):
        Q = genfn(gen, n_strat=int(gen.integers(4, 12)))
        g = freelunch_gain(Q)
        no_fl = (g is not None and g <= 1e-7)
        d = stiemke_feasible(Q)
        if no_fl: nofl += 1
        if d: dual += 1
        if no_fl == d: agree += 1
    return {"n": n, "frac_no_free_lunch": nofl / n,
            "frac_stiemke_dual_feasible": dual / n,
            "primal_dual_agreement": agree / n}


def fine_coverage_gen(gen, holds, n_strat=8, pps=40, tau=10.0):
    """
    Build an exclusive-specialized instance with a HARD block (minimal base
    success). Fine Coverage (primitive): the hardest prompts are solved only by a
    strategy whose marginal fitness f_j < R (tilted mean). 'holds': the hard
    specialist is low-fitness (tilt starves it) -> crossover. not holds: the hard
    prompts are ALSO covered by a high-fitness strategy -> floor kept -> no
    crossover. Returns p0, ptau, and whether min p dropped.
    """
    N = n_strat * pps; M = n_strat
    Q = np.zeros((N, M))
    # easy specialists (high success, high marginal fitness)
    for j in range(1, n_strat):
        pr = slice(j * pps, (j + 1) * pps)
        Q[pr, j] = gen.uniform(0.55, 0.9, size=pps)
    # hard block (block 0): low base success
    hb = slice(0, pps)
    Q[hb, 0] = gen.uniform(0.04, 0.12, size=pps)         # only strategy 0 solves them
    if not holds:
        # give a high-fitness easy specialist (say strat 1) backup coverage of the
        # hard block, so the tilt (which keeps strat 1) keeps the hard prompts up
        Q[hb, 1] = gen.uniform(0.25, 0.45, size=pps)
    w0 = np.full(M, 1.0 / M); f = Q.mean(axis=0)
    p0 = p_of(w0, Q); pt = p_of(tilt(w0, f, tau), Q)
    return p0, pt


def fine_coverage_test(seed=5, n=400):
    gen = rng(seed)
    res = {}
    for holds in (True, False):
        cross = 0; mindrop = 0
        for _ in range(n):
            p0, pt = fine_coverage_gen(gen, holds)
            nc = count_sign_changes(delta_k(p0, pt, KS), tol=1e-9)
            if nc >= 1: cross += 1
            if pt.min() < p0.min() - 1e-9: mindrop += 1
        res["holds" if holds else "fails"] = {
            "frac_crossover": cross / n,
            "frac_min_success_dropped": mindrop / n}
    return res


def main():
    gen = rng(11)
    out = {}
    # the user's counterexample: unique-best but free lunch
    g_ce = freelunch_gain(COUNTEREXAMPLE)
    out["counterexample_unique_best"] = {
        "Q": COUNTEREXAMPLE.tolist(),
        "freelunch_gain": g_ce,
        "has_free_lunch": bool(g_ce is not None and g_ce > 1e-7),
        "stiemke_dual_feasible": stiemke_feasible(COUNTEREXAMPLE),
        "note": "unique-best specialization holds yet a free lunch exists"}
    # exclusive specialization: should be NO free lunch everywhere, incl adversarial sizes
    out["exclusive_specialized"] = scan(exclusive_specialized, gen, 1000, True)
    # insufficient / adversarial generators: free lunch should be common
    out["unique_best"] = scan(unique_best, gen, 500, False)
    out["quasi_generalist"] = scan(quasi_generalist, gen, 500, False)
    out["overlapping"] = scan(overlapping, gen, 500, False)
    out["near_generalist_adversarial"] = scan(near_generalist, gen, 500, False)
    out["dominant_specialized"] = scan(dominant_specialized, gen, 2000, True)
    out["fine_coverage"] = fine_coverage_test()
    out["fine_coverage_threshold_counterexample"] = threshold_counterexample()

    with open(os.path.join(RES, "tstar.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
