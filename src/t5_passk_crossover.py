"""
T5 / R2 / R3 -- pass@k crossover: the CORE, novel result (not in Winner-Take-All).

R2  Bound:   #(pass@k crossings between base and RL) <= S-(G),
             G(p) = F_tau(p) - F_0(p), the CDF difference of per-prompt success.
             Proof route: Δ_k = -k ∫ (1-p)^{k-1} G(p) dp; the kernel is totally
             positive in (k,p), so by Karlin's variation-diminishing property the
             number of sign changes of k↦Δ_k is <= S-(G). We check 0 violations.

R3  Multiplicity: on a shared-strategy ensemble, multiple crossings are common
             (reported fraction). Uniqueness is NOT universal; it holds when RL
             preserves the difficulty order of prompts (p_tau a monotone map of
             p_0). Yang & Chen (2609.22547) is that special case.

R4 (necessity of sharing) is checked in t2_tabular_monotone.py.

Writes results/t5.json and figures/fig_main_passk.png.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import rng, tilt, p_of, delta_k, passk_curve, count_sign_changes, cdf_diff_sign_changes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

KS = np.unique(np.round(np.logspace(0, 5, 500)).astype(int))   # k = 1 .. 1e5
KS = KS[KS >= 1]


def make_instance(gen, n_tiers, tau, prompts_per_tier=120, strat_per_tier=40,
                  floor=0.02):
    """
    Shared-strategy instance with difficulty TIERS. Each tier t has its own
    prompts and its own 'specialist' strategies (high on tier-t prompts, floor
    elsewhere). The RL tilt chases marginal fitness f_j; to produce a pass@k
    INVERSION, some prompts must LOSE mass, which happens when the tier whose
    specialists the tilt starves is the SOLE solver of those prompts.

    Key design choice for R3: a tier's intrinsic success level (how high its
    specialists push p) is DECOUPLED from its specialists' marginal fitness
    (what the tilt chases), via an independent generic-competence term per tier.
    When the two are not aligned, the p_tau-vs-p0 relation is non-monotone and
    G(p) = F_tau - F_0 can change sign several times -> multiple crossings.

    Returns p0, ptau, f.
    """
    N = n_tiers * prompts_per_tier
    M = n_tiers * strat_per_tier
    Q = np.full((N, M), floor)
    for t in range(n_tiers):
        pr = slice(t * prompts_per_tier, (t + 1) * prompts_per_tier)
        st = slice(t * strat_per_tier, (t + 1) * strat_per_tier)
        succ = gen.uniform(0.25, 0.95)        # how well tier-t specialists solve tier t
        block = floor + succ * gen.uniform(0.3, 1.0, size=(prompts_per_tier, strat_per_tier))
        Q[pr, st] = np.clip(block, floor, 0.99)
    # independent generic competence per tier's strategies, broadcast to all
    # prompts. This is the marginal-fitness knob, DECOUPLED from `succ` above.
    for t in range(n_tiers):
        st = slice(t * strat_per_tier, (t + 1) * strat_per_tier)
        generic = gen.uniform(0.0, 0.35)      # independent of this tier's succ
        Q[:, st] = np.clip(Q[:, st] + generic, floor, 0.99)
    f = Q.mean(axis=0)
    w0 = np.full(M, 1.0 / M)
    wt = tilt(w0, f, tau)
    return p_of(w0, Q), p_of(wt, Q), f


def crossings_of_delta(p0, ptau):
    d = delta_k(p0, ptau, KS)
    return count_sign_changes(d, tol=1e-9), d


# ---------------------------------------------------------------- R2 + R3
def run_ensemble(seed=20261028, n_instances=3000):
    gen = rng(seed)
    cross_counts = []
    violations = 0
    for _ in range(n_instances):
        n_tiers = int(gen.integers(1, 7))      # 1..6 difficulty tiers
        tau = gen.uniform(2.0, 15.0)
        p0, ptau, _ = make_instance(gen, n_tiers, tau, prompts_per_tier=80,
                                    strat_per_tier=30)
        nc, _ = crossings_of_delta(p0, ptau)
        sg, _, _ = cdf_diff_sign_changes(p0, ptau)
        cross_counts.append(nc)
        if nc > sg:                            # Karlin bound violated
            violations += 1
    cc = np.array(cross_counts)
    inverters = cc >= 1
    return {
        "n_instances": int(n_instances),
        "frac_0_crossings": float(np.mean(cc == 0)),
        "frac_unique_crossing": float(np.mean(cc == 1)),
        "frac_multiple_crossings": float(np.mean(cc >= 2)),
        "max_crossings_seen": int(cc.max()),
        "n_inverters": int(inverters.sum()),
        "among_inverters_frac_multiple": float(np.mean(cc[inverters] >= 2)) if inverters.any() else 0.0,
        "karlin_checked": int(n_instances),
        "karlin_violations": int(violations),
    }


# ---------------------------------------------------------------- order preserved
def run_order_preserving(seed=7, n_instances=1035, N=600):
    """
    RL preserves difficulty order: p_tau = g(p_0), g monotone increasing,
    g crosses the identity once (sharpening around a threshold). Theory: exactly
    one pass@k crossing. We count how many instances are exactly unique.
    """
    gen = rng(seed)
    unique = 0
    nonunique = 0
    for _ in range(n_instances):
        # base success distribution over prompts
        a, b = gen.uniform(0.5, 3.0), gen.uniform(0.5, 3.0)
        p0 = gen.beta(a, b, size=N)
        # linear tilt around a pivot c: ptau - p0 = delta*(p0 - c), monotone in p0,
        # one sign change across prompts (help easy, hurt hard). pivot below mean
        # so pass@1 improves; guarantees a single pass@k crossing in theory.
        # pivot strictly between min and mean => Δ_1>0 (pass@1 up) and the
        # hardest prompts are lowered (large-k down): exactly one crossing.
        pmin, pmean = p0.min(), p0.mean()
        c = pmin + gen.uniform(0.15, 0.85) * (pmean - pmin)
        delta = gen.uniform(0.25, 0.8)
        ptau = np.clip(p0 + delta * (p0 - c), 1e-6, 1 - 1e-6)
        nc, _ = crossings_of_delta(p0, ptau)
        if nc == 1:
            unique += 1
        else:
            nonunique += 1
    return {
        "n_instances": int(n_instances),
        "n_unique": int(unique),
        "n_nonunique": int(nonunique),
        "frac_unique": float(unique / n_instances),
    }


# ---------------------------------------------------------------- figure instance
def figure_double_crossing(seed=20261028):
    """Find and plot one instance with 2 crossings (the headline phenomenon)."""
    gen = rng(seed)
    best = None
    for _ in range(8000):
        n_tiers = int(gen.integers(2, 5))
        tau = gen.uniform(4.0, 12.0)
        p0, ptau, f = make_instance(gen, n_tiers, tau)
        nc, d = crossings_of_delta(p0, ptau)
        sg, _, _ = cdf_diff_sign_changes(p0, ptau)
        if nc == 2 and sg == 2:
            best = (p0, ptau, d, tau)
            break
    return best


def run_tightness(m_max=6):
    """
    Tightness of the Karlin bound: for each target m, build a signed measure
    ν = μ_tau - μ_0 (as point masses at distinct success levels) whose sign
    pattern alternates m times across p. Theory (R2) says #pass@k crossings
    <= m; we check the bound is ACHIEVED (exactly m crossings). This replaces
    any ensemble-dependent 'fraction multiple' claim: multiplicity is a real,
    fully controllable phenomenon governed by S-(G).
    """
    gen = rng(987)
    rows = []
    any_violation = False
    # Bound (CDF form): #pass@k crossings <= S-(G), G = F_tau - F_0.
    # With n levels and mass-conserving alternating ν, S-(G) = n-2. We target
    # T = S-(G) crossings using n = T+2 levels and confirm the bound is achieved.
    for T in range(1, m_max + 1):
        n = T + 2
        best_nc = -1
        achieved = False
        for _ in range(6000):
            lo = gen.uniform(-6.0, -5.0); hi = gen.uniform(-0.3, 0.0)
            s = np.geomspace(10 ** lo, 10 ** hi, n) * gen.uniform(0.8, 1.25, size=n)
            s = np.sort(s)
            q = np.exp(-s)
            mag = gen.uniform(0.6, 1.4, size=n)
            nu = np.array([(-1) ** i for i in range(n)]) * mag
            nu = nu - nu.mean()                               # Σν = 0
            # CDF difference G at the level points (cumulative of ν ordered by p)
            order = np.argsort(1 - q)                          # by success level p
            G = np.cumsum(nu[order])
            sg = count_sign_changes(G[:-1], tol=1e-15)         # last ~0 by Σν=0
            d = -(nu[None, :] * q[None, :] ** KS[:, None]).sum(axis=1)
            nc = count_sign_changes(d, tol=1e-18)
            if nc > sg:
                any_violation = True
            best_nc = max(best_nc, nc)
            if nc == T and sg == T:
                achieved = True
                break
        rows.append({"target_crossings": int(T), "n_levels": int(n),
                     "max_crossings_found": int(best_nc), "achieved": bool(achieved)})
    return {"per_T": rows, "any_bound_violation_in_search": bool(any_violation)}


def main():
    out = {"ks_min": int(KS.min()), "ks_max": int(KS.max()), "n_k": int(KS.size)}
    out["R2_R3_ensemble"] = run_ensemble()
    out["R2_tightness_constructive"] = run_tightness()
    out["R3_order_preserving"] = run_order_preserving()

    fig_inst = figure_double_crossing()
    out["figure_found_double_crossing"] = bool(fig_inst is not None)

    with open(os.path.join(RES, "t5.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # ---- main figure ----
    if fig_inst is not None:
        p0, ptau, d, tau = fig_inst
        pk0 = passk_curve(p0, KS); pkt = passk_curve(ptau, KS)
        fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
        ax[0].semilogx(KS, pk0, label="base policy", lw=2, color="#444")
        ax[0].semilogx(KS, pkt, label=f"RL (tilt τ={tau:.1f})", lw=2, color="#c1121f")
        ax[0].set_xlabel("k"); ax[0].set_ylabel("pass@k")
        ax[0].set_title("Two pass@k crossings", fontsize=11)
        ax[0].legend(frameon=False, fontsize=9)
        ax[1].semilogx(KS, d, color="#c1121f", lw=2)
        ax[1].axhline(0, color="#888", lw=.8)
        ax[1].set_xlabel("k"); ax[1].set_ylabel(r"$\Delta_k$ = pass@k(RL) - pass@k(base)")
        ax[1].set_title("Advantage reverses twice", fontsize=11)
        for a in ax: a.grid(alpha=.25, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_main_passk.png"), dpi=150)
        plt.close(fig)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
