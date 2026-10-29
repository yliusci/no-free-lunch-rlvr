"""
C6 -- theory-derived correction to GRPO that removes the pass@k crossing.

Goal (from C2): make the RL success distribution first-order stochastically
dominate the base, F_tau <= F_0, i.e. S-(G)=0 => Delta_k >= 0 for all k.
Sufficient microscopic condition: no prompt loses, p_tau(x) >= p_0(x) for all x.

We test candidate corrections, each a SMALL change to the GRPO objective, on
inverting instances, and report: does it drive #crossings -> 0, and the pass@1
cost. Candidates:

  (A) base-mixture:     w = (1-eps) w_tilt + eps w0
  (B) strategy floor:   w_j >= c * w0_j   (water-filled projection of the tilt)
  (C) coverage bonus:   tilt on f~_j = f_j + lam * g_j,
                         g_j = sum_x q[x,j]/p0(x)   (upweights sole-solvers of
                         hard prompts; = per-prompt advantage reweighting by 1/p0)
  (D) hard-prompt advantage reweighting (the implementable one):
                         effective fitness f~_j = sum_x (q[x,j]/p0(x)) / Z,
                         i.e. each prompt contributes to the advantage with
                         weight 1/p0(x). One line in a GRPO trainer (scale each
                         prompt's advantage by 1/p0). Swept by a blend alpha.

Writes results/c6.json and figures/fig_c6_correction.png.
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
KS = np.unique(np.round(np.logspace(0, 5, 500)).astype(int)); KS = KS[KS >= 1]


def inverting_instance(gen, n_tiers=3, tau=9.0, ppt=100, spt=30, floor=0.02):
    N = n_tiers * ppt; M = n_tiers * spt
    Q = np.full((N, M), floor)
    for t in range(n_tiers):
        pr = slice(t * ppt, (t + 1) * ppt); st = slice(t * spt, (t + 1) * spt)
        succ = 0.9 - 0.25 * t
        Q[pr, st] = np.clip(floor + succ * gen.uniform(0.4, 1.0, size=(ppt, spt)), floor, 0.99)
    for t in range(n_tiers):
        st = slice(t * spt, (t + 1) * spt)
        Q[:, st] = np.clip(Q[:, st] + 0.15 * (1 - t / (n_tiers - 1)), floor, 0.99)
    return Q, tau


def n_cross(p0, pt):
    d = delta_k(p0, pt, KS)
    return count_sign_changes(d, tol=1e-9), d


def floor_project(w_tilt, w0, c):
    """Closest point (L1-ish) with w_j >= c*w0_j, sum=1: raise floored, scale rest."""
    lb = c * w0
    w = np.maximum(w_tilt, lb)
    excess = w.sum() - 1.0
    free = w - lb
    if free.sum() > 0 and excess > 0:
        w = w - excess * free / free.sum()
    w = np.maximum(w, lb)
    return w / w.sum()


def run():
    gen = rng(4242)
    out = {"candidates": {}}
    # ---- single representative instance for the figure + quick scan ----
    Q, tau = inverting_instance(gen)
    M = Q.shape[1]; w0 = np.full(M, 1.0 / M)
    f = Q.mean(axis=0)
    p0 = p_of(w0, Q)
    w_tilt = tilt(w0, f, tau)
    p_tilt = p_of(w_tilt, Q)
    base_nc, _ = n_cross(p0, p_tilt)
    pass1_base = float(passk_curve(p0, [1])[0])
    pass1_tilt = float(passk_curve(p_tilt, [1])[0])
    out["uncorrected"] = {"n_crossings": base_nc, "pass1_base": pass1_base,
                          "pass1_rl": pass1_tilt}

    # coverage term for (C)/(D)
    cov = (Q / p0[:, None]).sum(axis=0)          # g_j = sum_x q[x,j]/p0(x)

    scans = {}
    # (A) mixture
    scans["A_mixture_eps"] = []
    for eps in np.linspace(0, 0.95, 20):
        w = (1 - eps) * w_tilt + eps * w0
        pt = p_of(w, Q); nc, _ = n_cross(p0, pt)
        scans["A_mixture_eps"].append((float(eps), int(nc), float(passk_curve(pt, [1])[0])))
    # (B) floor
    scans["B_floor_c"] = []
    for c in np.linspace(0, 1.0, 21):
        w = floor_project(w_tilt, w0, c)
        pt = p_of(w, Q); nc, _ = n_cross(p0, pt)
        scans["B_floor_c"].append((float(c), int(nc), float(passk_curve(pt, [1])[0])))
    # (C) coverage-bonus tilt
    scans["C_covbonus_lam"] = []
    for lam in np.geomspace(1e-3, 5.0, 20):
        ftil = f + lam * cov / cov.mean()
        w = tilt(w0, ftil, tau)
        pt = p_of(w, Q); nc, _ = n_cross(p0, pt)
        scans["C_covbonus_lam"].append((float(lam), int(nc), float(passk_curve(pt, [1])[0])))
    # (D) hard-prompt advantage reweighting: effective fitness blends f with the
    # 1/p0-weighted marginal fitness f_hard_j = sum_x (q[x,j]/p0(x)) q[x,j]-ish.
    f_hard = (Q / p0[:, None]).sum(axis=0)
    f_hard = f_hard / f_hard.max() * f.max()
    scans["D_hardreweight_alpha"] = []
    for a in np.linspace(0, 1.0, 21):
        ftil = (1 - a) * f + a * f_hard
        w = tilt(w0, ftil, tau)
        pt = p_of(w, Q); nc, _ = n_cross(p0, pt)
        scans["D_hardreweight_alpha"].append((float(a), int(nc), float(passk_curve(pt, [1])[0])))
    out["scans"] = scans

    # smallest-cost setting that reaches 0 crossings, per candidate
    def best_zero(scan, pass1_free):
        zeros = [(s, nc, p1) for (s, nc, p1) in scan if nc == 0]
        if not zeros:
            return None
        # maximize retained pass@1
        s, nc, p1 = max(zeros, key=lambda r: r[2])
        return {"setting": s, "pass1": p1, "pass1_cost_vs_rl": pass1_tilt - p1}
    out["candidates"] = {
        "A_mixture": best_zero(scans["A_mixture_eps"], pass1_tilt),
        "B_floor": best_zero(scans["B_floor_c"], pass1_tilt),
        "C_covbonus": best_zero(scans["C_covbonus_lam"], pass1_tilt),
        "D_hardreweight": best_zero(scans["D_hardreweight_alpha"], pass1_tilt),
    }
    return out, (Q, p0, w0, f, f_hard, tau, p_tilt)


def guarantee_test(seed=99, n_instances=400, alpha=1.0):
    """Does candidate D (full hard-reweight, alpha=1) give Delta_k>=0 across many
    inverting instances? Report fraction with 0 crossings and FOSD (F_tau<=F_0)."""
    gen = rng(seed)
    zero_cross = 0; fosd = 0; pass1_costs = []
    for _ in range(n_instances):
        n_tiers = int(gen.integers(2, 5))
        Q, tau = inverting_instance(gen, n_tiers=n_tiers, tau=gen.uniform(5, 12))
        M = Q.shape[1]; w0 = np.full(M, 1.0 / M)
        f = Q.mean(axis=0); p0 = p_of(w0, Q)
        f_hard = (Q / p0[:, None]).sum(axis=0); f_hard = f_hard / f_hard.max() * f.max()
        w = tilt(w0, (1 - alpha) * f + alpha * f_hard, tau)
        pt = p_of(w, Q)
        nc, _ = n_cross(p0, pt)
        if nc == 0:
            zero_cross += 1
        sg, G, _ = cdf_diff_sign_changes(p0, pt)
        if np.all(G <= 1e-9):
            fosd += 1
        pass1_costs.append(float(passk_curve(p_of(tilt(w0, f, tau), Q), [1])[0] - passk_curve(pt, [1])[0]))
    return {"n_instances": n_instances, "alpha": alpha,
            "frac_zero_crossings": zero_cross / n_instances,
            "frac_FOSD_Ftau_le_F0": fosd / n_instances,
            "median_pass1_cost_vs_uncorrected_rl": float(np.median(pass1_costs))}


def constrained_update(Q, w0, f, step=None):
    """
    C6 (principled form): reward-improving update that lets NO prompt regress.
        max_dw <f, dw>  s.t.  Q dw >= 0 (per-prompt non-regression),
                              w0+dw >= 0,  sum dw = 0.
    Any feasible dw gives p_tau(x) >= p0(x) for all x => F_tau <= F_0 => Delta_k>=0
    for all k (by C2). We solve the LP; the trainer form is a per-prompt
    non-regression penalty / Lagrangian on the GRPO advantage.
    """
    from scipy.optimize import linprog
    M = w0.size
    # linprog minimizes c^T x; we maximize <f,dw> => c = -f
    A_ub = -Q                      # Q dw >= 0  ->  -Q dw <= 0
    b_ub = np.zeros(Q.shape[0])
    A_eq = np.ones((1, M)); b_eq = np.zeros(1)
    bounds = [(-w0[j], 1 - w0[j]) for j in range(M)]
    res = linprog(c=-f, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return w0.copy()
    dw = res.x
    if step is not None:          # take a fractional step along the feasible dir
        dw = step * dw
    w = np.clip(w0 + dw, 0, None); w = w / w.sum()
    return w


def guarantee_test_E(seed=321, n_instances=400):
    """Does the constrained non-regression update give Delta_k>=0 across many
    inverting instances? Report fraction 0-crossings, fraction FOSD, pass@1 gain."""
    gen = rng(seed)
    zero_cross = 0; fosd = 0; pass1_gain = []; worst_loss = []
    for _ in range(n_instances):
        n_tiers = int(gen.integers(2, 5))
        Q, tau = inverting_instance(gen, n_tiers=n_tiers, tau=gen.uniform(5, 12))
        M = Q.shape[1]; w0 = np.full(M, 1.0 / M)
        f = Q.mean(axis=0); p0 = p_of(w0, Q)
        w = constrained_update(Q, w0, f)
        pt = p_of(w, Q)
        nc, _ = n_cross(p0, pt)
        if nc == 0:
            zero_cross += 1
        sg, G, _ = cdf_diff_sign_changes(p0, pt)
        if np.all(G <= 1e-9):
            fosd += 1
        pass1_gain.append(float(passk_curve(pt, [1])[0] - passk_curve(p0, [1])[0]))
        worst_loss.append(float((pt - p0).min()))
    return {"n_instances": n_instances,
            "frac_zero_crossings": zero_cross / n_instances,
            "frac_FOSD_Ftau_le_F0": fosd / n_instances,
            "median_pass1_gain_vs_base": float(np.median(pass1_gain)),
            "worst_per_prompt_loss_max": float(np.max(worst_loss))}


def relaxed_update(Q, w0, f, gamma):
    """
    C6 tunable form: max pass@1 gain subject to each prompt dropping by at most
    (1-gamma) of its base success:  Q dw >= -(1-gamma) p0.
    gamma=1 -> strict non-regression (no crossing, no gain here).
    gamma=0 -> unconstrained reward max (full crossing).
    The knob gamma trades pass@1 gain against how far the crossing is pushed out.
    Trainer form: a per-prompt success-floor penalty weighted toward hard prompts.
    """
    from scipy.optimize import linprog
    M = w0.size
    p0 = Q @ w0
    A_ub = -Q
    b_ub = (1 - gamma) * p0
    A_eq = np.ones((1, M)); b_eq = np.zeros(1)
    bounds = [(-w0[j], 1 - w0[j]) for j in range(M)]
    res = linprog(c=-f, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return w0.copy()
    w = np.clip(w0 + res.x, 0, None)
    return w / w.sum()


def tradeoff_curve(seed=7, n_instances=120, gammas=None):
    """Average pass@1 gain and first-crossing k* as a function of gamma."""
    if gammas is None:
        gammas = [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
    gen = rng(seed)
    insts = []
    for _ in range(n_instances):
        n_tiers = int(gen.integers(2, 5))
        Q, tau = inverting_instance(gen, n_tiers=n_tiers, tau=gen.uniform(5, 12))
        insts.append((Q, np.full(Q.shape[1], 1.0 / Q.shape[1])))
    rows = {}
    for g in gammas:
        gains, kstars, ncs = [], [], []
        for Q, w0 in insts:
            f = Q.mean(axis=0); p0 = p_of(w0, Q)
            w = relaxed_update(Q, w0, f, g)
            pt = p_of(w, Q)
            gains.append(float(passk_curve(pt, [1])[0] - passk_curve(p0, [1])[0]))
            d = delta_k(p0, pt, KS)
            sc = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
            ncs.append(count_sign_changes(d, tol=1e-9))
            kstars.append(int(KS[sc[0]]) if sc.size else int(KS[-1]) + 1)
        rows[str(g)] = {"median_pass1_gain": float(np.median(gains)),
                        "frac_no_crossing": float(np.mean(np.array(ncs) == 0)),
                        "median_first_crossing_k": float(np.median(kstars))}
    return rows


def main():
    out, fig_data = run()
    out["guarantee_test_E_constrained"] = guarantee_test_E()
    out["C6_tradeoff_gamma"] = tradeoff_curve()
    out["guarantee_test_D_alpha1"] = guarantee_test(alpha=1.0)
    out["guarantee_test_D_alpha0.6"] = guarantee_test(alpha=0.6)
    with open(os.path.join(RES, "c6.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in ["uncorrected", "candidates",
          "guarantee_test_D_alpha1", "guarantee_test_D_alpha0.6"]}, indent=2))

    # figure: base, GRPO (crossing), strict non-regression (no crossing, Δ_k>=0),
    # and an intermediate gamma (crossing pushed out) -> the no-free-lunch frontier
    Q, p0, w0, f, f_hard, tau, p_tilt = fig_data
    p_strict = p_of(constrained_update(Q, w0, f, step=None), Q)       # gamma=1
    p_mid = p_of(relaxed_update(Q, w0, f, gamma=0.9), Q)              # gamma=0.9
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.8))
    ax[0].semilogx(KS, passk_curve(p0, KS), color="#444", lw=2, label="base")
    ax[0].semilogx(KS, passk_curve(p_tilt, KS), color="#c1121f", lw=2, label="GRPO (crossing)")
    ax[0].semilogx(KS, passk_curve(p_mid, KS), color="#e9a000", lw=2, label=r"C6 $\gamma$=0.9 (pushed out)")
    ax[0].semilogx(KS, passk_curve(p_strict, KS), color="#2a9d8f", lw=2, label=r"C6 strict (no crossing)")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("pass@k"); ax[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    ax[0].set_title("The crossing is the price of pass@1 gain", fontsize=10.5)
    ax[1].semilogx(KS, delta_k(p0, p_tilt, KS), color="#c1121f", lw=2, label="GRPO")
    ax[1].semilogx(KS, delta_k(p0, p_mid, KS), color="#e9a000", lw=2, label=r"$\gamma$=0.9")
    ax[1].semilogx(KS, delta_k(p0, p_strict, KS), color="#2a9d8f", lw=2, label="strict")
    ax[1].axhline(0, color="#888", lw=.8)
    ax[1].set_xlabel("k"); ax[1].set_ylabel(r"$\Delta_k$"); ax[1].legend(frameon=False, fontsize=8.5)
    ax[1].set_title(r"strict C6: $\Delta_k \geq 0$ for all k (zero pass@1 gain)", fontsize=10.5)
    for a in ax: a.grid(alpha=.22, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_c6_correction.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
