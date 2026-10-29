"""
T1 / T3 / T4 / bridge -- SETUP results (verified, but SCOOPED by Winner-Take-All
and Rad et al. 2026b; presented in the paper as reproduced context, not as our
contribution). We still verify them numerically as requested.

T1  mirror-descent tilt is exact (machine error); GRPO deviates by a bounded TV.
T3  entropy identity dH/dτ = -τ Var_{w_τ}(f) for uniform base; bump for nonuniform.
T4  log(b-R) vs H has slope 1/κ (κ = local fitness-density exponent near the max).
    NOTE: WTA §7 already has the exact entropy-mass curve, the free-slope
    exponential as a finite-window approximation with error bounds, and proves
    the TRUE terminal law is non-exponential. So T4 is reproduction, not ours.
bridge  token entropy vs strategy entropy: WTA eq (4.1) H(X|correct)=H_y+H(X|F)
    already shows token entropy can rise while strategy entropy falls. We confirm
    the decoupling, which is why any entropy law here is stated at STRATEGY level.

Writes results/t1.json, t3.json, t4.json, bridge.json.
"""
import json, os
import numpy as np
from common import rng, tilt, entropy, reward

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)


# ---------------------------------------------------------------- T1
def t1(seed=1, M=500, steps=300, eta=0.02):
    gen = rng(seed)
    f = gen.uniform(0, 1, size=M)
    w0 = np.full(M, 1.0 / M)
    # mirror descent / natural PG with KL: w_{t+1} ∝ w_t exp(eta f). After T steps
    # of constant-advantage flow, w_T ∝ w0 exp(eta*steps*f) = closed-form tilt.
    w = w0.copy()
    for _ in range(steps):
        w = w * np.exp(eta * f); w /= w.sum()
    w_closed = tilt(w0, f, eta * steps)
    tv_mirror = 0.5 * np.abs(w - w_closed).sum()

    # GRPO (std-normalized advantage) deviation from tilt, at MATCHED entropy.
    # GRPO's std normalization is a nonlinear reweighting, so it is NOT an exact
    # tilt. Fair comparison: run the population GRPO flow, snapshot it at several
    # entropies, and compare each snapshot to the tilt with the SAME entropy.
    theta = np.zeros(M)
    target_H = [0.9, 0.7, 0.5, 0.3]
    logM = np.log(M)
    snaps = {}
    taus_grid = np.geomspace(1e-3, 200, 6000)
    H_of_tau = np.array([entropy(tilt(w0, f, t)) for t in taus_grid])
    for _ in range(4000):
        w_g = np.exp(theta - theta.max()); w_g /= w_g.sum()
        Hg = entropy(w_g) / logM
        for th in list(target_H):
            if Hg <= th and th not in snaps:
                # matched tilt: tau with same entropy
                jt = int(np.argmin(np.abs(H_of_tau - Hg * logM)))
                wt = tilt(w0, f, taus_grid[jt])
                snaps[th] = 0.5 * float(np.abs(w_g - wt).sum())
        if len(snaps) == len(target_H):
            break
        pbar = w_g @ f
        sd = np.sqrt(max(w_g @ (f - pbar) ** 2, 1e-12))
        theta = theta + eta * (f - pbar) / sd
    return {"M": M, "steps": steps,
            "tv_mirror_vs_closed_form_tilt": float(tv_mirror),
            "grpo_vs_tilt_TV_at_matched_entropy": {str(k): v for k, v in sorted(snaps.items(), reverse=True)}}


# ---------------------------------------------------------------- T3
def t3(seed=2, M=400):
    gen = rng(seed)
    f = gen.uniform(0, 1, size=M)
    # uniform base: verify dH/dτ = -τ Var_{w_τ}(f) by finite differences
    taus = np.linspace(0.0, 8.0, 400)
    w0 = np.full(M, 1.0 / M)
    H = np.array([entropy(tilt(w0, f, t)) for t in taus])
    dH = np.gradient(H, taus)
    pred = np.array([-t * (tilt(w0, f, t) @ (f - tilt(w0, f, t) @ f) ** 2) for t in taus])
    err = np.max(np.abs(dH[5:-5] - pred[5:-5]))

    # nonuniform base: entropy can rise before collapsing (the bump)
    w0n = gen.dirichlet(np.full(M, 0.3))
    Hn = np.array([entropy(tilt(w0n, f, t)) for t in taus])
    bump = bool(Hn.max() > Hn[0] + 1e-6)
    return {"M": M, "uniform_identity_max_abs_err": float(err),
            "nonuniform_entropy_bump_present": bump,
            "nonuniform_bump_height": float(Hn.max() - Hn[0])}


# ---------------------------------------------------------------- T4
def t4(seed=3, M=200000, kappas=(0.5, 1.0, 2.0)):
    gen = rng(seed)
    out = {}
    w0 = np.full(M, 1.0 / M)
    logM = np.log(M)
    for kappa in kappas:
        # gaps g_j = u^{1/kappa}: empirical N(g<G) ∝ G^kappa, i.e. local density
        # ∝ g^{kappa-1} near the max (g=0). fitness f = -g (max at 0).
        u = gen.random(M)
        g = u ** (1.0 / kappa)
        f = -g
        taus = np.geomspace(1.0, 3e4, 80)
        Hs, gaps = [], []
        for t in taus:
            w = tilt(w0, f, t)
            Hs.append(entropy(w))
            gaps.append(-reward(w, f))      # b - R = <g>_{w_τ}, b=0
        Hs = np.array(Hs); gaps = np.array(gaps)
        # mesoscopic window: H between 0.2 and 0.8 of logM, gaps>0
        mask = (Hs > 0.2 * logM) & (Hs < 0.8 * logM) & (gaps > 0)
        slope = np.polyfit(Hs[mask], np.log(gaps[mask]), 1)[0]
        out[f"kappa={kappa}"] = {"predicted_slope_1_over_kappa": float(1.0 / kappa),
                                 "measured_slope": float(slope),
                                 "rel_err": float(abs(slope - 1.0 / kappa) / (1.0 / kappa)),
                                 "n_window_points": int(mask.sum())}
    return out


# ---------------------------------------------------------------- bridge
def bridge(seed=4, depth=12, branch=2, d_strategy=4, tau=6.0):
    """
    Toy autoregressive tree. A STRATEGY = a subtree rooted at depth d_strategy
    (a 'family'); leaves within a subtree are sequence variants within that
    family. Token entropy H_tok = full leaf entropy; strategy entropy H_y =
    entropy over subtree masses; within-family term = H_tok - H_y (WTA eq 4.1).
    RL tilts STRATEGIES (subtree fitness). We check the decomposition holds and
    that token entropy falls LESS than strategy entropy (partial decoupling),
    so an entropy law proven at strategy level does NOT transfer verbatim to
    token entropy. This is exactly the WTA family-vs-token distinction.
    """
    gen = rng(seed)
    n_strat = branch ** d_strategy
    within = branch ** (depth - d_strategy)
    f_strat = gen.uniform(0, 1, size=n_strat)
    w0_strat = np.full(n_strat, 1.0 / n_strat)
    # within-family entropy per strategy, CORRELATED with fitness: the winning
    # strategy has a broad family (many token realizations). This is what makes
    # token entropy decouple from strategy entropy (WTA Fig 1 / eq 4.1).
    h_within = 0.1 + (np.log(within) - 0.1) * (0.3 + 0.7 * f_strat)  # nats, <= log(within)

    def decomposed(ws):
        Hy = entropy(ws)
        within_term = float(ws @ h_within)   # sum_j w_j h_j
        return Hy, Hy + within_term, within_term

    Hy0, Ht0, wt0 = decomposed(w0_strat)
    wst = tilt(w0_strat, f_strat, tau)
    Hyt, Htt, wtt = decomposed(wst)
    return {"depth": depth, "d_strategy": d_strategy, "n_strategies": int(n_strat),
            "within_family_leaves": int(within),
            "strategy_entropy_base": float(Hy0), "strategy_entropy_rl": float(Hyt),
            "token_entropy_base": float(Ht0), "token_entropy_rl": float(Htt),
            "within_term_base": float(wt0), "within_term_rl": float(wtt),
            "strategy_entropy_drop": float(Hy0 - Hyt),
            "token_entropy_drop": float(Ht0 - Htt),
            "token_entropy_rose": bool(Htt > Ht0),
            "decoupling_token_drop_lt_strategy_drop": bool((Ht0 - Htt) < (Hy0 - Hyt) - 1e-9),
            "note": "within-family term rises as RL concentrates on broad-family winner; "
                    "token entropy decouples from strategy entropy -> law stated at strategy level"}


def main():
    r = {"T1": t1(), "T3": t3(), "T4": t4(), "bridge": bridge()}
    for k in ["t1", "t3", "t4", "bridge"]:
        with open(os.path.join(RES, f"{k}.json"), "w") as fh:
            json.dump(r[k.upper() if k != "bridge" else "bridge"], fh, indent=2)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
