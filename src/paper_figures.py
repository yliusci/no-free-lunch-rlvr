"""
Paper figures for Section 2.
Fig 1 (iconic): the price frontier -- pass@1 gain vs first-crossing k*, as the
                per-prompt floor gamma sweeps 0->1, for one representative
                inverting instance. Marks GRPO (gamma=0) and gamma=0.9,0.99,1.
Fig 2: pass@k curves base/RL, regime A (crossover) vs regime B (no crossover),
       with the Karlin bound S-(G) annotated.
Writes figures/fig1_price_frontier.png, figures/fig2_regimes.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import rng, tilt, p_of, delta_k, passk_curve, count_sign_changes, cdf_diff_sign_changes
from c6_correction import inverting_instance, relaxed_update
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
KS = np.unique(np.round(np.logspace(0, 5, 600)).astype(int)); KS = KS[KS >= 1]
KMAX1 = int(KS[-1]) + 1


def first_cross(p0, pt):
    d = delta_k(p0, pt, KS)
    tol = 1e-4 * max(np.max(np.abs(d)), 1e-12)
    s = np.sign(np.where(np.abs(d) < tol, 0.0, d))
    last = 0.0
    for i, si in enumerate(s):
        if si != 0.0:
            if last != 0.0 and si != last:
                return int(KS[i])
            last = si
    return KMAX1  # no crossing within range


# ---------------- Fig 1: price frontier ----------------
def fig1(seed=20261028):
    gen = rng(seed)
    Q, tau = inverting_instance(gen, n_tiers=3, tau=9.0)
    M = Q.shape[1]; w0 = np.full(M, 1.0 / M); f = Q.mean(axis=0); p0 = p_of(w0, Q)
    pk0 = passk_curve(p0, KS)
    gammas = np.linspace(0, 1, 41)
    gains, losses = [], []
    for g in gammas:
        w = relaxed_update(Q, w0, f, g)
        pt = p_of(w, Q)
        gains.append(float(passk_curve(pt, [1])[0] - passk_curve(p0, [1])[0]))
        deficit = pk0 - passk_curve(pt, KS)           # base - RL across k
        losses.append(float(max(deficit.max(), 0.0)))  # worst pass@k loss = the price
    gains = np.array(gains); losses = np.array(losses)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(gains, losses, "-", color="#1d3557", lw=2, zorder=1)
    ax.scatter(gains, losses, s=20, color="#1d3557", zorder=2)
    ax.fill_between(gains, losses, 0, color="#1d3557", alpha=0.06)
    marks = [(0.0, "GRPO (raw)", (6, -6)), (0.9, r"$\gamma$=0.9", (8, 6)),
             (0.99, r"$\gamma$=0.99", (10, 18)), (1.0, "no free lunch\nboundary ($\\gamma$=1)", (12, -2))]
    for gv, lab, off in marks:
        i = int(np.argmin(np.abs(gammas - gv)))
        ax.annotate(lab, (gains[i], losses[i]), textcoords="offset points",
                    xytext=off, fontsize=8.5, color="#c1121f")
        ax.scatter([gains[i]], [losses[i]], s=55, facecolors="none", edgecolors="#c1121f", zorder=3)
    ax.set_xlabel("pass@1 gain over base")
    ax.set_ylabel("the price: worst pass@k loss  $\\max_k(\\mathrm{base}-\\mathrm{RL})$")
    ax.set_title("No free lunch: every pass@1 gain has a pass@k price", fontsize=10.5)
    ax.set_xlim(left=-0.003); ax.set_ylim(bottom=-0.003)
    ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_price_frontier.png"), dpi=150)
    plt.close(fig)
    return float(gains.max()), float(losses.max())


# ---------------- Fig 2: regime A vs B ----------------
def regimeA(gen, n_strat=10, pps=40, floor=0.02):
    """Fine coverage with a genuinely hard tail: the hardest blocks are solved
    only by low-success specialists, driven to the floor by the tilt."""
    N = n_strat * pps
    Q = floor + 0.05 * gen.random((N, n_strat))
    # hardest blocks have very low success; easiest high
    levels = np.linspace(0.05, 0.9, n_strat)
    for j in range(n_strat):
        Q[j*pps:(j+1)*pps, j] = np.clip(levels[j] * (0.7 + 0.5*gen.random(pps)), floor, 0.99)
    return np.clip(Q, floor, 0.99)


def regimeB(gen, n_strat=10, pps=40, floor=0.02):
    """Same, but every prompt has broad backup coverage, so the hard tail keeps
    a success floor and no aggregate crossover forms."""
    Q = regimeA(gen, n_strat, pps, floor)
    for j in range(n_strat):
        Q[:, j] = np.clip(Q[:, j] + 0.28 * gen.random(), floor, 0.99)
    return Q


def _instance(Qf, gen, want_cross, tries=200, tau=12.0):
    for _ in range(tries):
        Q = Qf(gen)
        M = Q.shape[1]; w0 = np.full(M, 1.0/M); f = Q.mean(axis=0); p0 = p_of(w0, Q)
        pt = p_of(tilt(w0, f, tau), Q)
        d = delta_k(p0, pt, KS); nc = count_sign_changes(d, tol=1e-9)
        dip = -min(d.min(), 0.0)
        gain = d[0]
        if want_cross and nc >= 1 and dip > 0.03 and gain > 0.03:
            return Q, p0, pt
        if (not want_cross) and nc == 0 and gain > 0.03:
            return Q, p0, pt
    return Q, p0, pt   # fallback


def fig2(seed=7):
    gen = rng(seed)
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.2))
    specs = [("A", regimeA, True, "Regime A (fine coverage): crossover"),
             ("B", regimeB, False, "Regime B (backup coverage): no crossover")]
    for col, (name, Qf, want, title) in enumerate(specs):
        Q, p0, pt = _instance(Qf, gen, want)
        sg, _, _ = cdf_diff_sign_changes(p0, pt)
        d = delta_k(p0, pt, KS); nc = count_sign_changes(d, tol=1e-9)
        ax[0, col].semilogx(KS, passk_curve(p0, KS), color="#444", lw=2, label="base")
        ax[0, col].semilogx(KS, passk_curve(pt, KS), color="#c1121f", lw=2, label="RL (tilt)")
        ax[0, col].set_title(title, fontsize=10.5); ax[0, col].set_ylabel("pass@k")
        ax[0, col].legend(frameon=False, fontsize=9, loc="lower right")
        ax[1, col].semilogx(KS, d, color="#c1121f", lw=2)
        ax[1, col].axhline(0, color="#888", lw=.8)
        ax[1, col].set_xlabel("k"); ax[1, col].set_ylabel(r"$\Delta_k$")
        ax[1, col].annotate(r"$S^-(G)$ = {};  crossings = {}".format(sg, nc),
                            xy=(0.04, 0.08), xycoords="axes fraction", fontsize=9, color="#1d3557")
        for r in (0, 1):
            ax[r, col].grid(alpha=.22, which="both")
    fig.suptitle("Per-prompt loss is inevitable; the crossover needs the loss to reach the hard tail",
                 fontsize=11.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(FIG, "fig2_regimes.png"), dpi=150)
    plt.close(fig)


def fig3(seed=3):
    """Tilt law: log(b-R) vs H is linear with slope 1/kappa, kappa in {0.5,1,2}."""
    gen = rng(seed)
    M = 200000
    w0 = np.full(M, 1.0 / M); logM = np.log(M)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    colors = {0.5: "#c1121f", 1.0: "#1d3557", 2.0: "#2a9d8f"}
    for kappa in (0.5, 1.0, 2.0):
        u = gen.random(M); g = u ** (1.0 / kappa); f = -g
        taus = np.geomspace(1.0, 3e4, 80)
        Hs = np.array([entropy_of(tilt(w0, f, t)) for t in taus])
        gaps = np.array([-(tilt(w0, f, t) @ f) for t in taus])
        mask = (Hs > 0.2 * logM) & (Hs < 0.8 * logM) & (gaps > 0)
        slope, intercept = np.polyfit(Hs[mask], np.log(gaps[mask]), 1)
        ax.scatter(Hs[mask], np.log(gaps[mask]), s=14, color=colors[kappa],
                   label=r"$\kappa$={}: slope {:.3f} ($1/\kappa$={:.2f})".format(
                       kappa, slope, 1.0 / kappa))
        xx = np.array([Hs[mask].min(), Hs[mask].max()])
        ax.plot(xx, slope * xx + intercept, color=colors[kappa], lw=1.3)
    ax.set_xlabel("entropy $H$ (nats)")
    ax.set_ylabel(r"$\log(b-R)$")
    ax.set_title(r"Tilt law: $\log(b-R)=H/\kappa+\mathrm{const}$ identifies $\kappa$", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_tiltlaw.png"), dpi=150)
    plt.close(fig)


def entropy_of(w):
    w = np.asarray(w, float); m = w > 0
    return float(-(w[m] * np.log(w[m])).sum())


if __name__ == "__main__":
    g, k = fig1()
    fig2()
    fig3()
    print("fig1 max pass@1 gain:", round(g, 4), " price:", round(k, 4))
    print("figures written: fig1_price_frontier, fig2_regimes, fig3_tiltlaw")
