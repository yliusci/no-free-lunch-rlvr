"""
make_figure.py -- main paper figure: the pass@k crossover story.

Panel A: a SINGLE crossing (the canonical RLVR inversion; order-preserving case,
         = Yang & Chen 2609.22547). RL wins pass@1, loses large-k.
Panel B: a DOUBLE crossing (S-(G)=2), crossings decades apart, clearly visible.
         Multiple crossings are possible and bounded by S-(G) (R2).

Illustrative prompt populations are constructed with a wide difficulty spread so
pass@k separates across many decades of k. Crossings are marked. 1-pass@k is
shown on log-y so the large-k tails are visible.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import passk_curve, delta_k, count_sign_changes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
KS = np.unique(np.round(np.logspace(0, 5, 600)).astype(int)); KS = KS[KS >= 1]


def crossings(p0, pt):
    """Crossings where Δ_k changes sign with meaningful magnitude (ignore the
    Δ≈0 plateau at large k where both curves have saturated to 1)."""
    d = delta_k(p0, pt, KS)
    tol = 1e-4
    s = np.sign(np.where(np.abs(d) < tol, 0.0, d))
    idx = []
    last = 0.0
    for i, si in enumerate(s):
        if si != 0.0:
            if last != 0.0 and si != last:
                idx.append(i)
            last = si
    return KS[idx], d


# ---- Panel A: single crossing, order-preserving (wide difficulty spread) ----
# base: easy prompts p=0.4, hard prompts p=0.02. RL sharpens difficulty order:
# easy -> 0.6, hard -> 0.004 (monotone map, one identity crossing).
pA0 = np.concatenate([np.full(700, 0.40), np.full(300, 0.02)])
pAt = np.concatenate([np.full(700, 0.60), np.full(300, 0.004)])
kA, dA = crossings(pA0, pAt)

# ---- Panel B: double crossing, three difficulty bands ----
# base: three bands at p = 0.5, 0.05, 0.002 (equal mass).
# RL: help the top and bottom bands, hurt the middle -> G changes sign twice.
pB0 = np.concatenate([np.full(400, 0.50), np.full(400, 0.05), np.full(400, 0.002)])
pBt = np.concatenate([np.full(400, 0.70), np.full(400, 0.012), np.full(400, 0.006)])
kB, dB = crossings(pB0, pBt)

fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))

# --- A top: pass@k
ax[0, 0].semilogx(KS, passk_curve(pA0, KS), color="#444", lw=2, label="base")
ax[0, 0].semilogx(KS, passk_curve(pAt, KS), color="#c1121f", lw=2, label="RL (tilt)")
for kc in kA:
    ax[0, 0].axvline(kc, color="#1d3557", ls=":", lw=1)
ax[0, 0].set_title("A. Single crossing (order-preserving = Yang & Chen)", fontsize=10.5)
ax[0, 0].set_ylabel("pass@k"); ax[0, 0].legend(frameon=False, fontsize=9, loc="lower right")

# --- A bottom: Δ_k
ax[1, 0].semilogx(KS, dA, color="#c1121f", lw=2)
ax[1, 0].axhline(0, color="#888", lw=.8)
for kc in kA:
    ax[1, 0].axvline(kc, color="#1d3557", ls=":", lw=1)
ax[1, 0].annotate("k* = {}".format(int(kA[0])) if kA.size else "no crossing",
                  xy=(kA[0] if kA.size else 10, 0), xytext=(5, 10),
                  textcoords="offset points", fontsize=9, color="#1d3557")
ax[1, 0].set_xlabel("k"); ax[1, 0].set_ylabel(r"$\Delta_k$")

# --- B top: pass@k
ax[0, 1].semilogx(KS, passk_curve(pB0, KS), color="#444", lw=2, label="base")
ax[0, 1].semilogx(KS, passk_curve(pBt, KS), color="#c1121f", lw=2, label="RL (tilt)")
for kc in kB:
    ax[0, 1].axvline(kc, color="#1d3557", ls=":", lw=1)
ax[0, 1].set_title("B. Double crossing (bounded by $S^-(G)=2$)", fontsize=10.5)
ax[0, 1].legend(frameon=False, fontsize=9, loc="lower right")

# --- B bottom: Δ_k
ax[1, 1].semilogx(KS, dB, color="#c1121f", lw=2)
ax[1, 1].axhline(0, color="#888", lw=.8)
for kc in kB:
    ax[1, 1].axvline(kc, color="#1d3557", ls=":", lw=1)
ax[1, 1].set_xlabel("k"); ax[1, 1].set_ylabel(r"$\Delta_k$")
ax[1, 1].annotate("two crossings:\nk = {}".format(", ".join(map(str, map(int, kB)))),
                  xy=(0.04, 0.08), xycoords="axes fraction", fontsize=9, color="#1d3557")

for a in ax.flat:
    a.grid(alpha=.22, which="both")
fig.suptitle("Reinforcement reweights shared strategies: pass@k can cross more than once",
             fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(FIG, "fig_main_passk.png"), dpi=150)
print("panel A crossings at k =", list(map(int, kA)))
print("panel B crossings at k =", list(map(int, kB)))
