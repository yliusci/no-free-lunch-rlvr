"""
empirical.py -- Section 4 anchor on PUBLISHED pass@k curves (no model run).

Data are pass@k values transcribed from Zhou (2607.20543), "When RLVR Shrinks
the Reasoning Boundary", Table 2 (Qwen2.5-7B, GRPO on Omni-MATH-Train). They are
INDICATIVE and MUST be re-checked against the source table before submission.

We pass them through the diagnostic (diagnostic.report_from_curves) and show:
  - GRPO vs base on Omni-MATH: exactly one pass@k crossing (the price is paid),
    between k=64 and k=256.
  - GRPO+PBA (the paper's diversity intervention) vs base: no crossing within the
    measured budget -- it moves ALONG the price frontier, delaying the crossover.
  - MATH500: no crossing within k<=256 (the price lies beyond the measured budget).

Writes figures/fig4_published.png and results/empirical.json.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diagnostic import report_from_curves

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
FIG = os.path.join(ROOT, "figures"); RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

# --- transcribed from 2607.20543 Table 2 (Qwen2.5-7B). TO VERIFY AT SOURCE. ---
KS = np.array([1, 4, 16, 64, 256], float)
DATA = {
    "Omni-MATH-Test": {
        "Base":      np.array([10.2, 22.4, 39.2, 57.8, 69.1]) / 100,
        "GRPO":      np.array([25.1, 36.8, 49.6, 60.8, 68.3]) / 100,
        "GRPO+PBA":  np.array([29.0, 42.9, 58.0, 67.9, 73.0]) / 100,
    },
    "MATH500": {
        "Base":      np.array([34.5, 61.0, 82.0, 93.0, 96.2]) / 100,
        "GRPO":      np.array([74.1, 88.4, 95.1, 96.8, 97.2]) / 100,
        "GRPO+PBA":  np.array([75.6, 89.2, 95.7, 97.4, 97.8]) / 100,
    },
}


def main():
    out = {"provenance": "2607.20543 Table 2, transcribed, INDICATIVE, verify at source",
           "ks": KS.tolist(), "results": {}}
    for bench, d in DATA.items():
        out["results"][bench] = {}
        for rl in ["GRPO", "GRPO+PBA"]:
            rep = report_from_curves(KS, d["Base"], d[rl], rel_tol=1e-6)
            # crossing requires a sign change in the measured window
            delta = d[rl] - d["Base"]
            out["results"][bench][rl] = {
                "delta_pass_at_k": [round(float(x), 4) for x in delta],
                "n_crossings_in_window": rep["n_crossings"],
                "crossing_ks": rep["crossing_ks"],
                "price_visible_in_window": rep["n_crossings"] >= 1,
            }

    with open(os.path.join(RES, "empirical.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # figure: Omni-MATH (price visible) and MATH500 (price beyond window)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    cols = {"Base": "#444", "GRPO": "#c1121f", "GRPO+PBA": "#2a9d8f"}
    for j, bench in enumerate(["Omni-MATH-Test", "MATH500"]):
        for name, y in DATA[bench].items():
            ax[j].semilogx(KS, y, "-o", color=cols[name], lw=2, ms=4, label=name)
        ax[j].set_title(bench, fontsize=10.5)
        ax[j].set_xlabel("k"); ax[j].set_ylabel("pass@k")
        ax[j].legend(frameon=False, fontsize=8.5, loc="lower right")
        ax[j].grid(alpha=.25, which="both")
        # annotate crossing if any
        rg = out["results"][bench]["GRPO"]
        note = ("GRPO crosses base at k in (64,256)" if rg["price_visible_in_window"]
                else "no crossing for k<=256\n(price beyond budget)")
        ax[j].annotate(note, xy=(0.03, 0.08), xycoords="axes fraction", fontsize=8, color="#1d3557")
    fig.suptitle("Published GRPO pass@k (Qwen2.5-7B, 2607.20543, transcribed/indicative): "
                 "the price is real and intervention delays it", fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_published.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(out["results"], indent=2))


if __name__ == "__main__":
    main()
