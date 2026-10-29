"""
diagnostic.py -- the reusable artifact (FORM): read-off tool for pass@k curves.

Given a base model's and an RL model's pass@k curves (or their per-prompt success
distributions), report:
  - the number of crossings and their locations k*,
  - the inferred REGIME:
      0 crossings      -> no inversion (RL dominates, or is dominated, at all k)
      1 crossing       -> difficulty-order-PRESERVING sharpening
                          (the Yue 2504.13837 / Yang-Chen 2609.22547 regime)
      >=2 crossings    -> order NOT preserved: sharing-induced non-monotone
                          reweighting across prompts (our regime),
  - when per-prompt success is available, the Karlin bound S-(G) and whether the
    observed crossing count saturates it.

This turns the theorem (R2/R3) into something a practitioner runs on their own
base-vs-RL eval to read off what RL did to their prompt-difficulty structure.
Accepts numeric curves from CSV/JSON so published or released data drops straight in.
"""
import json
import numpy as np
from common import passk_curve, count_sign_changes, cdf_diff_sign_changes


def _crossings(ks, delta, rel_tol=1e-3):
    """Sign changes of delta over k, ignoring |delta| below rel_tol*max|delta|
    (removes the delta≈0 plateau at large k where both curves saturate)."""
    delta = np.asarray(delta, float)
    tol = rel_tol * np.max(np.abs(delta)) if np.max(np.abs(delta)) > 0 else 1e-12
    s = np.sign(np.where(np.abs(delta) < tol, 0.0, delta))
    idx, last = [], 0.0
    for i, si in enumerate(s):
        if si != 0.0:
            if last != 0.0 and si != last:
                idx.append(i)
            last = si
    return np.asarray(ks)[idx]


def report_from_curves(ks, passk_base, passk_rl, rel_tol=1e-3):
    """Core diagnostic from two pass@k curves sampled at the same ks."""
    ks = np.asarray(ks, float)
    d = np.asarray(passk_rl, float) - np.asarray(passk_base, float)
    kc = _crossings(ks, d, rel_tol)
    n = int(kc.size)
    small_k_sign = np.sign(d[np.argmin(ks)])
    if n == 0:
        regime = ("RL dominates at all k" if small_k_sign > 0
                  else "RL dominated at all k" if small_k_sign < 0
                  else "curves coincide")
    elif n == 1:
        regime = ("order-preserving sharpening (Yue/Yang-Chen regime): "
                  "RL wins small-k, loses large-k" if small_k_sign > 0
                  else "single crossing, RL loses small-k wins large-k")
    else:
        regime = ("difficulty order NOT preserved: sharing-induced non-monotone "
                  "reweighting (>=2 crossings)")
    return {"n_crossings": n, "crossing_ks": [float(x) for x in kc],
            "advantage_sign_at_min_k": float(small_k_sign), "regime": regime}


def report_from_success(p0, ptau, ks=None, rel_tol=1e-3):
    """Diagnostic + Karlin bound when per-prompt success probs are available."""
    if ks is None:
        ks = np.unique(np.round(np.logspace(0, 5, 500)).astype(int)); ks = ks[ks >= 1]
    pk0, pkt = passk_curve(p0, ks), passk_curve(ptau, ks)
    rep = report_from_curves(ks, pk0, pkt, rel_tol)
    sg, _, _ = cdf_diff_sign_changes(p0, ptau)
    rep["karlin_bound_Sminus_G"] = int(sg)
    rep["bound_respected"] = bool(rep["n_crossings"] <= sg)
    rep["bound_saturated"] = bool(rep["n_crossings"] == sg)
    return rep


def load_curves(path):
    """Load pass@k curves from JSON {'k':[...], 'base':[...], 'rl':[...]} or CSV
    with columns k,base,rl (header required). Returns (ks, base, rl)."""
    if path.endswith(".json"):
        with open(path) as fh:
            d = json.load(fh)
        return np.array(d["k"]), np.array(d["base"]), np.array(d["rl"])
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return arr["k"], arr["base"], arr["rl"]


if __name__ == "__main__":
    # self-test on the two synthetic figure instances
    ks = np.unique(np.round(np.logspace(0, 5, 600)).astype(int)); ks = ks[ks >= 1]
    pA0 = np.concatenate([np.full(700, 0.40), np.full(300, 0.02)])
    pAt = np.concatenate([np.full(700, 0.60), np.full(300, 0.004)])
    pB0 = np.concatenate([np.full(400, 0.50), np.full(400, 0.05), np.full(400, 0.002)])
    pBt = np.concatenate([np.full(400, 0.70), np.full(400, 0.012), np.full(400, 0.006)])
    print("single-crossing instance:", json.dumps(report_from_success(pA0, pAt), indent=1))
    print("double-crossing instance:", json.dumps(report_from_success(pB0, pBt), indent=1))
