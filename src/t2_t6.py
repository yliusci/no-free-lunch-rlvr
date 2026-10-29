"""
T2 / R4 -- necessity of strategy sharing for a pass@k inversion.
T6        -- group size G shifts the crossing budget k*.

T2 : with per-prompt INDEPENDENT (tabular) softmax policies and exact gradient
     flow, the success prob p(x) is nondecreasing for every prompt, so
     Δ_k = E[(1-p0)^k - (1-pτ)^k] >= 0 for all k: NO inversion. Inversion
     therefore requires shared parameters across prompts (R4). We also show the
     finite-G stochastic caveat (2602.06717): a stochastic step can transiently
     lower p(x).

T6 : on a shared-strategy instance that inverts, the first crossing k* grows
     with the GRPO group size G (larger G rescues more low-p prompts).

Writes results/t2.json and results/t6.json.
"""
import json, os
import numpy as np
from common import rng, tilt, p_of, delta_k, count_sign_changes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
KS = np.unique(np.round(np.logspace(0, 5, 400)).astype(int)); KS = KS[KS >= 1]


# ---------------------------------------------------------------- T2 exact flow
def tabular_exact_flow(seed=11, N=500, modes=12, n_correct=3, steps=400, eta=0.1):
    """
    Each prompt has its own softmax over `modes` modes; the first n_correct are
    correct (reward 1). Exact natural-PG gradient flow. Track p(x)=correct mass.
    Returns min increment over prompts and steps (must be >= 0).
    """
    gen = rng(seed)
    theta = gen.standard_normal((N, modes)) * 0.5
    correct = np.zeros(modes); correct[:n_correct] = 1.0
    def probs(th):
        e = np.exp(th - th.max(axis=1, keepdims=True)); return e / e.sum(1, keepdims=True)
    p_hist = []
    w = probs(theta)
    p0 = w @ correct
    worst_decrease = 0.0
    prev = p0.copy()
    for _ in range(steps):
        w = probs(theta)
        p = w @ correct
        # natural PG on binary reward: advantage A_i = reward_i - <reward> = correct_i - p
        # dtheta_i ∝ w_i * A_i ; natural grad => theta += eta * A (per mode)
        A = correct[None, :] - p[:, None]
        theta = theta + eta * A
        worst_decrease = min(worst_decrease, float((p - prev).min()))
        prev = p
    w = probs(theta); pT = w @ correct
    return {
        "N": N, "modes": modes, "n_correct": n_correct, "steps": steps,
        "min_p_increment_any_prompt_any_step": float(worst_decrease),  # >= 0 expected
        "min_pT_minus_p0_over_prompts": float((pT - p0).min()),        # >= 0 expected
        "delta_k_min_over_k": float(delta_k(p0, pT, KS).min()),        # >= 0 => no inversion
        "n_passk_crossings": int(count_sign_changes(delta_k(p0, pT, KS), tol=1e-12)),
    }


# ---------------------------------------------------------------- T2 stochastic caveat
def tabular_stochastic_caveat(seed=5, N=300, modes=12, n_correct=3, G=4,
                              steps=200, eta=0.2):
    """Finite-G stochastic GRPO on independent per-prompt policies can transiently
    lower p(x). Report fraction of prompts whose p ever dips below its start."""
    gen = rng(seed)
    theta = gen.standard_normal((N, modes)) * 0.5
    correct = np.zeros(modes); correct[:n_correct] = 1.0
    def probs(th):
        e = np.exp(th - th.max(axis=1, keepdims=True)); return e / e.sum(1, keepdims=True)
    p0 = probs(theta) @ correct
    ever_dipped = np.zeros(N, bool)
    for _ in range(steps):
        w = probs(theta)
        newtheta = theta.copy()
        for x in range(N):
            picks = gen.choice(modes, size=G, p=w[x])
            rew = correct[picks]
            sd = rew.std()
            if sd < 1e-12:
                continue
            adv = (rew - rew.mean()) / sd
            g = np.zeros(modes)
            for i in range(G):
                g[picks[i]] += adv[i]
            newtheta[x] += eta * g / G
        theta = newtheta
        p = probs(theta) @ correct
        ever_dipped |= (p < p0 - 1e-9)
    return {"G": G, "N": N, "frac_prompts_ever_dipped": float(ever_dipped.mean())}


# ---------------------------------------------------------------- T6 group size
def make_inverting_instance(gen, n_tiers=3, tau=8.0, ppt=100, spt=30, floor=0.02):
    N = n_tiers * ppt; M = n_tiers * spt
    Q = np.full((N, M), floor)
    for t in range(n_tiers):
        pr = slice(t * ppt, (t + 1) * ppt); st = slice(t * spt, (t + 1) * spt)
        succ = 0.9 - 0.25 * t
        Q[pr, st] = np.clip(floor + succ * gen.uniform(0.4, 1.0, size=(ppt, spt)), floor, 0.99)
    for t in range(n_tiers):
        st = slice(t * spt, (t + 1) * spt)
        Q[:, st] = np.clip(Q[:, st] + 0.15 * (1 - t / (n_tiers - 1)), floor, 0.99)
    return Q


def grpo_tilt_passk(Q, G, tau_target, gen, steps=120, eta=0.3):
    """Run stochastic GRPO on shared logits; return p0, pT after reaching ~tau."""
    M = Q.shape[1]; N = Q.shape[0]
    theta = np.zeros(M)
    w0 = np.full(M, 1.0 / M)
    p0 = p_of(w0, Q)
    for _ in range(steps):
        w = np.exp(theta - theta.max()); w /= w.sum()
        grad = np.zeros(M)
        for x in range(N):
            picks = gen.choice(M, size=G, p=w)
            rew = (gen.random(G) < Q[x, picks]).astype(float)
            sd = rew.std()
            if sd < 1e-12:
                continue
            adv = (rew - rew.mean()) / sd
            for i in range(G):
                grad[picks[i]] += adv[i]
        theta += eta * grad / N
    w = np.exp(theta - theta.max()); w /= w.sum()
    return p0, p_of(w, Q)


def run_t6_stochastic(seed=3, Gs=(2, 4, 8, 16, 64), reps=6):
    """Naive: fixed steps per G. CONFOUNDED (larger G also concentrates faster)."""
    gen = rng(seed)
    Q = make_inverting_instance(gen)
    out = {}
    for G in Gs:
        kstars = []
        for r in range(reps):
            p0, pT = grpo_tilt_passk(Q, G, 8.0, rng(1000 + 100 * G + r))
            d = delta_k(p0, pT, KS)
            sc = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
            if sc.size >= 1:
                kstars.append(int(KS[sc[0]]))
        out[str(G)] = {"n_crossed": len(kstars),
                       "median_kstar": float(np.median(kstars)) if kstars else None}
    return out


def run_t6_matched(seed=3, Gs=(2, 4, 8, 16, 64), tau=8.0, reps=8):
    """
    Clean test of the T6 mechanism at MATCHED concentration (same τ): the
    idealized tilt is applied, but a prompt is 'frozen' at its base success with
    probability (1-p0)^G (its GRPO group was degenerate, no learning signal).
    Larger G lowers the freezing threshold ~1/G, so fewer hard prompts stay
    frozen. We measure the first crossing k* vs G.
    """
    gen = rng(seed)
    Q = make_inverting_instance(gen)
    M = Q.shape[1]; w0 = np.full(M, 1.0 / M)
    f = Q.mean(axis=0)
    p0 = p_of(w0, Q)
    ptilt = p_of(tilt(w0, f, tau), Q)
    out = {}
    for G in Gs:
        kstars = []
        for r in range(reps):
            g = rng(2000 + 7 * G + r)
            frozen = g.random(p0.size) < (1.0 - p0) ** G   # degenerate => frozen
            pT = np.where(frozen, p0, ptilt)
            d = delta_k(p0, pT, KS)
            sc = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
            if sc.size >= 1:
                kstars.append(int(KS[sc[0]]))
        out[str(G)] = {"n_crossed": len(kstars), "frozen_frac_approx": float(np.mean((1.0 - p0) ** G)),
                       "median_kstar": float(np.median(kstars)) if kstars else None}
    return out


def main():
    t2 = {"exact_flow": tabular_exact_flow(),
          "stochastic_caveat": tabular_stochastic_caveat()}
    with open(os.path.join(RES, "t2.json"), "w") as fh:
        json.dump(t2, fh, indent=2)
    t6 = {"stochastic_confounded": run_t6_stochastic(),
          "matched_concentration": run_t6_matched()}
    with open(os.path.join(RES, "t6.json"), "w") as fh:
        json.dump(t6, fh, indent=2)
    print(json.dumps({"t2": t2, "t6": t6}, indent=2))


if __name__ == "__main__":
    main()
