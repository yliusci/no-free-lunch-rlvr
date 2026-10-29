"""
common.py -- Shared model, tilt dynamics, pass@k, sign-variation utilities.

Paper 2: "Pass@k Can Cross Twice" (working title). Solvable model of RLVR as
exponential tilting of a policy over strategies SHARED across prompts.

All randomness goes through a seeded numpy Generator. Every script fixes its
own seed so each results/*.json and figure is regenerable.

Model
-----
M strategies, N prompts. Q[x, j] in [0,1] = P(strategy j solves prompt x).
Policy w on the simplex over M strategies, SHARED across prompts.
Per-prompt success p(x) = (Q @ w)[x].
Single sample solves prompt x w.p. p(x); pass@k(x) = 1 - (1-p(x))^k.
PassK(k) = E_x[1 - (1-p(x))^k].

Tilt (idealized natural PG / mirror descent):
    w_tau ∝ w0 * exp(tau * f),  f_j = E_x[Q[x,j]]  (marginal fitness).
Sharing across prompts is what lets RL LOWER p(x) for some prompts (those
solved mainly by low-fitness strategies), which is required for a pass@k
inversion (Thm T2 / R4).
"""
import numpy as np


# ----------------------------------------------------------------------
# RNG
# ----------------------------------------------------------------------
def rng(seed):
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------
# Core model
# ----------------------------------------------------------------------
def p_of(w, Q):
    """Per-prompt success probabilities p(x) = sum_j w_j Q[x,j]."""
    return Q @ w


def tilt(w0, f, tau):
    """Exponentially tilted weights w ∝ w0 * exp(tau f), normalized. Stable."""
    z = tau * f
    z = z - z.max()
    w = w0 * np.exp(z)
    s = w.sum()
    return w / s


def entropy(w, eps=0.0):
    """Shannon entropy in nats of a probability vector."""
    w = np.asarray(w, dtype=float)
    mask = w > 0
    return float(-(w[mask] * np.log(w[mask])).sum())


def reward(w, f):
    """Mean fitness R = <f>_w = sum_j w_j f_j (the model's accuracy proxy)."""
    return float(w @ f)


# ----------------------------------------------------------------------
# pass@k
# ----------------------------------------------------------------------
def passk_curve(p, ks):
    """
    PassK(k) = mean_x [1 - (1-p(x))^k] for each k in ks.
    p : (N,) array of per-prompt success probs.  ks : (K,) ints.
    Returns (K,) array.
    """
    p = np.asarray(p, dtype=float)
    one_minus = np.clip(1.0 - p, 0.0, 1.0)
    # log for stability over large k
    with np.errstate(divide="ignore"):
        logq = np.log(one_minus)                     # (N,)
    ks = np.asarray(ks, dtype=float)                 # (K,)
    # (N,K): (1-p)^k = exp(k * log(1-p)); log(1-p)=-inf when p=1 -> term 0
    M = np.exp(np.outer(logq, ks))                   # (N,K), 0 where p==1
    M[np.isneginf(logq)] = 0.0
    return 1.0 - M.mean(axis=0)


def delta_k(p0, ptau, ks):
    """Δ_k = PassK_tau(k) - PassK_0(k) over ks."""
    return passk_curve(ptau, ks) - passk_curve(p0, ks)


# ----------------------------------------------------------------------
# Sign variation
# ----------------------------------------------------------------------
def count_sign_changes(seq, tol=1e-12):
    """
    Number of sign changes of a real sequence, ignoring entries with
    |value| <= tol (treated as zero / no contribution). Robust to flat zero
    runs: consecutive nonzero values of opposite sign count one change.
    """
    s = np.asarray(seq, dtype=float)
    signs = np.sign(np.where(np.abs(s) <= tol, 0.0, s))
    signs = signs[signs != 0.0]
    if signs.size < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def cdf_diff_sign_changes(p0, ptau, grid=None, tol=1e-9):
    """
    Sign changes of G(p) = F_tau(p) - F_0(p) across p, where F are empirical
    CDFs of the per-prompt success distributions. This is the Karlin bound
    S-(ν) that upper-bounds the number of pass@k crossings.
    """
    p0 = np.asarray(p0, float)
    ptau = np.asarray(ptau, float)
    if grid is None:
        grid = np.linspace(0.0, 1.0, 2001)
    F0 = np.searchsorted(np.sort(p0), grid, side="right") / p0.size
    Ft = np.searchsorted(np.sort(ptau), grid, side="right") / ptau.size
    G = Ft - F0
    return count_sign_changes(G, tol=tol), G, grid


# ----------------------------------------------------------------------
# GRPO stochastic step (for T1 deviation bound and T6 group size)
# ----------------------------------------------------------------------
def grpo_logit_step(theta, f_strategy, Q, G, eta, gen):
    """
    One stochastic GRPO-style update on mode logits theta (shared policy).
    For a batch of prompts, sample G rollouts, each rollout picks a strategy
    ~ softmax(theta), reward = Bernoulli(Q[x, strategy]). Group-normalize the
    advantage (subtract mean, divide by std with the degeneracy guard), map to
    a logit increment. Population-average over prompts.

    Returns updated theta. This is the stochastic counterpart to the
    mirror-descent tilt; used to measure deviation from the exact tilt (T1)
    and the group-size effect on k* (T6).
    """
    M = theta.size
    w = np.exp(theta - theta.max())
    w = w / w.sum()
    N = Q.shape[0]
    grad = np.zeros(M)
    for x in range(N):
        picks = gen.choice(M, size=G, p=w)
        rew = (gen.random(G) < Q[x, picks]).astype(float)
        mu = rew.mean()
        sd = rew.std()
        if sd < 1e-12:
            continue  # degenerate group -> zero advantage
        adv = (rew - mu) / sd
        # accumulate advantage onto picked modes
        for g in range(G):
            grad[picks[g]] += adv[g]
    grad /= N
    return theta + eta * grad


__all__ = [
    "rng", "p_of", "tilt", "entropy", "reward",
    "passk_curve", "delta_k", "count_sign_changes",
    "cdf_diff_sign_changes", "grpo_logit_step",
]
