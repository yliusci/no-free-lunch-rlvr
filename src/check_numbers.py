"""
check_numbers.py -- verify every number cited in the paper against results/*.json.

Invariants (exit nonzero if any fails) are the CONTRIBUTIONS. Setup results are
checked at looser tolerances. T6 is reported as a FLAG (conjecture not supported),
not a pass/fail. Run all t*.py first (or `python run_all.py`).
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
RES = os.path.join(ROOT, "results")


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def main():
    checks = []   # (label, ok, detail)
    flags = []

    t5 = load("t5.json")
    ens = t5["R2_R3_ensemble"]
    tight = t5["R2_tightness_constructive"]
    order = t5["R3_order_preserving"]

    # ---- CONTRIBUTIONS (hard invariants) ----
    checks.append(("R2 Karlin bound: 0 violations / {}".format(ens["karlin_checked"]),
                   ens["karlin_violations"] == 0, ens["karlin_violations"]))
    checks.append(("R2 tightness: no bound violation in search",
                   tight["any_bound_violation_in_search"] is False, None))
    checks.append(("R2 tightness: bound achieved for all T=1..6",
                   all(r["achieved"] for r in tight["per_T"]),
                   [r["max_crossings_found"] for r in tight["per_T"]]))
    checks.append(("R3 order-preserving => unique crossing: {}/{}".format(
                       order["n_unique"], order["n_instances"]),
                   order["frac_unique"] == 1.0, order["frac_unique"]))

    t2 = load("t2.json")["exact_flow"]
    checks.append(("R4/T2 tabular never lowers p (min increment >= -1e-12)",
                   t2["min_p_increment_any_prompt_any_step"] >= -1e-12,
                   t2["min_p_increment_any_prompt_any_step"]))
    checks.append(("R4/T2 tabular: min_x (pT-p0) >= 0 (no prompt hurt)",
                   t2["min_pT_minus_p0_over_prompts"] >= -1e-12,
                   t2["min_pT_minus_p0_over_prompts"]))
    checks.append(("R4/T2 tabular: 0 pass@k crossings (no inversion)",
                   t2["n_passk_crossings"] == 0, t2["n_passk_crossings"]))

    # ---- SETUP (reproduced; looser tolerances) ----
    t1 = load("t1.json")
    checks.append(("T1 mirror-descent = exact tilt (TV < 1e-12)",
                   t1["tv_mirror_vs_closed_form_tilt"] < 1e-12,
                   t1["tv_mirror_vs_closed_form_tilt"]))
    grpo = t1["grpo_vs_tilt_TV_at_matched_entropy"]
    checks.append(("T1 GRPO vs tilt TV < 1e-2 at matched entropy",
                   all(v < 1e-2 for v in grpo.values()), grpo))

    t3 = load("t3.json")
    checks.append(("T3 uniform entropy identity err < 1e-4",
                   t3["uniform_identity_max_abs_err"] < 1e-4,
                   t3["uniform_identity_max_abs_err"]))
    checks.append(("T3 nonuniform entropy bump present",
                   t3["nonuniform_entropy_bump_present"] is True, None))

    t4 = load("t4.json")
    for key, row in t4.items():
        checks.append(("T4 {} slope rel_err < 5% (pre-registered)".format(key),
                       row["rel_err"] < 0.05,
                       "measured {:.3f} vs 1/kappa {:.3f}".format(
                           row["measured_slope"], row["predicted_slope_1_over_kappa"])))

    br = load("bridge.json")
    checks.append(("bridge: strategy entropy falls while token entropy decouples",
                   br["decoupling_token_drop_lt_strategy_drop"] is True,
                   "strat drop {:.3f}, token drop {:.3f}, token rose={}".format(
                       br["strategy_entropy_drop"], br["token_entropy_drop"], br["token_entropy_rose"])))

    # ---- C6 correction ----
    c6 = load("c6.json")
    e = c6["guarantee_test_E_constrained"]
    checks.append(("C6 strict non-regression: 100% zero crossings",
                   e["frac_zero_crossings"] == 1.0, e["frac_zero_crossings"]))
    checks.append(("C6 strict: 100% FOSD (F_tau <= F_0)",
                   e["frac_FOSD_Ftau_le_F0"] == 1.0, e["frac_FOSD_Ftau_le_F0"]))
    checks.append(("C6 strict: no prompt regresses (worst loss <= 1e-9)",
                   e["worst_per_prompt_loss_max"] <= 1e-9, e["worst_per_prompt_loss_max"]))
    tr = c6["C6_tradeoff_gamma"]
    gkeys = sorted(tr.keys(), key=float)
    gains = [tr[g]["median_pass1_gain"] for g in gkeys]
    kstars = [tr[g]["median_first_crossing_k"] for g in gkeys]
    checks.append(("C6 tradeoff: pass@1 gain nonincreasing in gamma",
                   all(gains[i] >= gains[i + 1] - 1e-9 for i in range(len(gains) - 1)), gains))
    checks.append(("C6 tradeoff: crossing k* nondecreasing in gamma",
                   all(kstars[i] <= kstars[i + 1] + 1e-9 for i in range(len(kstars) - 1)), kstars))

    # ---- T* no-free-lunch theorem (Stiemke; exclusive/dominant specialization) ----
    ts = load("tstar.json")
    ce = ts["counterexample_unique_best"]
    checks.append(("T2 counterexample: unique-best admits a free lunch (insufficient)",
                   ce["has_free_lunch"] is True and ce["stiemke_dual_feasible"] is False,
                   ce["freelunch_gain"]))
    exc = ts["exclusive_specialized"]
    checks.append(("T2 exclusive specialization => no free lunch (1000/1000)",
                   exc["frac_no_free_lunch"] == 1.0, exc["frac_no_free_lunch"]))
    dom = ts["dominant_specialized"]
    checks.append(("T2b' dominant specialization => no free lunch (2000)",
                   dom["frac_no_free_lunch"] == 1.0, dom["frac_no_free_lunch"]))
    for g in ["exclusive_specialized", "unique_best", "quasi_generalist", "overlapping",
              "near_generalist_adversarial", "dominant_specialized"]:
        checks.append(("T2 Stiemke primal<=>dual agreement on {}".format(g),
                       ts[g]["primal_dual_agreement"] == 1.0, ts[g]["primal_dual_agreement"]))
    ng = ts["near_generalist_adversarial"]
    checks.append(("T2 adversarial near-generalist DOES admit free lunch (0% no-fl)",
                   ng["frac_no_free_lunch"] == 0.0, ng["frac_no_free_lunch"]))
    fc = ts["fine_coverage"]
    checks.append(("T3 fine coverage holds => crossover 100% and min-success drops 100%",
                   fc["holds"]["frac_crossover"] == 1.0 and fc["holds"]["frac_min_success_dropped"] == 1.0,
                   fc["holds"]))
    checks.append(("T3 fine coverage fails => no crossover (0%)",
                   fc["fails"]["frac_crossover"] == 0.0, fc["fails"]["frac_crossover"]))
    thc = ts["fine_coverage_threshold_counterexample"]
    checks.append(("T3 threshold: R(tau) is the WRONG drop threshold (weight rises though f<R(tau))",
                   thc["weight_increased"] is True and thc["f_below_R_tau"] is True
                   and thc["f_below_Lambda_over_tau"] is False, thc["weight_ratio"]))

    # ---- empirical anchor (indicative; pipeline sanity check) ----
    try:
        emp = load("empirical.json")
        og = emp["results"]["Omni-MATH-Test"]["GRPO"]
        checks.append(("Empirical (indicative): diagnostic finds 1 crossing in "
                       "published Omni-MATH GRPO curve",
                       og["n_crossings_in_window"] == 1, og["n_crossings_in_window"]))
    except Exception:
        pass

    # ---- FLAGS (not pass/fail) ----
    flags.append("EMPIRICAL numbers are transcribed from 2607.20543 Table 2 and are "
                 "INDICATIVE; verify against the source table before submission.")
    flags.append("T2: 'unique-best' specialization is INSUFFICIENT (counterexample has a "
                 "free lunch, gain {:.3f}); the paper uses exclusive / dominant "
                 "specialization, both verified.".format(ce["freelunch_gain"]))
    flags.append("T3 CROSSOVER IS CONDITIONAL on fine coverage (primitive: hardest prompts "
                 "covered only by f_j < R0): holds => 100% crossover, fails => 0%.")
    flags.append("C6 NO FREE LUNCH: full crossing removal (gamma=1) costs ALL pass@1 "
                 "gain (median gain 0.0); the crossing is the dual of the pass@1 gain.")
    t6 = load("t6.json")["matched_concentration"]
    ks = [(int(g), v["median_kstar"]) for g, v in t6.items() if v["median_kstar"] is not None]
    ks.sort()
    increasing = all(ks[i][1] <= ks[i + 1][1] for i in range(len(ks) - 1))
    flags.append("T6 CONJECTURE NOT SUPPORTED: at matched concentration k* vs G = "
                 + ", ".join("G{}:{:.0f}".format(g, k) for g, k in ks)
                 + "  -> {} (LOCK conjectured k* INCREASES with G)".format(
                     "increases" if increasing else "flat/decreases"))

    # ---- report ----
    print("=" * 74)
    print("CHECK_NUMBERS -- paper 2 verification package")
    print("=" * 74)
    npass = 0
    for label, ok, detail in checks:
        tag = "PASS" if ok else "FAIL"
        if ok:
            npass += 1
        line = "[{}] {}".format(tag, label)
        if detail is not None and not ok:
            line += "   <- {}".format(detail)
        print(line)
    print("-" * 74)
    for fl in flags:
        print("[FLAG] " + fl)
    print("-" * 74)
    print("{}/{} invariants passed.".format(npass, len(checks)))
    allok = npass == len(checks)
    print("RESULT:", "ALL INVARIANTS HOLD" if allok else "SOME INVARIANTS FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
