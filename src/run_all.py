"""Regenerate every result and figure, then verify. Fixed seeds throughout."""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
for script in ["t5_passk_crossover.py", "t2_t6.py", "t1_t3_t4_bridge.py",
               "c6_correction.py", "t_star.py", "empirical.py",
               "make_figure.py", "paper_figures.py", "check_numbers.py"]:
    print("\n### running", script, flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, script)])
    if r.returncode != 0 and script != "check_numbers.py":
        sys.exit("FAILED: " + script)
    if script == "check_numbers.py":
        sys.exit(r.returncode)
