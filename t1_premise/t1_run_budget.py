#!/usr/bin/env python3
"""Round 95: run an existing evaluation script with a different question budget (PREREG_r95). Sets t1_run.BUDGET before the
script's main() runs; t1_run_s2 / t1_run_sig read T.BUDGET at call time, so the prompt says "at most B questions", the
progress line "You have asked k of B questions", and the final (forced-diagnosis) turn comes after B questions. Nothing else
changes; the signal (ddx_signal.py, teacher stops at 10) is unchanged.
  python t1_run_budget.py 10 t1_run_sig.py --model qwen3-4b --out sig0_b10_95 --stats sig86/sig_stats.npz --kid sig86/kid_fresh.json"""
import runpy, sys
import t1_run as T
T.BUDGET = int(sys.argv[1]); script = sys.argv[2]
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
