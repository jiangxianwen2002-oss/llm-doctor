#!/usr/bin/env python3
"""Round 97 addendum: t1_run_bad97.py with the "qonly" variant registered (ddx_bad97q.py); nothing else changes.
  python t1_run_bad97q.py --variant qonly --model qwen3-4b --out q_qonly97 --stats sig86/sig_stats.npz --kid sig86/kid_fresh.json
"""
import ddx_bad97q  # noqa: F401  registers "qonly"
import t1_run_bad97

if __name__ == "__main__":
    t1_run_bad97.main()
