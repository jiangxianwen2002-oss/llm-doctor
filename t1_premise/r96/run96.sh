#!/bin/bash
# round 96 launcher (PREREG_r96). usage: bash run96.sh smoke | full
# Run from a folder containing t1_run*.py, ddx_*.py, t1_run_api.py, data/ (= data_fresh) and sig86/. The API key is piped to each process's stdin; it is never in argv or logs.
cd "$(dirname "$0")"; PY=python
K() { printf '%s' "$DEEPSEEK_API_KEY"; }   # set DEEPSEEK_API_KEY in your environment
S="--stats sig86/sig_stats.npz --kid sig86/kid_fresh.json"
if [ "$1" = smoke ]; then
  U="--base-url https://api.deepseek.com --model deepseek-flash --patients 3 --samples 1 --concurrency 3"
  K | $PY t1_run_api.py 15 t1_run_s2.py $U --out smoke_none > smoke_none.log 2>&1 &
  K | $PY t1_run_api.py 15 t1_run_sig.py $U $S --out smoke_sig > smoke_sig.log 2>&1 &
  K | $PY t1_run_api.py 15 t1_run_bad.py $U $S --out smoke_bad > smoke_bad.log 2>&1 &
  wait; echo SMOKE_DONE
else
  U="--base-url https://api.deepseek.com --model deepseek-flash --samples 2 --concurrency 48"
  K | $PY t1_run_api.py 15 t1_run_s2.py $U --out ds_none96 > ds_none96.log 2>&1 &
  K | $PY t1_run_api.py 15 t1_run_sig.py $U $S --out ds_sig0_96 > ds_sig0_96.log 2>&1 &
  K | $PY t1_run_api.py 15 t1_run_bad.py $U $S --out ds_bad96 > ds_bad96.log 2>&1 &
  wait; echo R96_DONE
fi
