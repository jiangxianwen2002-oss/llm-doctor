#!/bin/bash
# round 99 launcher (PREREG_r99): GLM-5.3-Flash. usage: bash run99.sh smoke | full
# Run from a folder containing t1_run*.py, ddx_*.py, t1_run_api_glm.py, data/ (= data_fresh) and sig86/. The API key is piped to each process's stdin; it is never in argv or logs.
# This model cannot turn thinking off: MAX_TOKENS leaves room for its reasoning tokens (replies themselves stay within 40 tokens); TOKEN_CAP stops an arm past that many tokens.
cd "$(dirname "$0")"; PY=python
export TOKEN_CAP=${TOKEN_CAP:-11500000} MAX_TOKENS=${MAX_TOKENS:-4096}
K() { printf '%s\n' "$GLM_API_KEY"; }   # set GLM_API_KEY in your environment
A="--base-url ${BASE_URL:-https://open.bigmodel.cn/api/paas/v4} --model glm-5.3-flash"
S="--stats sig86/sig_stats.npz --kid sig86/kid_fresh.json"
run() { local b=$1 s=$2 o=$3; shift 3; K | $PY t1_run_api_glm.py "$b" "$s" $A --out "$o" "$@" > "$o.log" 2>&1 & }
case "$1" in
  smoke)
    U="--patients 3 --samples 1 --concurrency 3"
    run 15 t1_run_s2.py smoke_none15 $U
    run 15 t1_run_sig.py smoke_sig15 $S $U ;;
  full)
    U="--samples 2 --concurrency 24"
    run 10 t1_run_s2.py g_none_b10_99 $U
    run 15 t1_run_s2.py g_none_b15_99 $U
    run 10 t1_run_sig.py g_sig0_b10_99 $S $U
    run 15 t1_run_sig.py g_sig0_b15_99 $S $U
    run 15 t1_run_bad97.py g_dxonly_b15_99 --variant dxonly $S $U
    run 15 t1_run_bad97q.py g_qonly_b15_99 --variant qonly $S $U
    run 15 t1_run_bad97.py g_contra_b15_99 --variant contra $S $U ;;
  *) echo "usage: bash run99.sh smoke | full"; exit 1 ;;
esac
wait; echo R99_DONE
