#!/bin/bash
# round 98 launcher (PREREG_r98): Qwen3.8-Flash. usage: bash run98.sh smoke | full | extra
# Run from a folder containing t1_run*.py, ddx_*.py, t1_run_api_qwen.py, data/ (= data_fresh) and sig86/. The API key is piped to each process's stdin; it is never in argv or logs.
cd "$(dirname "$0")"; PY=python
K() { printf '%s\n' "$QWEN_API_KEY"; }   # set QWEN_API_KEY in your environment
A="--base-url ${BASE_URL:-https://maas.qianwenaiapi.com/compatible-mode/v1} --model qwen3.8-flash"
S="--stats sig86/sig_stats.npz --kid sig86/kid_fresh.json"
run() { local b=$1 s=$2 o=$3; shift 3; K | $PY t1_run_api_qwen.py "$b" "$s" $A --out "$o" "$@" > "$o.log" 2>&1 & }
case "$1" in
  smoke)
    U="--patients 3 --samples 1 --concurrency 3"
    run 15 t1_run_s2.py smoke_none15 $U
    run 15 t1_run_sig.py smoke_sig15 $S $U ;;
  full)
    U="--samples 2 --concurrency 48"
    run 15 t1_run_s2.py f_none_b15_98 $U
    run 10 t1_run_sig.py f_sig0_b10_98 $S $U
    run 15 t1_run_sig.py f_sig0_b15_98 $S $U
    run 15 t1_run_bad97q.py f_qonly_b15_98 --variant qonly $S $U ;;
  extra)
    U="--samples 2 --concurrency 48"
    run 10 t1_run_s2.py f_none_b10_98 $U
    run 15 t1_run_bad97.py f_dxonly_b15_98 --variant dxonly $S $U
    run 15 t1_run_bad97.py f_contra_b15_98 --variant contra $S $U ;;
  *) echo "usage: bash run98.sh smoke | full | extra"; exit 1 ;;
esac
wait; echo R98_DONE
