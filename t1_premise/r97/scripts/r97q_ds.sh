#!/bin/bash
# round 97 addendum: DS-QONLY via the DeepSeek API from the GPU server (key on stdin only). usage: ... | bash r97q_ds.sh smoke|full
H=$HOME/work; P=$H/venv_vllm/bin/python; cd $H/ddx86
read -r KEY; [ -n "$KEY" ] || { echo "no key on stdin"; exit 1; }
SIG="--stats sig86/sig_stats.npz --kid sig86/kid_fresh.json"
o=ds_qonly97; U="--samples 2 --concurrency 48"; [ "$1" = smoke ] && { o=smoke97q_ds; U="--patients 3 --samples 1 --concurrency 3"; }
pgrep -f -- "--out $o " >/dev/null && { echo "$o already running"; exit 0; }
printf '%s\n' "$KEY" | setsid nohup bash -c "$P t1_run_api.py 15 t1_run_bad97q.py --variant qonly --base-url https://api.deepseek.com --model deepseek-flash --out $o $SIG $U > $H/r97q_$o.log 2>&1; echo \"\$(date '+%m-%d %T') $o done: \$(wc -l < $o/episodes.jsonl) episodes\" >> $H/r97q.log" >/dev/null 2>&1 &
echo "launched $o"
