#!/bin/bash
# round 97 (PREREG_r97), DeepSeek API arms from the GPU server: DS-DXONLY, DS-CONTRA (budget 15), DS-NONE-B10, DS-SIG0-B10; 2 samples.
# usage (key typed at a hidden prompt locally, sent over ssh stdin, never in argv / files / logs):
#   read -rs K; printf '%s\n' "$K" | ssh <server> 'bash $HOME/work/ddx86/r97_ds.sh smoke|full'; unset K
H=$HOME/work; P=$H/venv_vllm/bin/python; cd $H/ddx86
read -r KEY; [ -n "$KEY" ] || { echo "no key on stdin"; exit 1; }
log() { echo "$(date '+%m-%d %T') $*" >> $H/r97.log; }
SIG="--stats sig86/sig_stats.npz --kid sig86/kid_fresh.json"
run() {  # $1 budget, $2 script, $3 out, rest: extra args. Detached so it survives the ssh session.
  local b=$1 s=$2 o=$3; shift 3
  pgrep -f -- "--out $o " >/dev/null && { echo "$o already running"; return; }   # an ssh retry must not start a second copy
  printf '%s\n' "$KEY" | setsid nohup bash -c "$P t1_run_api.py $b $s --base-url https://api.deepseek.com --model deepseek-flash --out $o $* > $H/r97_$o.log 2>&1; echo \"\$(date '+%m-%d %T') DS $o done: \$(wc -l < $o/episodes.jsonl) episodes\" >> $H/r97.log" >/dev/null 2>&1 &
}
if [ "$1" = smoke ]; then
  U="--patients 3 --samples 1 --concurrency 3"
  run 15 t1_run_bad97.py smoke97_ds_dxonly --variant dxonly $SIG $U
  run 15 t1_run_bad97.py smoke97_ds_contra --variant contra $SIG $U
  run 10 t1_run_s2.py smoke97_ds_none10 $U
  run 10 t1_run_sig.py smoke97_ds_sig10 $SIG $U
else
  log "R97DS start; md5 $(md5sum t1_run_api.py t1_run_bad97.py ddx_bad97.py t1_run_sig.py t1_run_s2.py | awk '{print substr($1,1,8)}' | tr '\n' ' ')"
  U="--samples 2 --concurrency 48"
  run 15 t1_run_bad97.py ds_dxonly97 --variant dxonly $SIG $U
  run 15 t1_run_bad97.py ds_contra97 --variant contra $SIG $U
  run 10 t1_run_s2.py ds_none_b10_97 $U
  run 10 t1_run_sig.py ds_sig0_b10_97 $SIG $U
fi
echo "launched $1"
