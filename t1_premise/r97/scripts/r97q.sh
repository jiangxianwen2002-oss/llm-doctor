#!/bin/bash
# round 97 addendum (PREREG_r97 section 8), one GPU: Q-QONLY (Qwen base) then SFT-QONLY (SFT w/ advice), budget 15, 8 samples.
# usage: bash r97q.sh smoke | full. Milestones in r97q.log. Waits only on its own client PIDs (never on the vLLM child).
H=$HOME/work; P=$H/venv_vllm/bin/python; M=$H/models; cd $H/ddx86
exec 9>$H/r97q_$1.lock; flock -n 9 || { echo "already running"; exit 1; }
log() { echo "$(date '+%m-%d %T') $*" >> $H/r97q.log; }
SIG="--stats sig86/sig_stats.npz --kid sig86/kid_fresh.json"
serve() {  # $1 model dir
  OLD=$(pgrep -f "vllm serve" | head -1)
  if [ -n "$OLD" ]; then
    curl -sf localhost:8000/v1/models | grep -q "$1" && { log "server $1 already up"; return 0; }
    kill $OLD; for i in $(seq 1 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && break; sleep 5; done
  fi
  bash $H/vllm_q35.sh 0 8000 $1 > $H/vllm_r97q_$(basename $1).log 2>&1 9>&- &
  for i in $(seq 1 90); do curl -sf localhost:8000/v1/models | grep -q "$1" && { log "server $1 up"; return 0; }; sleep 10; done
  log "R97Q_FAIL server $1"; exit 1
}
run() {  # $1 out, $2 samples, $3 patients(optional)
  local extra=""; [ -n "$3" ] && extra="--patients $3 --concurrency 3"
  $P t1_run_bad97q.py --variant qonly --model qwen3-4b --out $1 --samples $2 $extra $SIG > $H/r97q_$1.log 2>&1
  log "$1 done: $(wc -l < $1/episodes.jsonl) episodes"
}
if [ "$1" = smoke ]; then
  serve $M/Qwen3.5-4B; run smoke97q_q 1 3
  serve $M/q35-sigt86-serve; run smoke97q_sft 1 3
  log "R97Q_SMOKE_DONE"
else
  log "R97Q full start; md5 $(md5sum t1_run_bad97q.py ddx_bad97q.py t1_run_bad97.py ddx_bad97.py t1_run_bad.py | awk '{print substr($1,1,8)}' | tr '\n' ' ')"
  serve $M/Qwen3.5-4B; run q_qonly97 8
  serve $M/q35-sigt86-serve; run sft_qonly97 8
  log "R97Q_DONE"
fi
