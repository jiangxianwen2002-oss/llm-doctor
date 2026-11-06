#!/bin/bash
# round 97 (PREREG_r97) (one GPU): Qwen3.5-4B base, arms Q-DXONLY and Q-CONTRA (budget 15, 8 samples) in parallel on one vLLM server.
# usage: bash r97_qwen.sh smoke | full. Milestones in r97.log.
H=$HOME/work; P=$H/venv_vllm/bin/python; M=$H/models; cd $H/ddx86
exec 9>$H/r97q_$1.lock; flock -n 9 || { echo "r97 qwen already running"; exit 1; }
log() { echo "$(date '+%m-%d %T') $*" >> $H/r97.log; }
if ! curl -sf localhost:8000/v1/models >/dev/null; then
  bash $H/vllm_q35.sh 0 8000 $M/Qwen3.5-4B > $H/vllm_r97.log 2>&1 9>&- &
  for i in $(seq 1 90); do curl -sf localhost:8000/v1/models >/dev/null && break; sleep 10; done
  curl -sf localhost:8000/v1/models >/dev/null || { log "R97Q_FAIL server did not start"; exit 1; }
  log "qwen server up"
fi
SIG="--stats sig86/sig_stats.npz --kid sig86/kid_fresh.json"
if [ "$1" = smoke ]; then
  for v in dxonly contra; do $P t1_run_bad97.py --variant $v --model qwen3-4b --out smoke97_q_$v --patients 3 --samples 1 --concurrency 3 $SIG > $H/r97_smoke_q_$v.log 2>&1 & done
  wait; log "R97Q_SMOKE_DONE"
else
  log "R97Q start; md5 $(md5sum t1_run_bad97.py ddx_bad97.py t1_run_bad.py ddx_bad.py ddx_signal.py t1_run_s2.py t1_run.py | awk '{print substr($1,1,8)}' | tr '\n' ' ')"
  for v in dxonly contra; do
    ( $P t1_run_bad97.py --variant $v --model qwen3-4b --out q_${v}97 --samples 8 --concurrency 64 $SIG > $H/r97_q_$v.log 2>&1
      log "Q-${v^^} done: $(wc -l < q_${v}97/episodes.jsonl) episodes" ) &
  done
  wait; log "R97Q_DONE"
fi
