#!/bin/bash
# round 97 addendum (PREREG_r97 section 6) (one GPU): Qwen SFT w/ advice (q35-sigt86-serve), arms SFT-DXONLY and SFT-CONTRA
# (budget 15, 8 samples) in parallel. Stops the base-model vLLM first. Milestones in r97.log.
H=$HOME/work; P=$H/venv_vllm/bin/python; M=$H/models; cd $H/ddx86
exec 9>$H/r97sft.lock; flock -n 9 || { echo "r97 sft already running"; exit 1; }
log() { echo "$(date '+%m-%d %T') $*" >> $H/r97.log; }
OLD=$(pgrep -f "vllm serve $M/Qwen3.5-4B" | head -1)
[ -n "$OLD" ] && { kill $OLD; for i in $(seq 1 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && break; sleep 5; done; log "base server stopped"; }
bash $H/vllm_q35.sh 0 8000 $M/q35-sigt86-serve > $H/vllm_r97_sft.log 2>&1 9>&- &
for i in $(seq 1 90); do curl -sf localhost:8000/v1/models >/dev/null && break; sleep 10; done
curl -sf localhost:8000/v1/models >/dev/null || { log "R97SFT_FAIL server did not start"; exit 1; }
log "sft server up"
SIG="--stats sig86/sig_stats.npz --kid sig86/kid_fresh.json"
for v in dxonly contra; do $P t1_run_bad97.py --variant $v --model qwen3-4b --out smoke97_sft_$v --patients 3 --samples 1 --concurrency 3 $SIG > $H/r97_smoke_sft_$v.log 2>&1 & done; wait
for v in dxonly contra; do [ "$(wc -l < smoke97_sft_$v/episodes.jsonl)" = 3 ] || { log "R97SFT_FAIL smoke $v"; exit 1; }; done
log "R97SFT smoke ok; start; md5 $(md5sum t1_run_bad97.py ddx_bad97.py t1_run_bad.py | awk '{print substr($1,1,8)}' | tr '\n' ' ')"
for v in dxonly contra; do
  ( $P t1_run_bad97.py --variant $v --model qwen3-4b --out sft_${v}97 --samples 8 --concurrency 64 $SIG > $H/r97_sft_$v.log 2>&1
    log "SFT-${v^^} done: $(wc -l < sft_${v}97/episodes.jsonl) episodes" ) &
done
wait; log "R97SFT_DONE"
