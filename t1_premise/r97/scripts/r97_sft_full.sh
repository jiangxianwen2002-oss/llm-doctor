#!/bin/bash
# round 97 addendum, full SFT arms on the already running q35-sigt86-serve server (r97_sft.sh got stuck in `wait` on its vLLM child).
H=$HOME/work; P=$H/venv_vllm/bin/python; cd $H/ddx86
exec 9>$H/r97sftfull.lock; flock -n 9 || { echo "already running"; exit 1; }
log() { echo "$(date '+%m-%d %T') $*" >> $H/r97.log; }
curl -sf localhost:8000/v1/models | grep -q q35-sigt86-serve || { log "R97SFT_FAIL wrong or no server"; exit 1; }
for v in dxonly contra; do [ "$(wc -l < smoke97_sft_$v/episodes.jsonl)" = 3 ] || { log "R97SFT_FAIL smoke $v"; exit 1; }; done
SIG="--stats sig86/sig_stats.npz --kid sig86/kid_fresh.json"
log "R97SFT smoke ok; start; md5 $(md5sum t1_run_bad97.py ddx_bad97.py t1_run_bad.py | awk '{print substr($1,1,8)}' | tr '\n' ' ')"
P1=; for v in dxonly contra; do
  ( $P t1_run_bad97.py --variant $v --model qwen3-4b --out sft_${v}97 --samples 8 --concurrency 64 $SIG > $H/r97_sft_$v.log 2>&1
    log "SFT-${v^^} done: $(wc -l < sft_${v}97/episodes.jsonl) episodes" ) & P1="$P1 $!"
done
wait $P1; log "R97SFT_DONE"
