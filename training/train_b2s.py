#!/usr/bin/env python3
"""Round 42 step 2: LoRA SFT of Qwen3.5-4B (thinking off) on the few-label teacher demonstrations (data_b2/sft.jsonl).
Loss only on the assistant line. Hyper-parameters fixed before training. Writes the merged model to OUT.
  python train_b2s.py --model <path>/Qwen3.5-4B --data t1_premise/sig86/sft_sig.jsonl --out <path>/sft_w_advice --epochs 1 --seed 1
  (SFT w/o advice: --data t1_premise/sig86/sft_plain.jsonl)
"""
import argparse, json, math, random, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--r", type=int, default=16); ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--max-len", type=int, default=3072); ap.add_argument("--tokens-per-step", type=int, default=32768)
ap.add_argument("--seed", type=int); ap.add_argument("--limit", type=int); ap.add_argument("--dry", action="store_true")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
rows = [json.loads(l) for l in open(a.data)]
random.Random(0 if a.seed is None else a.seed).shuffle(rows)
if a.seed is not None:
    import numpy as _np; random.seed(a.seed); _np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
rows = rows[: a.limit] if a.limit else rows
END = "<|im_end|>\n"
data = []
for r in rows:
    m = r["messages"]
    prompt = tok.apply_chat_template(m[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]; t_ids = tok(m[2]["content"] + END, add_special_tokens=False)["input_ids"]
    ids = (p_ids + t_ids)[: a.max_len]; lab = ([-100] * len(p_ids) + t_ids)[: a.max_len]
    data.append((ids, lab))
print(f"samples {len(data)}; mean len {sum(len(d[0]) for d in data) / len(data):.0f}; example prompt tail: {prompt[-120:]!r} target {m[2]['content']!r}", flush=True)
if a.dry:
    raise SystemExit

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
names = {n.split(".")[-1] for n, mod in model.named_modules() if isinstance(mod, torch.nn.Linear) and "visual" not in n and "lm_head" not in n}
cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=sorted(names), task_type="CAUSAL_LM")
model = get_peft_model(model, cfg); model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)

# token-budget batches (sorted by length within shuffled chunks), gradient accumulation per sample
order = list(range(len(data)))
steps_total = 0
batches_per_epoch = []
cur, cur_tok = [], 0
for i in order:
    cur.append(i); cur_tok += len(data[i][0])
    if cur_tok >= a.tokens_per_step:
        batches_per_epoch.append(cur); cur, cur_tok = [], 0
if cur:
    batches_per_epoch.append(cur)
n_steps = math.ceil(len(batches_per_epoch) * a.epochs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / n_steps))
model.train(); t0 = time.time(); step = 0; seen_tok = 0
while step < n_steps:
    for b in batches_per_epoch:
        if step >= n_steps:
            break
        ntgt = sum(sum(1 for x in data[i][1] if x != -100) for i in b); loss_sum = 0.0
        for i in b:
            ids = torch.tensor([data[i][0]], device="cuda"); lab = torch.tensor([data[i][1]], device="cuda")
            k = sum(1 for x in data[i][1] if x != -100) + 1
            out = model(input_ids=ids, logits_to_keep=k)
            logits = out.logits[:, :-1].float(); tgt = lab[:, -k + 1:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="sum") / ntgt
            loss.backward(); loss_sum += loss.item(); seen_tok += ids.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        el = time.time() - t0
        print(f"\rstep {step}/{n_steps}  loss {loss_sum:.4f}  {seen_tok / el:6.0f} tok/s  elapsed {el / 60:5.1f} min  eta {el / step * (n_steps - step) / 60:5.1f} min   ", end="", flush=True)
print("\nmerging ...", flush=True)
model = model.merge_and_unload(); model.save_pretrained(a.out, safe_serialization=True); tok.save_pretrained(a.out)
print(f"saved -> {a.out}")
