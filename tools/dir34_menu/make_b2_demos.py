#!/usr/bin/env python3
"""Round 42 step 1 (free): few-label teacher demonstrations for B2. Statistics come ONLY from the labelled patients
(leave-one-out); the teacher asks inside the patient's S2 menu (t1_premise/data/s2_menu.json), stops at posterior >= 0.9 or
10 questions, then states the TRUE diagnosis. 4 demonstrations per patient (seed 0 argmax EIG, seeds 1-3 sample top-3).
Writes turn-level chat samples in the exact t1_run_s2 prompt format: t1_premise/data_b2/sft.jsonl
"""
import csv, json, random, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "tools/dir14_probes")); sys.path.insert(0, str(ROOT / "t1_premise"))
import a1_core as A
import t1_run_s2 as S
import t1_run as T

MAXQ, TAU, TOPK = 10, 0.9, 30
lab = json.load(open(ROOT / "t1_premise/data_b2/labelled.json"))
rows = list(csv.DictReader(open(A.SRC / "raw/release_validate_patients")))
menu = json.load(open(ROOT / "t1_premise/data/menu.json")); conds = json.load(open(ROOT / "t1_premise/data/conditions.json"))
s2 = {int(k): v for k, v in json.load(open(ROOT / "t1_premise/data/s2_menu.json")).items()}
Y = {n: i for i, n in enumerate(conds)}; nY = len(conds); nQ = A.nQ
keys = [A.keys_of(rows[p["validate_row"]]["EVIDENCES"]) for p in lab]
ys = [Y[p["diagnosis"]] for p in lab]
kmap = []
for qi in range(nQ):
    top = [k for k, _ in Counter(ks[qi] for ks in keys).most_common(TOPK)]; kmap.append({k: j for j, k in enumerate(top)})
Kq = np.array([len(m) + 1 for m in kmap])
C = np.zeros((nQ, nY, TOPK + 1)); n = np.zeros((nQ, nY))
for ks, y in zip(keys, ys):
    for qi in range(nQ):
        C[qi, y, kmap[qi].get(ks[qi], TOPK)] += 1; n[qi, y] += 1
mask = np.zeros((nQ, 1, TOPK + 1))
for qi in range(nQ):
    mask[qi, 0, :Kq[qi] - 1] = 1; mask[qi, 0, TOPK] = 1


def stats_without(j):
    Cj = C.copy(); nj = n.copy()
    for qi in range(nQ):
        Cj[qi, ys[j], kmap[qi].get(keys[j][qi], TOPK)] -= 1; nj[qi, ys[j]] -= 1
    L = (Cj + 1) / (nj[:, :, None] + Kq[:, None, None]) * mask
    return L, np.log(np.where(L > 0, L, 1))


sys_p = S.system_prompt(conds); out = open(ROOT / "t1_premise/data_b2/sft.jsonl", "w")
nturn = 0; nqs = []; teacher_ok = []
for j, p in enumerate(lab):
    L, logL = stats_without(j); local = s2[p["initial"]]
    for seed in range(4):
        rng = random.Random(1000 * j + seed); hist = []
        q0 = p["initial"]; logp = np.log(L[q0, :, kmap[q0].get(keys[j][q0], TOPK)])
        while True:
            pi = np.exp(logp - logp.max()); pi /= pi.sum()
            if pi.max() >= TAU or len(hist) >= MAXQ:
                break
            pk = np.einsum("y,qyk->qk", pi, L)
            eig = np.einsum("y,qyk->q", pi, L * (logL - np.log(np.where(pk > 0, pk, 1))[:, None, :]))
            cand = sorted((i for i, q in enumerate(local) if q not in hist and q != q0), key=lambda i: -eig[local[i]])
            if not cand:
                break
            i = cand[0] if seed == 0 else rng.choice(cand[:3]); q = local[i]
            out.write(json.dumps({"messages": [{"role": "system", "content": sys_p},
                                               {"role": "user", "content": S.user_prompt(p, menu, local, hist, False)},
                                               {"role": "assistant", "content": f"ASK {i + 1}"}]}) + "\n"); nturn += 1
            hist.append(q); logp = logp + np.log(L[q, :, kmap[q].get(keys[j][q], TOPK)])
        out.write(json.dumps({"messages": [{"role": "system", "content": sys_p},
                                           {"role": "user", "content": S.user_prompt(p, menu, local, hist, False)},
                                           {"role": "assistant", "content": f"DIAGNOSIS: {p['diagnosis']}"}]}) + "\n"); nturn += 1
        nqs.append(len(hist)); teacher_ok.append(int(np.argmax(logp)) == ys[j])
print(f"labelled patients {len(lab)}; demonstrations {len(nqs)}; turn samples {nturn}; questions/demo {np.mean(nqs):.1f}; "
      f"teacher's own LOO argmax = truth {100 * np.mean(teacher_ok):.1f}% (demos always end with the TRUE label)")
