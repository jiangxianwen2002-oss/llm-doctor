#!/usr/bin/env python3
"""Round 86 (free, offline): build the DDXPlus SIG-T inputs from the round-42 EIG teacher (tools/dir34_menu/make_b2_demos.py).
 1) t1_premise/sig86/sig_stats.npz      L[q, y, k] from ALL 235 labelled patients (used live at test time)
 2) t1_premise/sig86/kid_<set>.json     per evaluation patient: answer-key index k for every question (no raw data on the server)
 3) t1_premise/sig86/sft_plain.jsonl    the round-42 demonstrations regenerated (must be byte-identical to data_b2/sft.jsonl)
    t1_premise/sig86/sft_sig.jsonl      the same samples with the signal line inserted (leave-one-out statistics, as the teacher)
  python tools/dir34_menu/build_ddx_signal.py
"""
import csv, hashlib, json, random, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "tools/dir14_probes")); sys.path.insert(0, str(ROOT / "t1_premise"))
import a1_core as A
import t1_run_s2 as S
import ddx_signal as D

MAXQ, TAU, TOPK = 10, 0.9, 30
OUT = ROOT / "t1_premise/sig86"; OUT.mkdir(exist_ok=True)
lab = json.load(open(ROOT / "t1_premise/data_b2/labelled.json"))
rows = list(csv.DictReader(open(A.SRC / "raw/release_validate_patients")))
menu = json.load(open(ROOT / "t1_premise/data/menu.json")); conds = json.load(open(ROOT / "t1_premise/data/conditions.json"))
s2 = {int(k): v for k, v in json.load(open(ROOT / "t1_premise/data/s2_menu.json")).items()}
Y = {n: i for i, n in enumerate(conds)}; nY = len(conds); nQ = A.nQ
keys = [A.keys_of(rows[int(p["validate_row"])]["EVIDENCES"]) for p in lab]
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


def L_of(Cx, nx):
    return (Cx + 1) / (nx[:, :, None] + Kq[:, None, None]) * mask


def stats_without(j):
    Cj = C.copy(); nj = n.copy()
    for qi in range(nQ):
        Cj[qi, ys[j], kmap[qi].get(keys[j][qi], TOPK)] -= 1; nj[qi, ys[j]] -= 1
    L = L_of(Cj, nj); return L, np.log(np.where(L > 0, L, 1))


assert len(lab) == D.LABEL_N or print(f"NOTE: {len(lab)} labelled patients (signal text says {D.LABEL_N})")
# 1) full statistics for test time
np.savez_compressed(OUT / "sig_stats.npz", L=L_of(C, n))
# 2) answer-key indices for evaluation sets
for name, path in [("fresh", "t1_premise/data_fresh/patients.json"), ("explore", "t1_premise/data/patients.json")]:
    pts = json.load(open(ROOT / path))
    kid = {p["pid"]: [kmap[qi].get(k, TOPK) for qi, k in enumerate(A.keys_of(rows[int(p["validate_row"])]["EVIDENCES"]))] for p in pts}
    json.dump(kid, open(OUT / f"kid_{name}.json", "w")); print(f"kid_{name}: {len(kid)} patients")
# 3) demonstrations: plain (must reproduce round 42) and with the signal (LOO statistics, as the teacher)
sys_p = S.system_prompt(conds)
fp = open(OUT / "sft_plain.jsonl", "w"); fs = open(OUT / "sft_sig.jsonl", "w")
agree = {"ask_best": 0, "ask_seed0": 0, "ask_in_top3": 0, "ask_rand": 0, "dx_rec": 0, "dx_rec_eq_truth": 0}
for j, p in enumerate(lab):
    L, logL = stats_without(j); st = D.Stats(L, conds); local = s2[int(p["initial"])]
    kid = [kmap[qi].get(keys[j][qi], TOPK) for qi in range(nQ)]
    for seed in range(4):
        rng = random.Random(1000 * j + seed); hist = []
        q0 = int(p["initial"]); logp = np.log(L[q0, :, kmap[q0].get(keys[j][q0], TOPK)])
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
            up = S.user_prompt(p, menu, local, hist, False); line, kind, val = st.line(q0, local, hist, kid)
            assert kind == "ask"; agree["ask_seed0" if seed == 0 else "ask_rand"] += 1
            agree["ask_best"] += i == val[0]; agree["ask_in_top3"] += i in val
            for f, u in ((fp, up), (fs, D.insert(up, line))):
                f.write(json.dumps({"messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": u},
                                                 {"role": "assistant", "content": f"ASK {i + 1}"}]}) + "\n")
            hist.append(q); logp = logp + np.log(L[q, :, kmap[q].get(keys[j][q], TOPK)])
        up = S.user_prompt(p, menu, local, hist, False); line, kind, val = st.line(q0, local, hist, kid)
        agree["dx_rec"] += kind == "dx"; agree["dx_rec_eq_truth"] += kind == "dx" and conds[val] == p["diagnosis"]
        for f, u in ((fp, up), (fs, D.insert(up, line))):
            f.write(json.dumps({"messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": u},
                                             {"role": "assistant", "content": f"DIAGNOSIS: {p['diagnosis']}"}]}) + "\n")
fp.close(); fs.close()
h = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
same = h(OUT / "sft_plain.jsonl") == h(ROOT / "t1_premise/data_b2/sft.jsonl")
print(f"REPRO sft_plain == data_b2/sft.jsonl: {same}")
print("teacher vs signal:", agree)
ex = [json.loads(l) for l in open(OUT / "sft_sig.jsonl")][1]["messages"][1]["content"]
print("example signal:", [l for l in ex.splitlines() if l.startswith("Experience")][0])
