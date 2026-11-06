#!/usr/bin/env python3
"""Round 36: label-free S2 complaint-conditioned menus for the premise patients. Diagnoses are never read.
  python -I tools/dir34_menu/make_s2_menu.py
"""
import csv, json, random, sys
from collections import defaultdict
sys.path.insert(0, "./tools/dir14_probes")
import a1_core as A
K = 50
val = list(csv.DictReader(open(A.SRC / "raw/release_validate_patients")))
by = defaultdict(list)
for j, r in enumerate(val):
    by[r["PATHOLOGY"]].append(j)          # only to reproduce the bank exclusion (rng 1000), not used for menus
rng = random.Random(1000)
bank = {j for n in sorted(by) for j in rng.sample(by[n], min(60, len(by[n])))}
prem = {p["validate_row"] for p in json.load(open("./t1_premise/data/patients.json"))}
pool = [j for j in range(len(val)) if j not in bank and j not in prem]
dk = [A.dkey(q) for q in A.QS]
pos = [[ks[i] != dk[i] for i in range(A.nQ)] for ks in (A.keys_of(val[j]["EVIDENCES"]) for j in pool)]
init = [A.QIDX[val[j]["INITIAL_EVIDENCE"]] for j in pool]
glob = [sum(p[i] for p in pos) / len(pos) for i in range(A.nQ)]
grp = defaultdict(list)
for p, i0 in zip(pos, init):
    grp[i0].append(p)
menus = {}
for i0 in range(A.nQ):
    g = grp.get(i0, [])
    sc = glob[:] if len(g) < 20 else [sum(p[i] for p in g) / len(g) for i in range(A.nQ)]
    sc[i0] = -1
    menus[i0] = sorted(range(A.nQ), key=lambda i: -sc[i])[:K]
json.dump(menus, open("./t1_premise/data/s2_menu.json", "w"))
print("pool", len(pool), "complaints with own stats", sum(len(v) >= 20 for v in grp.values()))
