#!/usr/bin/env python3
"""Rounds 40/42: fresh patient sets + S2 menus rebuilt without them. Diagnoses are read only to stratify sampling
(6 per disease) and to store the evaluation/teacher label; S2 menus never read diagnoses.
Outputs t1_premise/data_fresh/{patients.json, menu.json, conditions.json, s2_menu.json}, t1_premise/data_b2/labelled.json
"""
import csv, json, random, shutil, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "ddx/tools")); sys.path.insert(0, str(ROOT / "tools/dir14_probes"))
import make_t1_package as MK
import a1_core as A
rows = list(csv.DictReader(open(MK.SRC / "raw/release_validate_patients")))
by = defaultdict(list)
for i, r in enumerate(rows):
    by[r["PATHOLOGY"]].append(i)
names = sorted(by); used = set()
for seed in (1000, 1):
    rng = random.Random(seed)
    for n in names:
        used.update(rng.sample(by[n], min(60, len(by[n]))))
man = json.load(open(ROOT / "ddx/data/ddx_split/split_manifest.json"))
used.update(v[1] for v in man["provenance"]["source_row"].values() if v[0] == "validate")
used.update(p["validate_row"] for p in json.load(open(ROOT / "t1_premise/data/patients.json")))

def draw(seed, per, total):
    rng = random.Random(seed); free = {n: [i for i in by[n] if i not in used] for n in names}; pick = []
    for n in names:
        pick += rng.sample(free[n], min(per, len(free[n])))
    if total:
        rest = [i for n in names for i in free[n] if i not in set(pick)]; pick += rng.sample(rest, total - len(pick))
    rng.shuffle(pick); used.update(pick); return pick

def rec(k, i):
    r = rows[i]
    return {"pid": k, "validate_row": i, "age": r["AGE"], "sex": "female" if r["SEX"] == "F" else "male",
            "initial": MK.QS.index(r["INITIAL_EVIDENCE"]), "diagnosis": r["PATHOLOGY"], "answers": MK.answers(r)}

fresh = draw(40, 6, 300); lab = draw(42, 5, 0)
out = ROOT / "t1_premise/data_fresh"; out.mkdir(exist_ok=True)
json.dump([rec(k, i) for k, i in enumerate(fresh)], open(out / "patients.json", "w"))
for f in ("menu.json", "conditions.json"):
    shutil.copy(ROOT / "t1_premise/data" / f, out / f)
(ROOT / "t1_premise/data_b2").mkdir(exist_ok=True)
json.dump([rec(k, i) for k, i in enumerate(lab)], open(ROOT / "t1_premise/data_b2/labelled.json", "w"))
# S2 menus from the unlabelled pool, excluding banks, donors, exploration 300, fresh 300 and labelled 245
rng = random.Random(1000); bank = set()
for n in names:
    bank.update(rng.sample(by[n], min(60, len(by[n]))))
pool = [j for j in range(len(rows)) if j not in used or j in bank]
pool = [j for j in pool if j not in bank and j not in set(fresh) and j not in set(lab)]
pool = [j for j in range(len(rows)) if j not in used]
dk = [A.dkey(q) for q in A.QS]
pos = [[ks[i] != dk[i] for i in range(A.nQ)] for ks in (A.keys_of(rows[j]["EVIDENCES"]) for j in pool)]
init = [A.QIDX[rows[j]["INITIAL_EVIDENCE"]] for j in pool]
glob = [sum(p[i] for p in pos) / len(pos) for i in range(A.nQ)]
grp = defaultdict(list)
for p, i0 in zip(pos, init):
    grp[i0].append(p)
menus = {}
for i0 in range(A.nQ):
    g = grp.get(i0, []); sc = glob[:] if len(g) < 20 else [sum(p[i] for p in g) / len(g) for i in range(A.nQ)]
    sc[i0] = -1; menus[i0] = sorted(range(A.nQ), key=lambda i: -sc[i])[:50]
json.dump(menus, open(out / "s2_menu.json", "w"))
old = {int(k): v for k, v in json.load(open(ROOT / "t1_premise/data/s2_menu.json")).items()}
same = sum(len(set(old[k]) & set(menus[k])) for k in menus) / (50 * len(menus))
print(f"fresh {len(fresh)}, labelled {len(lab)} ({len(set(rows[i]['PATHOLOGY'] for i in lab))} diseases), pool {len(pool)}; S2 overlap with exploration menus {100*same:.1f}%")
