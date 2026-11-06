#!/usr/bin/env python3
"""Round 49: 800 random unused validate patients for label-free self-training. Writes t1_selftrain_data/patients.json with
diagnosis = "" (unknown) and the true diagnoses to t1_premise/sealed_round49_labels.json (read only after training)."""
import csv, json, random, shutil, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path("."); sys.path.insert(0, str(ROOT / "ddx/tools"))
import make_t1_package as MK
rows = list(csv.DictReader(open(MK.SRC / "raw/release_validate_patients")))
by = defaultdict(list)
for i, r in enumerate(rows):
    by[r["PATHOLOGY"]].append(i)
used = set()
for seed in (1000, 1):
    rng = random.Random(seed)
    for n in sorted(by):
        used.update(rng.sample(by[n], min(60, len(by[n]))))
man = json.load(open(ROOT / "ddx/data/ddx_split/split_manifest.json"))
used.update(v[1] for v in man["provenance"]["source_row"].values() if v[0] == "validate")
for f in ("t1_premise/data/patients.json", "t1_premise/data_fresh/patients.json", "t1_premise/data_b2/labelled.json"):
    used.update(p["validate_row"] for p in json.load(open(ROOT / f)))
free = [i for i in range(len(rows)) if i not in used]
pick = random.Random(49).sample(free, 800)
out = ROOT / "t1_premise/data_selftrain"; out.mkdir(exist_ok=True)
pats, sealed = [], {}
for k, i in enumerate(pick):
    r = rows[i]
    pats.append({"pid": k, "validate_row": i, "age": r["AGE"], "sex": "female" if r["SEX"] == "F" else "male",
                 "initial": MK.QS.index(r["INITIAL_EVIDENCE"]), "diagnosis": "", "answers": MK.answers(r)})
    sealed[k] = r["PATHOLOGY"]
json.dump(pats, open(out / "patients.json", "w"))
for f in ("menu.json", "conditions.json", "s2_menu.json"):
    shutil.copy(ROOT / "t1_premise/data" / f, out / f)
json.dump(sealed, open(ROOT / "t1_premise/sealed_round49_labels.json", "w"))
print("unlabelled patients", len(pats), "from", len(free), "free rows")
