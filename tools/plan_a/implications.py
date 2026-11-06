#!/usr/bin/env python3
"""Numbers for the paper's Section 4 (larger models with advice; "flash" in the code). Read-only; run from the project root.
Question counts; disagreement between the LLM and the statistician's most likely condition given the same answers; calls and tokens per interview.
Patient-paired bootstrap, 10k, seed 0, sorted keys. Writes t1_premise/implications.json."""
import json, os, random, statistics as ST, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
R = Path("t1_premise").resolve(); sys.path.insert(0, str(R / "r98")); os.chdir(R / "r98")
import t1_run as T, ddx_signal as D  # noqa: E401
pts, menu, conds = T.load(); s2 = {int(k): v for k, v in json.load(open("data/s2_menu.json")).items()}
st = D.Stats.load(str(R / "sig86/sig_stats.npz"), conds); kids = json.load(open(R / "sig86/kid_fresh.json")); P = {str(p["pid"]): p for p in pts}
def stat_q(B):
    D.MAXQ, D.TAU = B, .9; out = {}
    for pid, p in P.items():
        q0 = int(p["initial"]); h = []
        while True:
            pi = st.posterior(q0, h, kids[pid]); k, v = st.recommend(pi, q0, s2[q0], h)
            if k == "dx": break
            h.append(s2[q0][v[0]])
        out[pid] = len(h)
    D.MAXQ, D.TAU = 10, .9
    return out
A = {"DS": {10: "r97/ds_sig0_b10_97", 15: "r96/ds_sig0_96"}, "QF": {10: "r98/f_sig0_b10_98", 15: "r98/f_sig0_b15_98"}, "GLM": {10: "r99/g_sig0_b10_99", 15: "r99/g_sig0_b15_99"}}
res = {}
def desc(q, B): return dict(mean=ST.mean(q), median=ST.median(q), at_cap=100 * sum(x == B for x in q) / len(q))
for B in (10, 15):
    SQ = stat_q(B); res[f"statistician_q_{B}"] = desc(list(SQ.values()), B); print(f"B={B} statistician questions", res[f"statistician_q_{B}"])
    rows = []
    for m, v in A.items():
        E = [json.loads(l) for l in open(R / v[B] / "episodes.jsonl")]; print(f"  {m} questions", desc([e["n_questions"] for e in E], B))
        for e in E:
            pid = str(e["pid"]); p = P[pid]; nb = conds[int(np.argmax(st.posterior(int(p["initial"]), e["asked"], kids[pid])))]
            rows.append(dict(pid=pid, m=m, q=e["n_questions"], c=float(e["correct"]), dis=e["dx"] != nb, s=float(nb == p["diagnosis"])))
    print("  flash questions", desc([r["q"] for r in rows], B))
    def summ(rs):
        d = [r for r in rs if r["dis"]]; a = [r for r in rs if not r["dis"]]; err = sum(1 - r["c"] for r in rs)
        return dict(share=100 * len(d) / len(rs), agree=100 * ST.mean(r["c"] for r in a), dis_llm=100 * ST.mean(r["c"] for r in d), dis_stat=100 * ST.mean(r["s"] for r in d), err_captured=100 * sum(1 - r["c"] for r in d) / err)
    s = summ(rows); print("  disagreement", {k: round(v, 1) for k, v in s.items()})
    for m in A: print(f"    {m}", {k: round(v, 1) for k, v in summ([r for r in rows if r["m"] == m]).items()})
    by = defaultdict(list)
    for r in rows: by[r["pid"]].append(r)
    keys = sorted(by); rng = random.Random(0); bs = {k: [] for k in s}; gap = []
    for _ in range(10000):
        rs = [r for _k in range(len(keys)) for r in by[keys[rng.randrange(len(keys))]]]
        x = summ(rs)
        for k in x: bs[k].append(x[k])
        gap.append(x["agree"] - x["dis_llm"])
    ci = lambda v: (sorted(v)[249], sorted(v)[9749])
    print("  95% intervals", {k: tuple(round(c, 1) for c in ci(v)) for k, v in bs.items()}, "agree - disagree LLM", round(s["agree"] - s["dis_llm"], 1), tuple(round(c, 1) for c in ci(gap)))
    res[f"disagreement_{B}"] = dict(point=s, intervals={k: ci(v) for k, v in bs.items()}, gap=s["agree"] - s["dis_llm"], gap_interval=ci(gap))
for m, v in A.items():
    for B, d in v.items():
        f = R / d / "api_usage.json"
        if f.exists():
            u = json.load(open(f)); n = sum(1 for _ in open(R / d / "episodes.jsonl"))
            print(f"{m} B={B}: per interview calls {u['calls']/n:.1f}, input tokens {u['tin']/n:.0f}, output tokens {u['tout']/n:.0f}")
        else:
            print(f"{m} B={B}: no api_usage.json")
json.dump(res, open(R / "implications.json", "w"), indent=1)
