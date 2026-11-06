#!/usr/bin/env python3
"""Recompute every number in the paper from the raw interview logs and compare with what the tables, figures and text print.
The paper's LaTeX sources are read from paper/ (not included).
Read-only; run from the project root. Prints each mismatch, then a summary."""
import json, os, re, statistics as ST, sys, random
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
ROOT = Path(".").resolve(); PV = ROOT / "paper"
R = Path("t1_premise").resolve(); sys.path.insert(0, str(R / "r98")); os.chdir(R / "r98")
import t1_run as T, ddx_signal as D, ddx_bad97 as B97, ddx_bad97q  # noqa
pts, menu, conds = T.load(); s2 = {int(k): v for k, v in json.load(open("data/s2_menu.json")).items()}
st = D.Stats.load(str(R / "sig86/sig_stats.npz"), conds); kids = json.load(open(R / "sig86/kid_fresh.json")); P = {str(p["pid"]): p for p in pts}
def rd(pid, h):
    p = P[pid]; return float(conds[int(np.argmax(st.posterior(int(p["initial"]), h, kids[pid])))] == p["diagnosis"])
def policy(B):
    D.MAXQ, D.TAU = B, .9; a = {}; q = {}
    for pid, p in P.items():
        q0 = int(p["initial"]); h = []
        while True:
            pi = st.posterior(q0, h, kids[pid]); k, v = st.recommend(pi, q0, s2[q0], h)
            if k == "dx": break
            h.append(s2[q0][v[0]])
        a[pid] = float(conds[v] == p["diagnosis"]); q[pid] = float(len(h))
    D.MAXQ, D.TAU = 10, .9
    return a, q
C = {}
def eig(pid, n):
    if (pid, n) not in C:
        p = P[pid]; q0 = int(p["initial"]); h = []; D.MAXQ, D.TAU = 99, 2.0
        while len(h) < n:
            pi = st.posterior(q0, h, kids[pid]); k, v = st.recommend(pi, q0, s2[q0], h)
            if k == "dx": break
            h.append(s2[q0][v[0]])
        D.MAXQ, D.TAU = 10, .9; C[pid, n] = rd(pid, h)
    return C[pid, n]
def E(d): return [json.loads(l) for l in open(R / d / "episodes.jsonl")]
def pm(es, f):
    per = defaultdict(list)
    for e in es: per[str(e["pid"])].append(f(e))
    return {k: ST.mean(v) for k, v in per.items()}
def mean(xs): return {k: ST.mean(x[k] for x in xs) for k in xs[0]}
def M(x, c=100): return c * ST.mean(x.values())
ARM = {"Qwen 4B": dict(n10="r100/q_none_b10_100", s10="r95/sig0_b10_95", n15="r86/ddx86/untrained86_fresh", s15="r86/ddx86/sig0_86_fresh", dx="r97/q_dxonly97", q="r97/q_qonly97", all="r97/q_contra97"),
       "Gemma 4B": dict(n10="r100/g_none_b10_100", s10="r95/gemma_sig0_b10_95", n15="r94/gemma_untrained94", s15="r94/gemma_sig0_94"),
       "DS-V4.1-Flash": dict(n10="r97/ds_none_b10_97", s10="r97/ds_sig0_b10_97", n15="r96/ds_none96", s15="r96/ds_sig0_96", dx="r97/ds_dxonly97", q="r97/ds_qonly97", all="r97/ds_contra97"),
       "Qwen3.8-Flash": dict(n10="r98/f_none_b10_98", s10="r98/f_sig0_b10_98", n15="r98/f_none_b15_98", s15="r98/f_sig0_b15_98", dx="r98/f_dxonly_b15_98", q="r98/f_qonly_b15_98", all="r98/f_contra_b15_98"),
       "GLM-5.3-Flash": dict(n10="r99/g_none_b10_99", s10="r99/g_sig0_b10_99", n15="r99/g_none_b15_99", s15="r99/g_sig0_b15_99", dx="r99/g_dxonly_b15_99", q="r99/g_qonly_b15_99", all="r99/g_contra_b15_99")}
SM = ("Qwen 4B", "Gemma 4B"); LG = ("DS-V4.1-Flash", "Qwen3.8-Flash", "GLM-5.3-Flash")
acc = {m: {k: pm(E(d), lambda e: float(e["correct"])) for k, d in v.items()} for m, v in ARM.items()}
qs = {m: {k: pm(E(v[k]), lambda e: float(e["n_questions"])) for k in ("s10", "s15")} for m, v in ARM.items()}
for name, g in (("Small mean", SM), ("Larger mean", LG)):
    acc[name] = {k: mean([acc[m][k] for m in g]) for k in ("n10", "s10", "n15", "s15")}
    qs[name] = {k: mean([qs[m][k] for m in g]) for k in ("s10", "s15")}
acc["Larger mean"].update({k: mean([acc[m][k] for m in LG]) for k in ("dx", "q", "all")})
sft = {"s15": pm(E("r86/ddx86/sigt86_fresh"), lambda e: float(e["correct"])), "dx": pm(E("r97/sft_dxonly97"), lambda e: float(e["correct"])), "q": pm(E("r97/sft_qonly97"), lambda e: float(e["correct"])), "all": pm(E("r97/sft_contra97"), lambda e: float(e["correct"]))}
T10, Q10 = policy(10); T15, Q15 = policy(15)
bad = []; n_ok = 0
def chk(label, printed, true, d=1):
    global n_ok
    if abs(float(printed) - round(true, d)) > 10 ** (-d) * 0.51: bad.append(f"{label}: printed {printed}, data {true:.3f}")
    else: n_ok += 1
# Table 1
t = (PV / "sections/rq2.tex").read_text()
for line in t.splitlines():
    m = re.match(r"\s*(?:\\rowcolor\{[^}]*\}\s*)?([A-Za-z].*?)\s*&\s*(.*?)\\\\", line)
    if not m or m.group(1) == "Model": continue
    name = m.group(1); cells = [re.sub(r"\\textbf\{|\}", "", c).strip() for c in m.group(2).split("&")]
    if name == "Statistician":
        chk("T1 stat acc10", cells[1], M(T10)); chk("T1 stat Q10", cells[2], M(Q10, 1)); chk("T1 stat acc15", cells[4], M(T15)); chk("T1 stat Q15", cells[5], M(Q15, 1)); continue
    if name in acc:
        for i, k in ((0, "n10"), (1, "s10"), (3, "n15"), (4, "s15")): chk(f"T1 {name} {k}", cells[i], M(acc[name][k]))
        chk(f"T1 {name} Q10", cells[2], M(qs[name]["s10"], 1)); chk(f"T1 {name} Q15", cells[5], M(qs[name]["s15"], 1))
# Table 2
t = (PV / "sections/rq3.tex").read_text()
for line in t.splitlines():
    m = re.match(r"\s*(?:\\rowcolor\{[^}]*\}\s*)?([A-Za-z].*?)\s*&\s*([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\", line)
    if not m: continue
    name = m.group(1); src = sft if "fine-tuned" in name else acc[name]
    for i, k in ((2, "s15"), (3, "dx"), (4, "q"), (5, "all")): chk(f"T2 {name} {k}", m.group(i), M(src[k]))
# Figure A and B
fa = (PV / "figures/figA.tex").read_text(); own = {}; rdA = {}; rnA = {}
def rnd_(e, s):
    pid = str(e["pid"]); q0 = int(P[pid]["initial"]); cand = [q for q in s2[q0] if q != q0]
    return rd(pid, random.Random(f"{s}|{pid}|{e['seed']}").sample(cand, min(e["n_questions"], len(cand))))
for m in re.finditer(r"([\-\d.]+)/([A-Za-z][^/]*)/([\d.]+)/([\d.]+)/([\d.]+)", fa):
    name = m.group(2); es = E(ARM[name]["n15"]); chk(f"FA {name} own", m.group(3), M(acc[name]["n15"]))
    r = mean([pm(es, lambda e, s=s: rnd_(e, s)) for s in range(5)]); ro = pm(es, lambda e: rd(str(e["pid"]), e["asked"]))
    chk(f"FA {name} random", m.group(4), M(r)); chk(f"FA {name} LLM questions", m.group(5), M(ro)); rnA[name] = r; rdA[name] = ro
fb = (PV / "figures/figB.tex").read_text(); SL = {}
for m in ARM:
    for B, k in ((10, "s10"), (15, "s15")): SL[m, B] = pm(E(ARM[m][k]), lambda e: eig(str(e["pid"]), e["n_questions"]))
for B in (10, 15):
    SL["Small mean", B] = mean([SL[m, B] for m in SM]); SL["Larger mean", B] = mean([SL[m, B] for m in LG])
for m in re.finditer(r"[\-\d.]+/([A-Za-z][^/]*)/(-?[\d.]+)/(-?[\d.]+)/\+?\$?-?\$?[\d.]+/", fb):
    name = m.group(1)
    for B, k, g in ((10, "s10", m.group(2)), (15, "s15", m.group(3))): chk(f"FB {name} B{B}", g, M(acc[name][k]) - M(SL[name, B]))
# Text
def rng(vals): return min(vals), max(vals)
lifts = [M(acc[m]["s10"]) - M(acc[m]["n10"]) for m in ARM] + [M(acc[m]["s15"]) - M(acc[m]["n15"]) for m in ARM]
print("text checks (data value -> paper):")
rows = [("lift range", "%.1f to %.1f" % rng(lifts), "26 to 51"),
 ("LLM questions - random", "%.1f to %.1f" % rng([M(rdA[m]) - M(rnA[m]) for m in ARM]), "5.0 to 8.4"),
 ("statistician - LLM own", "%.1f to %.1f" % rng([M(rdA[m]) - M(acc[m]["n15"]) for m in ARM]), "23.0 to 51.8"),
 ("Larger B10 - stat", "%.2f" % (M(acc["Larger mean"]["s10"]) - M(T10)), "4.9"),
 ("Larger Q difference B15", "%.2f" % (M(Q15, 1) - M(qs["Larger mean"]["s15"], 1)), "1.38"),
 ("QF B15 acc / Q / same-number", "%.1f / %.2f / %.1f" % (M(acc["Qwen3.8-Flash"]["s15"]), M(qs["Qwen3.8-Flash"]["s15"], 1), M(SL["Qwen3.8-Flash", 15])), "79.0 / 9.4 / 70.8"),
 ("Larger same-number gain", "%.1f, %.1f" % (M(acc["Larger mean"]["s10"]) - M(SL["Larger mean", 10]), M(acc["Larger mean"]["s15"]) - M(SL["Larger mean", 15])), "about 6"),
 ("per 1,000", "%.0f, %.0f" % (10 * (M(acc["Larger mean"]["s10"]) - M(T10)), 10 * (M(acc["Qwen3.8-Flash"]["s10"]) - M(T10))), "49, 73"),
 ("Larger correct - wrong", "%.1f, %.1f" % (M(acc["Larger mean"]["s15"]) - M(acc["Larger mean"]["dx"]), M(acc["Larger mean"]["s15"]) - M(acc["Larger mean"]["q"])), "within 5"),
 ("Larger all wrong", "%.1f" % M(acc["Larger mean"]["all"]), "29.7"),
 ("patients / labeled / conditions / questions / menu", f"{len(P)} / see sig86 / {len(conds)} / {len(menu)} / {sorted(set(len(v) for v in s2.values()))}", "300 / 235 / 49 / 223 / 50"),
 ("Ebola patients", str(sum('ebola' in p['diagnosis'].lower() for p in P.values())), "6 of 300 = 2%")]
for r in rows: print("  %-48s %-30s %s" % r)
print(f"\ntables and figures: {n_ok} cells match, {len(bad)} mismatches")
for b in bad: print("  MISMATCH", b)
