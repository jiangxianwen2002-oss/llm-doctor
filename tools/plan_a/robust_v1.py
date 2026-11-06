#!/usr/bin/env python3
"""Statistical robustness of every comparison the paper draws a conclusion from.
For each: difference, 95% interval, p, MDE (80% power), and the same difference and interval without the 6 Ebola patients.
Per-patient means; patient-paired bootstrap, 10k resamples, seed 0, sorted keys (same index sequence as the other plan_a scripts).
Read-only on the logs; run from the project root. Writes tools/plan_a/robust_v1.json and prints a table."""
import json, os, random, statistics as ST, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
OUTDIR = Path(__file__).resolve().parent
R = Path("t1_premise").resolve(); sys.path.insert(0, str(R / "r98")); os.chdir(R / "r98")
import t1_run as T, ddx_signal as D  # noqa: E401
pts, menu, conds = T.load(); s2 = {int(k): v for k, v in json.load(open("data/s2_menu.json")).items()}
st = D.Stats.load(str(R / "sig86/sig_stats.npz"), conds); kids = json.load(open(R / "sig86/kid_fresh.json")); P = {str(p["pid"]): p for p in pts}
ALL = sorted(P); KEEP = [k for k in ALL if "ebola" not in P[k]["diagnosis"].lower()]


def readout(pid, h):
    p = P[pid]; return float(conds[int(np.argmax(st.posterior(int(p["initial"]), h, kids[pid])))] == p["diagnosis"])


def policy(B):
    D.MAXQ, D.TAU = B, .9; a, q = {}, {}
    for pid, p in P.items():
        q0 = int(p["initial"]); h = []
        while True:
            pi = st.posterior(q0, h, kids[pid]); k, v = st.recommend(pi, q0, s2[q0], h)
            if k == "dx": break
            h.append(s2[q0][v[0]])
        a[pid] = float(conds[v] == p["diagnosis"]); q[pid] = float(len(h))
    D.MAXQ, D.TAU = 10, .9
    return a, q


CACHE = {}
def eig_same_n(pid, n):
    if (pid, n) not in CACHE:
        p = P[pid]; q0 = int(p["initial"]); h = []; D.MAXQ, D.TAU = 10**6, 2.0
        while len(h) < n:
            pi = st.posterior(q0, h, kids[pid]); k, v = st.recommend(pi, q0, s2[q0], h)
            if k == "dx" or not v: break
            h.append(s2[q0][v[0]])
        D.MAXQ, D.TAU = 10, .9
        CACHE[(pid, n)] = readout(pid, h)
    return CACHE[(pid, n)]


LOADED = {}
def load(d):
    if d not in LOADED: LOADED[d] = [json.loads(l) for l in open(R / d / "episodes.jsonl")]
    return LOADED[d]
def pm(E, f):
    per = defaultdict(list)
    for e in E: per[str(e["pid"])].append(f(e))
    return {k: sum(v) / len(v) for k, v in per.items()}
def acc(d): return pm(load(d), lambda e: float(e["correct"]))
def nq(d): return pm(load(d), lambda e: float(e["n_questions"]))
def mean(xs): return {k: ST.mean(x[k] for x in xs) for k in xs[0]}
def rnd(d):
    E = load(d)
    def one(e, s):
        pid = str(e["pid"]); q0 = int(P[pid]["initial"]); cand = [q for q in s2[q0] if q != q0]
        return readout(pid, random.Random(f"{s}|{pid}|{e['seed']}").sample(cand, min(e["n_questions"], len(cand))))
    return mean([pm(E, lambda e, s=s: one(e, s)) for s in range(5)])


IDX = {}
def idx(n):
    if n not in IDX:
        rng = random.Random(0); IDX[n] = np.array([[rng.randrange(n) for _ in range(n)] for _ in range(10000)])
    return IDX[n]


def boot_d(d, keys):
    """d: per-patient difference in final units. Returns mean, lo, hi, p, se."""
    keys = sorted(keys); v = np.array([d[k] for k in keys]); bs = np.sort(v[idx(len(v))].mean(axis=1))
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return dict(d=float(v.mean()), lo=float(bs[249]), hi=float(bs[9749]), p=float(p), se=float(bs.std(ddof=1)))


ROWS = []
def add(group, label, d, unit="pts", note=""):
    a, b = boot_d(d, ALL), boot_d(d, KEEP)
    ROWS.append(dict(group=group, label=label, unit=unit, note=note, all=a, without=b))
def diff(x, y, s=100.0): return {k: s * (x[k] - y[k]) for k in x}


# ---------------- runs ----------------
T10, Q10 = policy(10); T15, Q15 = policy(15)
RUN = {"Qwen 4B": dict(n10="r100/q_none_b10_100", n15="r86/ddx86/untrained86_fresh", s10="r95/sig0_b10_95", s15="r86/ddx86/sig0_86_fresh"),
       "Gemma 4B": dict(n10="r100/g_none_b10_100", n15="r94/gemma_untrained94", s10="r95/gemma_sig0_b10_95", s15="r94/gemma_sig0_94"),
       "DS-V4.1-Flash": dict(n10="r97/ds_none_b10_97", n15="r96/ds_none96", s10="r97/ds_sig0_b10_97", s15="r96/ds_sig0_96"),
       "Qwen3.8-Flash": dict(n10="r98/f_none_b10_98", n15="r98/f_none_b15_98", s10="r98/f_sig0_b10_98", s15="r98/f_sig0_b15_98"),
       "GLM-5.3-Flash": dict(n10="r99/g_none_b10_99", n15="r99/g_none_b15_99", s10="r99/g_sig0_b10_99", s15="r99/g_sig0_b15_99")}
SMALL, LARGE = ("Qwen 4B", "Gemma 4B"), ("DS-V4.1-Flash", "Qwen3.8-Flash", "GLM-5.3-Flash")
MODELS = list(RUN)
A = {m: {k: acc(d) for k, d in v.items()} for m, v in RUN.items()}
NQ = {m: {k: nq(v[k]) for k in ("s10", "s15")} for m, v in RUN.items()}
for name, g in (("Small mean", SMALL), ("Larger mean", LARGE)):
    A[name] = {k: mean([A[m][k] for m in g]) for k in ("n10", "n15", "s10", "s15")}
    NQ[name] = {k: mean([NQ[m][k] for m in g]) for k in ("s10", "s15")}

# RQ1: statistician reading the LLM's answers (no advice, B=15)
RO = {m: pm(load(RUN[m]["n15"]), lambda e: readout(str(e["pid"]), e["asked"])) for m in MODELS}
RA = {m: rnd(RUN[m]["n15"]) for m in MODELS}
EQ = {m: pm(load(RUN[m]["n15"]), lambda e: eig_same_n(str(e["pid"]), e["n_questions"])) for m in LARGE}
G = "RQ1: LLM's questions, read by the statistician ($B=15$, no advice)"
for m in MODELS: add(G + " | vs. random questions", m, diff(RO[m], RA[m]))
for m in LARGE: add(G + " | vs. the statistician's own questions", m, diff(RO[m], EQ[m]))
add(G + " | vs. the statistician's own questions", "Larger mean", diff(mean([RO[m] for m in LARGE]), mean([EQ[m] for m in LARGE])))
for m in MODELS: add(G + " | vs. the LLM's own diagnosis", m, diff(RO[m], A[m]["n15"]))

# RQ2
G = "RQ2"
for b in (10, 15):
    for m in MODELS: add(f"RQ2: advice $-$ no advice | $B={b}$", m, diff(A[m][f"s{b}"], A[m][f"n{b}"]))
TB = {10: T10, 15: T15}; QB = {10: Q10, 15: Q15}
for b in (10, 15):
    for m in MODELS + ["Small mean", "Larger mean"]:
        add(f"RQ2: LLM with advice $-$ statistician, same budget | $B={b}$", m, diff(A[m][f"s{b}"], TB[b]))
SFTA, SFTN = acc("r86/ddx86/sigt86_fresh"), acc("r86/ddx86/b2r86_fresh")
add("RQ2: fine-tuned Qwen 4B | ", "SFT w/ advice $-$ statistician ($B=10$)", diff(SFTA, T10))
add("RQ2: fine-tuned Qwen 4B | ", "SFT w/ advice $-$ SFT w/o advice", diff(SFTA, SFTN))
add("RQ2: questions asked, LLM with advice $-$ statistician | $B=15$", "Larger mean", diff(NQ["Larger mean"]["s15"], QB[15], 1.0), unit="questions")

SL = {}
for m in MODELS:
    for b in (10, 15): SL[m, b] = pm(load(RUN[m][f"s{b}"]), lambda e: eig_same_n(str(e["pid"]), e["n_questions"]))
for b in (10, 15):
    SL["Small mean", b] = mean([SL[m, b] for m in SMALL]); SL["Larger mean", b] = mean([SL[m, b] for m in LARGE])
for b in (10, 15):
    for m in MODELS + ["Small mean", "Larger mean"]:
        add(f"RQ2: LLM with advice $-$ statistician with as many questions (Figure 3) | $B={b}$", m, diff(A[m][f"s{b}"], SL[m, b]))

# statistician's own stopping rule, interpolated at the LLM's mean number of questions
C = {b: policy(b) for b in range(2, 16)}; QM = {b: ST.mean(C[b][1].values()) for b in C}
def curve(qbar):
    bs = sorted(C)
    if qbar >= QM[bs[-1]]: return dict(C[bs[-1]][0])
    lo = max(b for b in bs if QM[b] <= qbar); hi = min(b for b in bs if QM[b] >= qbar)
    w = 0 if hi == lo else (qbar - QM[lo]) / (QM[hi] - QM[lo])
    return {k: (1 - w) * C[lo][0][k] + w * C[hi][0][k] for k in P}
CV = {}
for m in LARGE:
    for b in (10, 15):
        CV[m, b] = diff(A[m][f"s{b}"], curve(ST.mean(NQ[m][f"s{b}"].values())))
for b in (10, 15):
    G = f"RQ2: LLM with advice $-$ statistician's own stopping rule at the same mean number of questions | $B={b}$"
    for m in LARGE: add(G, m, CV[m, b])
    add(G, "Larger mean", mean([CV[m, b] for m in LARGE]))

# RQ3 (B=15)
W = {"Qwen 4B": ("r97/q_dxonly97", "r97/q_qonly97", "r97/q_contra97"), "Qwen 4B, fine-tuned": ("r97/sft_dxonly97", "r97/sft_qonly97", "r97/sft_contra97"),
     "DS-V4.1-Flash": ("r97/ds_dxonly97", "r97/ds_qonly97", "r97/ds_contra97"), "Qwen3.8-Flash": ("r98/f_dxonly_b15_98", "r98/f_qonly_b15_98", "r98/f_contra_b15_98"),
     "GLM-5.3-Flash": ("r99/g_dxonly_b15_99", "r99/g_qonly_b15_99", "r99/g_contra_b15_99")}
WA = {m: [acc(d) for d in v] for m, v in W.items()}
WA["Larger mean"] = [mean([WA[m][i] for m in LARGE]) for i in range(3)]
CORR = {m: A[m]["s15"] for m in MODELS + ["Larger mean"]}; CORR["Qwen 4B, fine-tuned"] = SFTA
NONE = {m: A[m]["n15"] for m in MODELS + ["Larger mean"]}
WM = ["Qwen 4B", "Qwen 4B, fine-tuned", "DS-V4.1-Flash", "Qwen3.8-Flash", "GLM-5.3-Flash", "Larger mean"]
# all-wrong advice is compared with no advice (the paper's point: wrong statistics fall below no advice), not with correct advice
for i, n in ((0, "wrong diagnosis"), (1, "wrong questions")):
    for m in WM: add(f"RQ3: {n} $-$ correct advice | $B=15$", m, diff(WA[m][i], CORR[m]))
for i, n in ((0, "wrong diagnosis"), (1, "wrong questions"), (2, "all wrong")):
    for m in ["Qwen 4B", "DS-V4.1-Flash", "Qwen3.8-Flash", "GLM-5.3-Flash", "Larger mean"]:
        add(f"RQ3: {n} $-$ no advice | $B=15$", m, diff(WA[m][i], NONE[m]))

# Section 4: disagreement with the statistician as a review signal (larger models, correct advice)
DIS = {}
for b in (10, 15):
    rows = []
    for m in LARGE:
        for e in load(RUN[m][f"s{b}"]):
            pid = str(e["pid"]); p = P[pid]; nb = conds[int(np.argmax(st.posterior(int(p["initial"]), e["asked"], kids[pid])))]
            rows.append(dict(pid=pid, c=float(e["correct"]), dis=e["dx"] != nb))
    by = defaultdict(list)
    for r in rows: by[r["pid"]].append(r)
    def stats(keys_sample):
        rs = [r for k in keys_sample for r in by[k]]; a = [r["c"] for r in rs if not r["dis"]]; d = [r["c"] for r in rs if r["dis"]]
        err = sum(1 - r["c"] for r in rs)
        return 100 * (ST.mean(a) - ST.mean(d)), 100 * sum(1 - c for c in d) / err
    for lab, keys in (("all", ALL), ("without", KEEP)):
        keys = sorted(keys); I = idx(len(keys)); pt = stats(keys)
        bs = np.array([stats([keys[j] for j in row]) for row in I])
        out = {}
        for j, name in ((0, "gap"), (1, "captured")):
            s = np.sort(bs[:, j]); out[name] = dict(d=pt[j], lo=float(s[249]), hi=float(s[9749]),
                                                     p=float(min(1.0, 2 * min((s <= 0).mean(), (s >= 0).mean()))), se=float(s.std(ddof=1)))
        DIS[b, lab] = out
for b in (10, 15):
    ROWS.append(dict(group="Section 4: larger models, accuracy when agreeing $-$ when disagreeing with the statistician | ", label=f"$B={b}$", unit="pts", note="",
                     all=DIS[b, "all"]["gap"], without=DIS[b, "without"]["gap"]))
for b in (10, 15):
    ROWS.append(dict(group="Section 4: share of all errors in disagreeing interviews (\\%) | ", label=f"$B={b}$", unit="pct", note="",
                     all=DIS[b, "all"]["captured"], without=DIS[b, "without"]["captured"]))

meta = dict(n_all=len(ALL), n_without=len(KEEP), dropped=sorted(set(P[k]["diagnosis"] for k in ALL if k not in KEEP)),
            stat=dict(b10=100 * ST.mean(T10.values()), b15=100 * ST.mean(T15.values()),
                      b10_without=100 * ST.mean(T10[k] for k in KEEP), b15_without=100 * ST.mean(T15[k] for k in KEEP)))
json.dump(dict(meta=meta, rows=ROWS), open(OUTDIR / "robust_v1.json", "w"), indent=1)
mx = max(abs(r["without"]["d"] - r["all"]["d"]) for r in ROWS)
for r in ROWS:
    a, b = r["all"], r["without"]; sa, sb = a["p"] < .05, b["p"] < .05
    print(f"{r['group'][:60]:60s} {r['label'][:22]:22s} {a['d']:+6.1f} [{a['lo']:+6.1f},{a['hi']:+6.1f}] p={a['p']:.3f} MDE={2.8*a['se']:4.1f} | "
          f"{b['d']:+6.1f} [{b['lo']:+6.1f},{b['hi']:+6.1f}]{'  FLIP' if sa != sb else ''}")
print(f"{len(ROWS)} rows; largest change without Ebola {mx:.2f}; statistician", {k: round(v, 1) for k, v in meta["stat"].items()})
