#!/usr/bin/env python3
"""Typeset the robustness note (robustness_note.pdf) from tools/plan_a/robust_v1.json.
Every number in the PDF is read from robust_v1.json (written by tools/plan_a/robust_v1.py from the raw logs).
Layout as the earlier note: summary box, forest plots, the full table, and how each quantity is computed.
  python tools/plan_a/make_robust_v1.py   (from the project root; needs tectonic)"""
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
J = json.load(open(ROOT / "tools/plan_a/robust_v1.json"))
OUT = ROOT / "robustness_note"
ROWS, META = J["rows"], J["meta"]


def find(group_start, label, budget=None):
    hits = [r for r in ROWS if r["group"].startswith(group_start) and r["label"] == label and (budget is None or r["group"].endswith(f"$B={budget}$"))]
    assert len(hits) == 1, (group_start, label, budget, len(hits))
    return hits[0]


sig = lambda s: s["p"] < 0.05
f1 = lambda x, n=1: f"{x:+.{n}f}".replace("-", "$-$").replace("+", "$+$")
ci = lambda s, n=1: f"[{f1(s['lo'], n)}, {f1(s['hi'], n)}]"
pv = lambda p: r"$<$0.001" if p < 0.001 else f"{p:.3f}"
LARGE = ["DS-V4.1-Flash", "Qwen3.8-Flash", "GLM-5.3-Flash"]
FIVE = ["Qwen 4B", "Gemma 4B"] + LARGE

# ---------------- conclusions the paper draws, checked on all patients and without Ebola ----------------
CLAIMS = [(r, True) for r in ROWS if r["group"].endswith("vs. random questions")]
CLAIMS += [(find("RQ1: LLM's questions", "Larger mean"), False)]
CLAIMS += [(r, True) for r in ROWS if r["group"].startswith("RQ2: advice $-$ no advice")]
CLAIMS += [(find("RQ2: LLM with advice $-$ statistician, same budget", "Larger mean", 10), True),
           (find("RQ2: LLM with advice $-$ statistician, same budget", "Larger mean", 15), False),
           (find("RQ2: fine-tuned", "SFT w/ advice $-$ statistician ($B=10$)"), False),
           (find("RQ2: fine-tuned", "SFT w/ advice $-$ SFT w/o advice"), True),
           (find("RQ2: questions asked", "Larger mean"), True)]
for b in (10, 15):
    CLAIMS += [(find("RQ2: LLM with advice $-$ statistician with as many", "Larger mean", b), True),
               (find("RQ2: LLM with advice $-$ statistician's own stopping rule", "Larger mean", b), True)]
CLAIMS += [(r, True) for r in ROWS if r["group"].startswith("Section 4: larger models, accuracy")]
for r, want in CLAIMS:
    for k in ("all", "without"):
        assert sig(r[k]) == want, ("conclusion changes", r["group"], r["label"], k, r[k])
# "four of the five models match or exceed the statistician at both budgets": only Gemma is significantly below
for b in (10, 15):
    for m in FIVE:
        s = find("RQ2: LLM with advice $-$ statistician, same budget", m, b)
        for k in ("all", "without"):
            assert (m == "Gemma 4B") == (sig(s[k]) and s[k]["d"] < 0), (m, b, k)

# ---------------- full table ----------------
body, last_head = [], None
for r in ROWS:
    head, sub = [x.strip() for x in r["group"].split("|")]
    if head != last_head:
        if last_head is not None: body.append(r"\midrule")
        body.append(r"\multicolumn{7}{l}{\textbf{" + head + r"}} \\")
        last_head, last_sub = head, None
    if sub and sub != last_sub:
        body.append(r"\multicolumn{7}{l}{\quad\emph{" + sub + r"}} \\")
        last_sub = sub
    a, w = r["all"], r["without"]
    n = 2 if r["unit"] == "questions" else 1   # 11.74 - 10.36; the paper's 1.3 is 11.7 - 10.4 from Table 1
    d = r"\textbf{" + f1(a["d"], n) + "}" if sig(a) else f1(a["d"], n)
    dw = r"\textbf{" + f1(w["d"], n) + "}" if sig(w) else f1(w["d"], n)
    if r["unit"] == "pct":   # a share, not a difference: no test
        d, pcol, mde, dw = f"{a['d']:.1f}", "--", "--", f"{w['d']:.1f}"
    else:
        pcol, mde = pv(a["p"]), f"{2.8 * a['se']:.{n}f}"
    unit = {"questions": r"\textsuperscript{q}", "pct": r"\textsuperscript{\%}"}.get(r["unit"], "")
    cis = (ci(a, n), ci(w, n)) if r["unit"] != "pct" else (ci(a, n).replace("$+$", ""), ci(w, n).replace("$+$", ""))
    body.append(r"\qquad " + " & ".join([r["label"] + unit, d, cis[0], pcol, mde, dw, cis[1]]) + r" \\")


# ---------------- forest plots ----------------
def forest(sections, xmin, xmax, height, xlabel, legend):
    """sections: [(title, [(label, [(stats, colour), ...])])]; several dots per row are spread vertically."""
    y, ys, labels, plots, seps, titles = 0, [], [], [], [], []
    for title, items in sections:
        y -= 1; titles.append((title, y))   # each section gets its own title row
        for lab, dots in items:
            y -= 1; ys.append(y); labels.append(lab)
            off = [0] if len(dots) == 1 else [0.18, -0.18]
            for (s, col), o in zip(dots, off):
                shape = "square" if col == "b15c" else ""
                mk = (shape or "") + "*" if sig(s) else ((shape or "o") + ", fill=white")
                plots.append(rf"\addplot[only marks, mark={mk}, mark size=2.1pt, mark options={{line width=0.8pt}}, color={col}, error bars/.cd, x dir=both, x explicit, "
                             rf"error bar style={{line width=1.1pt, {col}}}] coordinates {{({s['d']:.2f},{y + o:.2f}) -= ({s['d'] - s['lo']:.2f},0) += ({s['hi'] - s['d']:.2f},0)}};")
        seps.append(y - 0.5)
    ymin = ys[-1] - 0.7
    sepl = "\n".join(rf"\draw[black!20] (axis cs:{xmin},{s}) -- (axis cs:{xmax},{s});" for s in seps[:-1])
    tl = "\n".join(rf"\node[anchor=west, font=\footnotesize\bfseries, text=black!80, fill=white, inner sep=1pt] at (axis cs:{xmin},{yy}) {{{t}}};" for t, yy in titles)
    return "\n".join([
        r"\begin{tikzpicture}",
        rf"\begin{{axis}}[width=0.86\linewidth, height={height}, xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax=-0.3,",
        rf"  ytick={{{','.join(map(str, ys))}}}, yticklabels={{{','.join('{' + l + '}' for l in labels)}}},",
        r"  yticklabel style={font=\small}, xticklabel style={font=\small}, xlabel style={font=\small},",
        rf"  xlabel={{{xlabel}}}, axis y line*=left, axis x line*=bottom, tick align=outside, xmajorgrids, grid style={{black!8}}, clip=false]",
        rf"\draw[black!65, line width=0.9pt] (axis cs:0,{ymin}) -- (axis cs:0,-0.3);",
        sepl, tl, "\n".join(plots), r"\end{axis}",
        legend, r"\end{tikzpicture}"])


KEY = r"\node[anchor=south west, font=\footnotesize, align=left, draw=black!25, rounded corners=2pt, inner sep=3pt] at ([yshift=3pt]current axis.north west) {"
LEG2 = (KEY + r"\tikz\fill[llmc] (0,0) circle (2.2pt); $B=10$\quad \tikz\fill[b15c] (-2pt,-2pt) rectangle (2pt,2pt); $B=15$"
        r"\qquad filled: significant\quad open: not significant};")
LEG1 = KEY + r"\tikz\fill[llmc] (0,0) circle (2.2pt); significant\quad \tikz\draw[llmc, fill=white, line width=0.8pt] (0,0) circle (2.2pt); not significant};"

def two(group_start, models):
    return [(m, [(find(group_start, m, 10)["all"], "llmc"), (find(group_start, m, 15)["all"], "b15c")]) for m in models]

fig1 = forest([
    ("Same budget", two("RQ2: LLM with advice $-$ statistician, same budget", FIVE + ["Small mean", "Larger mean"])),
    ("As many questions (paper, Figure 3)", two("RQ2: LLM with advice $-$ statistician with as many", FIVE + ["Small mean", "Larger mean"])),
    ("Statistician's own stopping rule", two("RQ2: LLM with advice $-$ statistician's own stopping rule", LARGE + ["Larger mean"]))],
    -14, 16, "13.2cm", r"LLM with advice $-$ statistician (points)", LEG2)
fig2 = forest([
    ("LLM's questions $-$ random questions", [(m, [([r for r in ROWS if r["group"].endswith("vs. random questions") and r["label"] == m][0]["all"], "llmc")]) for m in FIVE]),
    ("LLM's questions $-$ statistician's own questions", [(m, [([r for r in ROWS if r["group"].endswith("own questions") and r["label"] == m][0]["all"], "llmc")]) for m in LARGE + ["Larger mean"]])],
    -8, 14, "6.6cm", r"Accuracy difference (points)", LEG1)
WM = ["Qwen 4B", "Qwen 4B, fine-tuned"] + LARGE + ["Larger mean"]
fig3 = forest([
    ("Wrong diagnosis", [(m, [(find("RQ3: wrong diagnosis $-$ correct", m)["all"], "llmc")]) for m in WM]),
    ("Wrong questions", [(m, [(find("RQ3: wrong questions $-$ correct", m)["all"], "llmc")]) for m in WM])],
    -22, 4, "7.4cm", r"Wrong advice $-$ correct advice (points)", LEG1)

# ---------------- summary numbers ----------------
rnd = [r["all"]["d"] for r in ROWS if r["group"].endswith("vs. random questions")]
own = find("RQ1: LLM's questions", "Larger mean")["all"]
g10 = find("RQ2: LLM with advice $-$ statistician, same budget", "Larger mean", 10)["all"]
sl = {b: find("RQ2: LLM with advice $-$ statistician with as many", "Larger mean", b)["all"] for b in (10, 15)}
cv = {b: find("RQ2: LLM with advice $-$ statistician's own stopping rule", "Larger mean", b)["all"] for b in (10, 15)}
per = [find("RQ2: LLM with advice $-$ statistician with as many", m, b)["all"] for m in LARGE for b in (10, 15)]
assert all(s["d"] > 0 for s in per)
wd, wq = find("RQ3: wrong diagnosis $-$ correct", "Larger mean")["all"], find("RQ3: wrong questions $-$ correct", "Larger mean")["all"]
nd, nqq = find("RQ3: wrong diagnosis $-$ no advice", "Larger mean")["all"], find("RQ3: wrong questions $-$ no advice", "Larger mean")["all"]
gap = {b: find("Section 4: larger models, accuracy", f"$B={b}$")["all"] for b in (10, 15)}
flips = [r for r in ROWS if sig(r["all"]) != sig(r["without"])]
assert all(r["group"].startswith("RQ2: LLM with advice $-$ statistician with as many") and r["label"] in LARGE for r in flips), flips
fl = [f"{r['label']} at {r['group'].split('|')[1].strip()}" for r in flips]
FLIPS = "the gain with as many questions of " + (" and of ".join(fl) if fl else "none")
shift = max(abs(r["without"]["d"] - r["all"]["d"]) for r in ROWS)
words = {4: "four", 5: "five", 6: "six"}
s = lambda x: f"{f1(x['d'])} {ci(x)}"

tex = r"""\documentclass[manuscript,nonacm]{acmart}
% same class and fonts as the paper (acmart), single column, no ACM front matter; generated by tools/plan_a/make_robust_v1.py
\usepackage{array,booktabs,longtable,pgfplots}
\pgfplotsset{compat=1.17}
\settopmatter{printacmref=false, printccs=false, printfolios=true}
\definecolor{llmc}{HTML}{0072B2}\definecolor{b15c}{HTML}{D55E00}
\setlength{\parskip}{0.35em}\setlength{\parindent}{0pt}
\title{Statistical robustness of the reported comparisons}
\subtitle{Supplementary note to \emph{What Do LLM Doctors Add to Their Statistics?}}
\author{}\authorsaddresses{}
\begin{document}
\maketitle

\noindent\begin{tikzpicture}
\node[fill=llmc!6, draw=llmc!40, rounded corners=4pt, inner sep=9pt, text width=0.95\linewidth, align=left] {
\textbf{\large Summary}\par\smallskip
We report every comparison the paper draws a conclusion from (@NROWS@ in all) with its 95\% confidence interval, and repeat each without the @NU@ patients whose condition has no labeled patients. All numbers are produced by \texttt{tools/plan\_a/robust\_v1.py} from the raw interview logs.
\begin{itemize}\setlength{\itemsep}{1pt}
\item \textbf{The LLM's questions are useful.} Read by the statistician, they beat the same number of random questions for all five models (@RLO@ to @RHI@ points, every interval above zero). The larger models' questions match the statistician's own: @OWN@.
\item \textbf{Larger models with advice exceed the statistician.} At the same budget ($B=10$): @G10@. With as many questions: @SL10@ at $B=10$ and @SL15@ at $B=15$. Against the statistician's own stopping rule at the same mean number of questions: @CV10@ and @CV15@. All six per-model gains with as many questions are positive, and @NSIG@ are significant on their own.
\item \textbf{A wrong recommendation costs the larger models little.} With the statistics shown correctly, a wrong diagnosis changes their accuracy by @WD@ and wrong questions by @WQ@, while both stay far above no advice (@ND@ and @NQ@ points).
\item \textbf{Disagreement is a useful review signal.} When the larger models disagree with the statistician, they are less often right than when they agree: the gap is @GAP10@ at $B=10$ and @GAP15@ at $B=15$.
\item \textbf{The conditions without labeled patients do not drive any conclusion:} excluding their @NU@ patients changes every difference by at most @SHIFT@ points, and every conclusion of the paper stays the same.
\end{itemize}};
\end{tikzpicture}

\section*{1\quad Results at a glance}
Each dot is a difference in accuracy and each bar its 95\% confidence interval; filled dots are significant ($p<0.05$) and open dots are not. Dots to the right of zero favour the first condition.

\noindent\begin{minipage}{\linewidth}\centering
@FIG1@
\par\small\textbf{Figure 1.} LLMs with advice against the statistician (paper, Table~1 and Figure~3): at the same budget, asking as many questions in each interview, and with the statistician's own stopping rule at the same mean number of questions.
\end{minipage}

\bigskip
\noindent\begin{minipage}{\linewidth}\centering
@FIG2@
\par\small\textbf{Figure 2.} What the LLM's questions are worth (paper, Figure~2; $B=15$, no advice). In every case the statistician makes the diagnosis; only the questions differ. Top: the questions the LLM chose against the same number of random questions; to the right of zero, the LLM's questions are better. Bottom: the questions the LLM chose against the same number of the statistician's own information-gain questions; near zero, they are equally good.
\end{minipage}

\bigskip
\noindent\begin{minipage}{\linewidth}\centering
@FIG3@
\par\small\textbf{Figure 3.} One wrong recommendation, with the statistics shown correctly (paper, Table~2; $B=15$).
\end{minipage}

\section*{2\quad Full results}
{\footnotesize
\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.0}
\begin{longtable}{@{}l r c r r r c@{}}
\toprule
 & & & & & \multicolumn{2}{c}{Without unlabeled conditions} \\
\cmidrule(lr){6-7}
Comparison & Diff. & 95\% CI & $p$ & MDE & Diff. & 95\% CI \\
\midrule
\endfirsthead
\toprule
Comparison & Diff. & 95\% CI & $p$ & MDE & Diff. & 95\% CI \\
\midrule
\endhead
\bottomrule
\endfoot
@BODY@
\end{longtable}}
{\footnotesize Differences in accuracy points unless marked, rounded as described in Section~3; \textsuperscript{q}: in questions; \textsuperscript{\%}: share of errors in percent. \textbf{Bold}: $p<0.05$. MDE: minimum detectable effect (80\% power). Means are over the models listed, per patient. Statistician accuracy: @S10@\% at $B=10$ and @S15@\% at $B=15$ (@W10@\% and @W15@\% without the unlabeled conditions).}

\section*{3\quad How each quantity is computed}
\begin{center}\small
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.21\linewidth}>{\raggedright\arraybackslash}p{0.31\linewidth}>{\raggedright\arraybackslash}p{0.42\linewidth}@{}}
\toprule
Quantity & What it tells & How it is computed \\
\midrule
Difference & How much more (or less) accurate one condition is, in points. & Per patient: accuracy averaged over that patient's interviews (8 for Qwen 4B and Gemma 4B, 2 for each larger model; the statistician is deterministic), minus the same for the other condition. Then the mean over the @NP@ evaluation patients. \\
95\% confidence interval & How far the difference could move with another sample of patients. & Patient-paired bootstrap: draw @NP@ patients with replacement, average their differences, repeat 10{,}000 times (seed 0); take the 2.5th and 97.5th percentiles. \\
$p$-value & Whether the difference could be zero. & Twice the share of the 10{,}000 bootstrap averages on the other side of zero. Significant if $p<0.05$, i.e.\ the interval excludes zero. Two methods match when the difference is not significant. \\
MDE & The smallest true difference the evaluation would detect in 80\% of repetitions. & $2.80\times\mathrm{SE}$, with SE the standard deviation of the bootstrap averages ($2.80=1.96+0.84$). \\
Statistician reading the LLM's answers & What the LLM's questions are worth, apart from its diagnosis. & The statistician's most likely condition given exactly the answers the LLM collected. Random questions: the same number drawn from the same menu, averaged over 5 seeds. Own questions: the statistician's information-gain questions, stopped after the same number. \\
As many questions & The comparison of the paper's Figure~3. & In each interview, the statistician asks exactly as many of its own questions as the LLM asked, ignoring its 0.9 threshold, and diagnoses the most likely condition. \\
Own stopping rule & The same comparison with the statistician's usual rule. & The statistician's accuracy at budgets 2--15 with its 0.9 threshold, linearly interpolated per patient at the LLM's mean number of questions. \\
Review signal & Whether disagreement marks interviews worth a second look. & Interviews of the larger models with advice, split by whether the LLM's diagnosis equals the statistician's most likely condition given the same answers; accuracy in agreeing minus disagreeing interviews. The bootstrap resamples patients with all their interviews. \\
Rounding & How the numbers here relate to those in the paper. & The paper prints accuracies and numbers of questions with one decimal, and every number in its text that can be computed from its tables and figures is computed from these printed values (e.g., 1.3 fewer questions $=11.7-10.4$ in Table~1; gains of 26.3 to 50.7 points from Table~1). Here every difference is computed from the unrounded per-patient values and then rounded, so it can differ by 0.1 from the difference of two printed values (e.g., Gemma 4B, advice $-$ no advice at $B=15$: $+$@GEM@ here, $72.5-21.8=50.7$ in the paper). The difference in questions is shown with two decimals ($11.74-10.36$). \\
Without unlabeled conditions & Whether the conditions with no labeled patients drive a result. & @NU@ of the @NP@ evaluation patients (@PCT@\%) have such a condition (@COND@); each comparison is repeated without them. \\
\bottomrule
\end{tabular}
\end{center}
\end{document}
"""
NU = META["n_all"] - META["n_without"]
rep = {"FIG1": fig1, "FIG2": fig2, "FIG3": fig3, "BODY": "\n".join(body), "NROWS": str(len(ROWS)), "NU": str(NU), "NP": str(META["n_all"]),
       "PCT": f"{100 * NU / META['n_all']:.0f}", "COND": ", ".join(META["dropped"]),
       "RLO": f"{min(rnd):.1f}", "RHI": f"{max(rnd):.1f}", "OWN": s(own), "G10": s(g10), "SL10": s(sl[10]), "SL15": s(sl[15]),
       "CV10": s(cv[10]), "CV15": s(cv[15]), "NSIG": words[sum(sig(x) for x in per)], "WD": s(wd), "WQ": s(wq),
       "ND": f1(nd["d"]), "NQ": f1(nqq["d"]), "GAP10": s(gap[10]), "GAP15": s(gap[15]), "SHIFT": f"{shift:.1f}", "GEM": f"{find('RQ2: advice $-$ no advice', 'Gemma 4B', 15)['all']['d']:.1f}",
       "S10": f"{META['stat']['b10']:.1f}", "S15": f"{META['stat']['b15']:.1f}", "W10": f"{META['stat']['b10_without']:.1f}", "W15": f"{META['stat']['b15_without']:.1f}"}
for k, v in rep.items():
    tex = tex.replace("@" + k + "@", v)
assert "@" not in tex.replace("@{}", ""), "unfilled placeholder"
OUT.with_suffix(".tex").write_text(tex)
r = subprocess.run(["tectonic", "-X", "compile", str(OUT.with_suffix(".tex"))], capture_output=True, text=True)
if r.returncode:
    print(r.stderr[-2500:]); raise SystemExit(1)
print("ok ->", OUT.with_suffix(".pdf"))
