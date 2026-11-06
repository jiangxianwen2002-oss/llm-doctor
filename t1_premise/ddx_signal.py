#!/usr/bin/env python3
"""Round 86 (DDXPlus menu format): experience-statistics SIGNAL line for the t1_run_s2 doctor prompt.
Statistics = the round-42 few-label bank (235 labelled validate patients, t1_premise/data_b2/labelled.json), exactly the EIG
teacher of tools/dir34_menu/make_b2_demos.py (answer vocabulary top-30 per question, Laplace smoothing, prior from the initial
evidence, stop at posterior >= 0.9 or 10 questions). The signal shows what that teacher would do now:
  "Experience statistics from 235 past confirmed patients: most likely now: A 0.62, B 0.21, C 0.07. Recommended next: ASK 12 (best), ASK 7, ASK 30."
  or "... Recommended next: DIAGNOSIS: A."
Server side needs only numpy + two precomputed files (no raw DDXPlus): sig_stats.npz (L[q, y, k]) and a per-patient answer-key
index file (kid[q] for every question), built on the Mac by build_ddx_signal.py.
"""
import json
import numpy as np

MAXQ, TAU, LABEL_N = 10, 0.9, 235
INSERT_BEFORES = ("MENU (questions not yet asked):", "You must now reply with: DIAGNOSIS: <condition>")


class Stats:
    def __init__(self, L, conds):
        self.L = L; self.logL = np.log(np.where(L > 0, L, 1)); self.conds = conds

    @classmethod
    def load(cls, npz_path, conds):
        return cls(np.load(npz_path)["L"], conds)

    def posterior(self, q0, hist, kid):
        logp = np.log(self.L[q0, :, kid[q0]])
        for q in hist:
            logp = logp + np.log(self.L[q, :, kid[q]])
        pi = np.exp(logp - logp.max()); return pi / pi.sum()

    def recommend(self, pi, q0, local, hist):
        """teacher's move: ('dx', y) or ('ask', [local indices ranked by EIG, best first])"""
        if pi.max() >= TAU or len(hist) >= MAXQ:
            return "dx", int(np.argmax(pi))
        pk = np.einsum("y,qyk->qk", pi, self.L)
        eig = np.einsum("y,qyk->q", pi, self.L * (self.logL - np.log(np.where(pk > 0, pk, 1))[:, None, :]))
        cand = sorted((i for i, q in enumerate(local) if q not in hist and q != q0), key=lambda i: -eig[local[i]])
        return ("ask", cand[:3]) if cand else ("dx", int(np.argmax(pi)))

    def line(self, q0, local, hist, kid):
        pi = self.posterior(q0, hist, kid); top = np.argsort(-pi)[:3]
        lik = ", ".join(f"{self.conds[y]} {pi[y]:.2f}" for y in top)
        kind, val = self.recommend(pi, q0, local, hist)
        rec = (f"DIAGNOSIS: {self.conds[val]}." if kind == "dx" else
               ", ".join(f"ASK {i + 1}" + (" (best)" if j == 0 else "") for j, i in enumerate(val)) + ".")
        return (f"Experience statistics from {LABEL_N} past confirmed patients: most likely now: {lik}. "
                f"Recommended next: {rec}"), kind, val


def insert(user_prompt, line):
    for key in INSERT_BEFORES:
        i = user_prompt.find(key)
        if i != -1:
            return user_prompt[:i] + line + "\n" + user_prompt[i:]
    return user_prompt + "\n" + line
