#!/usr/bin/env python3
"""Round 97 (PREREG_r97): two graded variants of the round-93 corrupted line (ddx_bad.py), same format as ddx_signal.py's line.
  DXONLY  ("wrong diagnosis only"): "most likely now" = the CORRECT names and probabilities, ASK turns = the teacher's correct
          top-3; only DIAGNOSIS turns are corrupted: "DIAGNOSIS: X" with X a random condition outside the shown top-3 and never a
          (tied) teacher top-1. The wrong recommendation contradicts the probabilities printed in the same line.
  CONTRA  ("advice contradicts the collected answers"): as ddx_bad.Bad (ASK = 3 random questions outside the teacher's top-3,
          DIAGNOSIS: X = the shown top-1), but the three shown names are the conditions with the LOWEST posterior given the
          answers collected so far, i.e. the conditions the patient's own answers argue against most; the three correct top-3
          probabilities are kept. Recomputed every turn from the same answers the LLM sees.
Timing (when DIAGNOSIS is recommended) = the teacher's, as in round 93. Randomness: random.Random(string key) -> reproducible.
The correct statistics (ddx_signal.Stats, unchanged) are used only to build the lines.
"""
import random

import numpy as np

import ddx_bad as B


class DxOnly:
    def __init__(self, st):
        self.st = st; self.conds = st.conds

    def line(self, pid, seed, q0, local, hist, kid):
        """-> (line, kind, shown_val, teacher_val, shown), same contract as ddx_bad.Bad.line"""
        st = self.st; pi = st.posterior(q0, hist, kid); top = [int(y) for y in np.argsort(-pi)[:3]]
        kind, tval = st.recommend(pi, q0, local, hist)
        if kind == "dx":
            t1 = {int(y) for y in np.flatnonzero(pi == pi.max())}
            cand = [y for y in range(len(self.conds)) if y not in top and y not in t1]
            bval = random.Random(f"r97dxonly|dx|{pid}|{seed}").choice(cand)
        else:
            bval = tval
        return B._fmt(self.conds, top, [pi[y] for y in top], kind, bval), kind, bval, tval, top


class Contra:
    def __init__(self, st):
        self.st = st; self.conds = st.conds

    def line(self, pid, seed, q0, local, hist, kid):
        st = self.st; pi = st.posterior(q0, hist, kid); top = [int(y) for y in np.argsort(-pi)[:3]]
        kind, tval = st.recommend(pi, q0, local, hist)
        # lowest posterior first; exact ties broken by a fixed random key per (patient, seed)
        tie = random.Random(f"r97contra|tie|{pid}|{seed}").random
        keys = [tie() for _ in range(len(self.conds))]
        shown = sorted(range(len(self.conds)), key=lambda y: (pi[y], keys[y]))[:3]
        if kind == "dx":
            bval = shown[0]
        else:
            cand = [i for i, q in enumerate(local) if q not in hist and q != q0 and i not in tval]
            bval = random.Random(f"r97contra|ask|{pid}|{seed}|{len(hist)}").sample(cand, min(3, len(cand)))
        return B._fmt(self.conds, shown, [pi[y] for y in top], kind, bval), kind, bval, tval, shown


VARIANTS = {"dxonly": DxOnly, "contra": Contra}
