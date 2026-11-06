#!/usr/bin/env python3
"""Round 97 addendum (PREREG_r97 section 8): the third graded variant, kept in its own file so ddx_bad97.py stays unchanged.
  QONLY ("wrong questions only"): "most likely now" = the CORRECT names and probabilities, DIAGNOSIS turns = the teacher's
        correct diagnosis; only ASK turns are corrupted: 3 random not-yet-asked menu questions outside the teacher's top 3
        (as ddx_bad.Bad). The posterior and the recommended diagnosis are computed from the answers actually collected.
Importing this module registers the variant in ddx_bad97.VARIANTS under "qonly".
"""
import random

import numpy as np

import ddx_bad as B
import ddx_bad97 as B97


class QOnly:
    def __init__(self, st):
        self.st = st; self.conds = st.conds

    def line(self, pid, seed, q0, local, hist, kid):
        """-> (line, kind, shown_val, teacher_val, shown), same contract as ddx_bad.Bad.line"""
        st = self.st; pi = st.posterior(q0, hist, kid); top = [int(y) for y in np.argsort(-pi)[:3]]
        kind, tval = st.recommend(pi, q0, local, hist)
        if kind == "dx":
            bval = tval
        else:
            cand = [i for i, q in enumerate(local) if q not in hist and q != q0 and i not in tval]
            bval = random.Random(f"r97qonly|ask|{pid}|{seed}|{len(hist)}").sample(cand, min(3, len(cand)))
        return B._fmt(self.conds, top, [pi[y] for y in top], kind, bval), kind, bval, tval, top


B97.VARIANTS["qonly"] = QOnly
