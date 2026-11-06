#!/usr/bin/env python3
"""Round 93 (DDXPlus menu format): CORRUPTED experience-signal line (arm BAD), PREREG_r93 section 4.
Same format as ddx_signal.py's line, wrong content, deterministically avoiding the teacher's choices:
  1. ASK turns: 3 random not-yet-asked menu questions, EXCLUDING the teacher's top-3 (shown as "ASK a (best), ASK b, ASK c").
  2. DIAGNOSIS turns (same turns as the teacher: posterior >= 0.9 or 10 questions): "DIAGNOSIS: X" with X != the teacher's diagnosis.
  3. "most likely now": the three correct probabilities, condition names replaced through a fixed derangement of the 49
     conditions per (patient, seed) -> consistent across turns, the teacher's top-1 is never shown first; X = the shown top-1.
     Exact posterior ties at the top count as the teacher's top-1 too: if the mapped first name is a tied top-1 (or repeats
     another shown name), it walks on along the cycle (rare, deterministic).
Timing (when DIAGNOSIS is recommended) is kept = the teacher's (only content is corrupted).
Randomness: random.Random(string key) -> reproducible; the derangement is a single cycle (Sattolo), so no condition maps to itself.
The correct statistics (ddx_signal.Stats, unchanged) are used only to know what to avoid.
"""
import random

import numpy as np

import ddx_signal as D


def _fmt(conds, shown, probs, kind, val):
    lik = ", ".join(f"{conds[y]} {p:.2f}" for y, p in zip(shown, probs))
    rec = (f"DIAGNOSIS: {conds[val]}." if kind == "dx" else
           ", ".join(f"ASK {i + 1}" + (" (best)" if j == 0 else "") for j, i in enumerate(val)) + ".")
    return (f"Experience statistics from {D.LABEL_N} past confirmed patients: most likely now: {lik}. "
            f"Recommended next: {rec}")


def derangement(n, key):
    """Sattolo's algorithm: a uniformly random single n-cycle (no fixed points), deterministic for the key."""
    p = list(range(n)); rng = random.Random(key)
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)
        p[i], p[j] = p[j], p[i]
    return p


class Bad:
    def __init__(self, st):
        self.st = st; self.conds = st.conds; self._perm = {}

    def perm(self, pid, seed):
        k = (str(pid), int(seed))
        if k not in self._perm:
            self._perm[k] = derangement(len(self.conds), f"r93bad|perm|{pid}|{seed}")
        return self._perm[k]

    def line(self, pid, seed, q0, local, hist, kid):
        """-> (bad_line, kind, bad_val, teacher_val, shown); kind/teacher_val = what the correct signal would recommend."""
        st = self.st; pi = st.posterior(q0, hist, kid); top = [int(y) for y in np.argsort(-pi)[:3]]
        kind, tval = st.recommend(pi, q0, local, hist)
        pm = self.perm(pid, seed); shown = [pm[y] for y in top]
        # ties: the teacher's top-1 = every condition at the maximum posterior (exact float ties occur, e.g. conditions absent
        # from the 235-patient bank have identical smoothed likelihoods). If the mapped first name is one of them, keep walking
        # the single cycle until it is neither a tied top-1 nor one of the other two shown names (rare; deterministic).
        t1 = {int(y) for y in np.flatnonzero(pi == pi.max())}
        for _ in range(len(pm)):
            if shown[0] not in t1 and shown[0] not in shown[1:]:
                break
            shown[0] = pm[shown[0]]
        if kind == "dx":
            bval = shown[0]                      # X = the shown top-1, never a (tied) teacher top-1, so X != the teacher's dx
        else:
            cand = [i for i, q in enumerate(local) if q not in hist and q != q0 and i not in tval]
            rng = random.Random(f"r93bad|ask|{pid}|{seed}|{len(hist)}")
            bval = rng.sample(cand, min(3, len(cand)))
        return _fmt(self.conds, shown, [pi[y] for y in top], kind, bval), kind, bval, tval, shown
