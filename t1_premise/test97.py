import json, re, sys, statistics
sys.path.insert(0, '.')
import numpy as np
import t1_run as T, t1_run_s2 as S, ddx_signal as D, ddx_bad as B, ddx_bad97 as B97, t1_run_bad as RB
pts, menu, conds = T.load(); s2 = {int(k): v for k, v in json.load(open("data/s2_menu.json")).items()}
st = D.Stats.load("sig86/sig_stats.npz", conds); kids = json.load(open("sig86/kid_fresh.json"))
class Fake:  # always follows the shown recommendation
    calls = errors = 0
    def chat(self, sp, up, seed):
        l = [x for x in up.splitlines() if x.startswith("Experience")]; assert len(l) == 1
        rec = l[0].split("Recommended next: ")[1]; m = re.match(r"ASK (\d+)", rec)
        return f"ASK {m.group(1)}" if m else "DIAGNOSIS: " + rec[len("DIAGNOSIS: "):].rstrip(".")
sp = S.system_prompt(conds)
for name, cls in [("bad93", B.Bad), ("dxonly", B97.DxOnly), ("contra", B97.Contra)]:
    obj = cls(st); viol = 0; turns = 0; shown_post = []; acc = []
    for p in pts:
        kid = kids[str(p["pid"])]; local = s2[int(p["initial"])]; q0 = int(p["initial"])
        # walk the teacher's own path and check every line on it
        hist = []
        for _ in range(16):
            line, kind, bval, tval, shown = obj.line(p["pid"], 1, q0, local, hist, kid)
            pi = st.posterior(q0, hist, kid); turns += 1
            cl, ck, cv = st.line(q0, local, hist, kid)
            assert kind == ck and tval == cv
            shown_post.append(pi[shown[0]] / pi.max())
            if name == "dxonly":
                if kind == "ask": viol += line != cl
                else: viol += bval == tval or bval in shown or ("most likely now: " + cl.split("most likely now: ")[1].split(". Recommended")[0]) not in line
            else:
                if kind == "dx": viol += bval == tval
                else: viol += bool(set(bval) & set(tval))
            if kind == "dx": break
            hist.append(local[tval[0]])
        acc.append(RB.episode(p, 1, Fake(), sp, menu, conds, local, obj, kid)["correct"])
    print(f"{name:7s} turns {turns}  violations {viol}  shown-top1 posterior / max: median {statistics.median(shown_post):.2e}  mean {statistics.mean(shown_post):.3f}  "
          f"follow-everything accuracy {100*statistics.mean(acc):.1f}%")
# one example line per variant, mid-interview
p = pts[0]; kid = kids[str(p["pid"])]; local = s2[int(p["initial"])]
for cls in (B.Bad, B97.DxOnly, B97.Contra):
    print(cls.__name__, "|", cls(st).line(p["pid"], 1, int(p["initial"]), local, [local[0], local[1]], kid)[0])
