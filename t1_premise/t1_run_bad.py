#!/usr/bin/env python3
"""Round 93: arm BAD = t1_run_sig.py with the CORRUPTED line of ddx_bad.py instead of the correct one (PREREG_r93 section 4).
Everything else is t1_run_sig.py unchanged: S2 menus, t1_run_s2 prompts and parse, <= 15 questions, call seed = seed * 1000 + steps,
same insertion point. Adherence is logged twice per episode:
  follow         = vs the SHOWN (bad) recommendation (same keys as t1_run_sig, so analyze86-style readers work);
  follow_teacher = vs what the correct signal would have recommended (did the model still pick the teacher's move?).
  python t1_run_bad.py --model qwen3-4b --out sigt_bad93 --stats sig86/sig_stats.npz --kid sig86/kid_fresh.json
"""
import argparse, json, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import t1_run as T
import t1_run_s2 as S
import ddx_signal as D
import ddx_bad as B


def episode(p, seed, client, sys_p, menu, conds, local, bad, kid):
    hist = []; nbad = 0; steps = 0; dx = None
    fol = {"ask_turns": 0, "ask_best": 0, "ask_top3": 0, "dx_turns": 0, "dx_follow": 0}
    tfol = {"ask_turns": 0, "ask_best": 0, "ask_top3": 0, "dx_turns": 0, "dx_follow": 0}
    q0 = int(p["initial"])
    while dx is None:
        final = len(hist) >= T.BUDGET or steps >= T.BUDGET + 5
        line, kind, bval, tval, _ = bad.line(p["pid"], seed, q0, local, hist, kid)
        up = D.insert(S.user_prompt(p, menu, local, hist, final), line)
        mk, mv = T.parse(client.chat(sys_p, up, seed * 1000 + steps), conds)
        steps += 1
        if kind == "ask" and not final:
            for f, v in ((fol, bval), (tfol, tval)):
                f["ask_turns"] += 1; f["ask_best"] += mk == "ask" and mv == v[0]; f["ask_top3"] += mk == "ask" and mv in v
        elif kind == "dx":
            for f, v in ((fol, bval), (tfol, tval)):
                f["dx_turns"] += 1; f["dx_follow"] += mk == "dx" and mv == conds[v]
        if mk == "dx":
            dx = mv
        elif mk == "ask" and 0 <= mv < len(local) and local[mv] not in hist and not final:
            hist.append(local[mv])
        else:
            nbad += 1
            if final and steps >= T.BUDGET + 8:
                break
    return {"pid": p["pid"], "seed": seed, "dx": dx, "correct": dx == p["diagnosis"], "n_questions": len(hist),
            "asked": hist, "bad_outputs": nbad, "truth": p["diagnosis"], "follow": fol, "follow_teacher": tfol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1"); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--patients", type=int); ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--stats", required=True); ap.add_argument("--kid", required=True)
    a = ap.parse_args()
    patients, menu, conds = T.load()
    s2 = {int(k): v for k, v in json.load(open(T.HERE / "data" / "s2_menu.json")).items()}
    bad = B.Bad(D.Stats.load(a.stats, conds)); kids = json.load(open(a.kid))
    patients = patients[: a.patients] if a.patients else patients
    out = T.HERE / a.out; out.mkdir(exist_ok=True); f = out / "episodes.jsonl"
    done = {(j["pid"], j["seed"]) for j in map(json.loads, open(f))} if f.exists() else set()
    todo = [(p, s) for p in patients for s in range(1, a.samples + 1) if (p["pid"], s) not in done]
    sys_p = S.system_prompt(conds); client = T.Client(a.base_url, a.model, "EMPTY"); wlock = threading.Lock()
    print(f"BAD arm: {len(patients)} patients x {a.samples}; {len(todo)} to run; model {a.model}")
    t0 = time.time(); k = ok = 0; last = 0
    with ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(episode, p, s, client, sys_p, menu, conds, s2[int(p["initial"])], bad, kids[str(p["pid"])]) for p, s in todo]
        for fu in as_completed(futs):
            r = fu.result(); k += 1; ok += r["correct"]
            with wlock, open(f, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            if time.time() - last > 3 or k == len(todo):
                last = time.time(); el = time.time() - t0
                print(f"\r{k}/{len(todo)}  {el / 60:5.1f} min  eta {el / k * (len(todo) - k) / 60:5.1f} min  correct so far {100 * ok / k:4.1f}%  "
                      f"calls {client.calls}  errors {client.errors}   ", end="", flush=True)
    print(f"\ndone -> {f}")


if __name__ == "__main__":
    main()
