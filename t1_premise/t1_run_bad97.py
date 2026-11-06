#!/usr/bin/env python3
"""Round 97 (PREREG_r97): t1_run_bad.py with a graded corrupted line from ddx_bad97.py (--variant dxonly | contra).
The episode loop is t1_run_bad.episode unchanged (same prompts, parse, <= T.BUDGET questions, call seed, adherence logging:
follow = vs the shown recommendation, follow_teacher = vs the correct one). Only the line object changes.
  python t1_run_bad97.py --variant contra --model qwen3-4b --out contra97 --stats sig86/sig_stats.npz --kid sig86/kid_fresh.json
"""
import argparse, json, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import t1_run as T
import t1_run_s2 as S
import ddx_signal as D
import t1_run_bad as RB
import ddx_bad97 as B97


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1"); ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True, choices=sorted(B97.VARIANTS))
    ap.add_argument("--out", required=True); ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--patients", type=int); ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--stats", required=True); ap.add_argument("--kid", required=True)
    a = ap.parse_args()
    patients, menu, conds = T.load()
    s2 = {int(k): v for k, v in json.load(open(T.HERE / "data" / "s2_menu.json")).items()}
    bad = B97.VARIANTS[a.variant](D.Stats.load(a.stats, conds)); kids = json.load(open(a.kid))
    patients = patients[: a.patients] if a.patients else patients
    out = T.HERE / a.out; out.mkdir(exist_ok=True); f = out / "episodes.jsonl"
    done = {(j["pid"], j["seed"]) for j in map(json.loads, open(f))} if f.exists() else set()
    todo = [(p, s) for p in patients for s in range(1, a.samples + 1) if (p["pid"], s) not in done]
    sys_p = S.system_prompt(conds); client = T.Client(a.base_url, a.model, "EMPTY"); wlock = threading.Lock()
    print(f"{a.variant.upper()} arm: {len(patients)} patients x {a.samples}; {len(todo)} to run; model {a.model}; budget {T.BUDGET}")
    t0 = time.time(); k = ok = 0; last = 0
    with ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(RB.episode, p, s, client, sys_p, menu, conds, s2[int(p["initial"])], bad, kids[str(p["pid"])]) for p, s in todo]
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
