#!/usr/bin/env python3
"""Round 36 arm S2: same premise as t1_run.py, but each patient gets a label-free 50-question menu (data/s2_menu.json,
chosen from unlabelled patients sharing the presenting complaint), shown in the user turn with stable local numbers;
already-asked questions are removed from the shown list. Output format identical to t1_run.py (asked = global indices).
  python t1_run_s2.py --model qwen3-4b --out results_s2
"""
import argparse, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import t1_run as T


def system_prompt(conds):
    return ("You are a doctor in a telehealth interview. The patient has exactly ONE of these conditions:\n" + "; ".join(conds) +
            f"\n\nYou can only learn more by asking questions from the numbered MENU given each turn (at most {T.BUDGET} questions). "
            "Each turn, reply with exactly ONE line and nothing else:\n  ASK <number>        to ask that menu question\n"
            "  DIAGNOSIS: <condition>   when you are ready (use a condition name from the list exactly)")


def user_prompt(p, menu, local, hist, final):
    lines = [f"Patient: {p['age']}-year-old {p['sex']}. Reason for consulting: {menu[p['initial']]} -> {p['answers'][p['initial']]}"]
    if hist:
        lines.append("Questions asked so far:")
        lines += [f"- {menu[q]} -> {p['answers'][q]}" for q in hist]
    lines.append(f"You have asked {len(hist)} of {T.BUDGET} questions.")
    if final:
        lines.append("You must now reply with: DIAGNOSIS: <condition>")
    else:
        lines.append("MENU (questions not yet asked):")
        lines += [f"{i + 1}. {menu[q]}" for i, q in enumerate(local) if q not in hist]
        lines.append("Your move (ASK <number> or DIAGNOSIS: <condition>):")
    return "\n".join(lines)


def episode(p, seed, client, sys_p, menu, conds, local):
    hist = []; bad = 0; steps = 0; dx = None
    while dx is None:
        final = len(hist) >= T.BUDGET or steps >= T.BUDGET + 5
        kind, val = T.parse(client.chat(sys_p, user_prompt(p, menu, local, hist, final), seed * 1000 + steps), conds)
        steps += 1
        if kind == "dx":
            dx = val
        elif kind == "ask" and 0 <= val < len(local) and local[val] not in hist and not final:
            hist.append(local[val])
        else:
            bad += 1
            if final and steps >= T.BUDGET + 8:
                break
    return {"pid": p["pid"], "seed": seed, "dx": dx, "correct": dx == p["diagnosis"], "n_questions": len(hist),
            "asked": hist, "bad_outputs": bad, "truth": p["diagnosis"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1"); ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results_s2"); ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--patients", type=int); ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    patients, menu, conds = T.load()
    s2 = {int(k): v for k, v in json.load(open(T.HERE / "data" / "s2_menu.json")).items()}
    patients = patients[: a.patients] if a.patients else patients
    out = T.HERE / a.out; out.mkdir(exist_ok=True); f = out / "episodes.jsonl"
    done = {(j["pid"], j["seed"]) for j in map(json.loads, open(f))} if f.exists() else set()
    todo = [(p, s) for p in patients for s in range(1, a.samples + 1) if (p["pid"], s) not in done]
    sys_p = system_prompt(conds); client = T.Client(a.base_url, a.model, "EMPTY"); wlock = threading.Lock()
    print(f"S2 arm: {len(patients)} patients x {a.samples}; {len(todo)} to run; model {a.model}")
    t0 = time.time(); k = ok = 0; last = 0
    with ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(episode, p, s, client, sys_p, menu, conds, s2[p["initial"]]) for p, s in todo]
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
