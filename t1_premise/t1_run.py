#!/usr/bin/env python3
"""T1 premise check : a local model interviews 300 DDXPlus patients 8 times each.
Standard library only. The doctor talks to any OpenAI-compatible server (local vLLM); the patient is a deterministic lookup.
Resumable: results are appended to OUT/episodes.jsonl and finished (patient, sample) pairs are skipped.

  python t1_run.py --base-url http://localhost:8000/v1 --model qwen3-4b --out results
  python t1_run.py --base-url http://localhost:8000/v1 --model qwen3-4b --out results_smoke --patients 3 --samples 2
"""
import argparse, difflib, json, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUDGET, TEMP = 15, 0.8


def load():
    d = HERE / "data"
    return (json.load(open(d / "patients.json")), json.load(open(d / "menu.json")), json.load(open(d / "conditions.json")))


def system_prompt(menu, conds):
    return ("You are a doctor in a telehealth interview. The patient has exactly ONE of these conditions:\n" + "; ".join(conds) +
            f"\n\nYou can only learn more by asking questions from the numbered MENU below (at most {BUDGET} questions). "
            "Each turn, reply with exactly ONE line and nothing else:\n  ASK <number>        to ask that menu question\n"
            "  DIAGNOSIS: <condition>   when you are ready (use a condition name from the list exactly)\n\nMENU:\n" +
            "\n".join(f"{i + 1}. {q}" for i, q in enumerate(menu)))


def user_prompt(p, menu, hist, final):
    lines = [f"Patient: {p['age']}-year-old {p['sex']}. Reason for consulting: {menu[p['initial']]} -> {p['answers'][p['initial']]}"]
    if hist:
        lines.append("Questions asked so far:")
        lines += [f"- [{q + 1}] {menu[q]} -> {p['answers'][q]}" for q in hist]
    lines.append(f"You have asked {len(hist)} of {BUDGET} questions.")
    lines.append("You must now reply with: DIAGNOSIS: <condition>" if final else "Your move (ASK <number> or DIAGNOSIS: <condition>):")
    return "\n".join(lines)


class Client:
    def __init__(self, base, model, key):
        self.url = base.rstrip("/") + "/chat/completions"; self.model = model; self.key = key
        self.lock = threading.Lock(); self.calls = self.tin = self.tout = self.errors = 0

    def chat(self, sys_p, user, seed):
        body = {"model": self.model, "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                "temperature": TEMP, "seed": seed, "max_tokens": 40,
                "chat_template_kwargs": {"enable_thinking": False}}
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    j = json.loads(r.read())
                u = j.get("usage") or {}
                with self.lock:
                    self.calls += 1; self.tin += u.get("prompt_tokens") or 0; self.tout += u.get("completion_tokens") or 0
                return j["choices"][0]["message"].get("content") or ""
            except Exception as ex:
                with self.lock:
                    self.errors += 1
                if attempt == 7:
                    raise RuntimeError(f"server not answering: {ex}")
                time.sleep(2 * (attempt + 1))


def parse(text, conds):
    t = (text or "").strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    m = re.search(r"DIAGNOSIS\s*:\s*(.+)", t, re.I)
    if m:
        name = m.group(1).strip().strip(".*\"' ")
        low = {c.lower(): c for c in conds}
        if name.lower() in low:
            return "dx", low[name.lower()]
        best = difflib.get_close_matches(name.lower(), list(low), n=1, cutoff=0.6)
        return ("dx", low[best[0]]) if best else ("bad", None)
    m = re.search(r"ASK\s*#?\s*(\d+)", t, re.I)
    if m:
        return "ask", int(m.group(1)) - 1
    return "bad", None


def episode(p, seed, client, sys_p, menu, conds):
    hist = []; bad = 0; steps = 0; dx = None
    while dx is None:
        final = len(hist) >= BUDGET or steps >= BUDGET + 5
        kind, val = parse(client.chat(sys_p, user_prompt(p, menu, hist, final), seed * 1000 + steps), conds)
        steps += 1
        if kind == "dx":
            dx = val
        elif kind == "ask" and 0 <= val < len(menu) and not final:
            hist.append(val)
        else:
            bad += 1
            if final and steps >= BUDGET + 8:
                break
    return {"pid": p["pid"], "seed": seed, "dx": dx, "correct": dx == p["diagnosis"], "n_questions": len(hist),
            "asked": hist, "bad_outputs": bad, "truth": p["diagnosis"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1"); ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY"); ap.add_argument("--out", default="results")
    ap.add_argument("--samples", type=int, default=8); ap.add_argument("--patients", type=int)
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    patients, menu, conds = load()
    patients = patients[: a.patients] if a.patients else patients
    out = HERE / a.out; out.mkdir(exist_ok=True); f = out / "episodes.jsonl"
    done = {(j["pid"], j["seed"]) for j in map(json.loads, open(f))} if f.exists() else set()
    todo = [(p, s) for p in patients for s in range(1, a.samples + 1) if (p["pid"], s) not in done]
    sys_p = system_prompt(menu, conds); client = Client(a.base_url, a.model, a.api_key); wlock = threading.Lock()
    print(f"{len(patients)} patients x {a.samples} samples; {len(todo)} episodes to run ({len(done)} already done); server {a.base_url}, model {a.model}")
    t0 = time.time(); k = 0; ok = 0; last = 0
    with ThreadPoolExecutor(a.concurrency) as ex:
        futs = [ex.submit(episode, p, s, client, sys_p, menu, conds) for p, s in todo]
        for fu in as_completed(futs):
            r = fu.result(); k += 1; ok += r["correct"]
            with wlock, open(f, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            if time.time() - last > 3 or k == len(todo):
                last = time.time(); el = time.time() - t0
                print(f"\r{k}/{len(todo)} episodes  {el / 60:5.1f} min  eta {el / k * (len(todo) - k) / 60:5.1f} min  "
                      f"correct so far {100 * ok / k:4.1f}%  calls {client.calls}  server errors {client.errors}   ", end="", flush=True)
    print(f"\ndone -> {f}\nnext: python t1_analyze.py --out {a.out}")


if __name__ == "__main__":
    main()
