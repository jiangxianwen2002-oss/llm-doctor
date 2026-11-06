#!/usr/bin/env python3
"""Round 98 (PREREG_r98): run t1_run_s2 / t1_run_sig / t1_run_bad97q unchanged against the qwen3.8-flash API.
Only the HTTP client changes: same messages, temperature 0.8 (t1_run.TEMP), max_tokens 40, thinking off via enable_thinking,
same per-call seed rule as the 4B arms. Usage, latency and a few reply samples go to <out>/api_usage*.json (one file per run).
After a fatal API error all later calls fail at once, so no paid calls are spent on interviews that could not be saved.
The API key is read from the first line of stdin (never argv, env file or logs).
  <key on stdin> | python t1_run_api_qwen.py 15 t1_run_sig.py --base-url <url> --model qwen3.8-flash --out <dir> ..."""
import atexit, collections, json, os, runpy, signal, statistics, sys, threading, time, urllib.error, urllib.request
import t1_run as T

KEY = sys.stdin.readline().strip()
assert KEY, "no API key on stdin"
OUT = sys.argv[sys.argv.index("--out") + 1]
FATAL = {400, 401, 403, 404}
LOCK = threading.Lock(); ABORT = threading.Event(); T0 = time.time(); DUMPED = []
S = dict(calls=0, tin=0, tout=0, cached=0, reasoning_nonempty=0, retries=0, http=collections.Counter(), finish=collections.Counter(),
         models=collections.Counter(), lat=[], samples=[])


def clean(text):
    return str(text).replace(KEY, "<key>")[:300]


class QwenClient(T.Client):
    def __init__(self, base, model, key):
        super().__init__(base, model, KEY)

    def chat(self, sys_p, user, seed):
        body = {"model": self.model, "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                "temperature": T.TEMP, "max_tokens": 40, "seed": seed, "enable_thinking": False}
        for attempt in range(12):
            if ABORT.is_set():
                raise RuntimeError("stopped after an earlier fatal API error")
            t = time.time()
            try:
                req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    j = json.loads(r.read())
                u = j.get("usage") or {}; c = j["choices"][0]; m = c["message"]; text = m.get("content") or ""
                with self.lock:
                    self.calls += 1; self.tin += u.get("prompt_tokens") or 0; self.tout += u.get("completion_tokens") or 0
                with LOCK:
                    S["calls"] += 1; S["tin"] += u.get("prompt_tokens") or 0; S["tout"] += u.get("completion_tokens") or 0
                    S["cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                    S["reasoning_nonempty"] += bool(m.get("reasoning_content")); S["finish"][str(c.get("finish_reason"))] += 1
                    S["models"][str(j.get("model"))] += 1; S["lat"].append(time.time() - t)
                    if len(S["samples"]) < 30:
                        S["samples"].append(text[:120])
                return text
            except urllib.error.HTTPError as ex:
                detail = clean(ex.read().decode("utf-8", "replace"))
                with self.lock:
                    self.errors += 1
                with LOCK:
                    S["http"][str(ex.code)] += 1; S["retries"] += 1
                if ex.code in FATAL or attempt == 11:
                    ABORT.set()
                    with LOCK:
                        S["fatal"] = f"HTTP {ex.code}: {detail}"
                    raise RuntimeError(f"HTTP {ex.code}: {detail}") from None
            except Exception as ex:
                with self.lock:
                    self.errors += 1
                with LOCK:
                    S["http"][type(ex).__name__] += 1; S["retries"] += 1
                if attempt == 11:
                    ABORT.set()
                    raise RuntimeError(f"API not answering: {type(ex).__name__}") from None
            time.sleep(min(60, 3 * (attempt + 1)))


def dump():
    if DUMPED:
        return
    DUMPED.append(1)
    lat = S.pop("lat") or [0.0]
    S.update(elapsed_s=round(time.time() - T0, 1), lat_mean_s=round(statistics.mean(lat), 2), lat_max_s=round(max(lat), 2),
             budget=T.BUDGET)
    d = T.HERE / OUT; d.mkdir(parents=True, exist_ok=True)
    f = d / "api_usage.json"; i = 2
    while f.exists():
        f = d / f"api_usage_{i}.json"; i += 1
    json.dump(S, open(f, "w"), indent=1, ensure_ascii=False, default=dict)


def on_term(signum, frame):
    dump(); os._exit(143)


atexit.register(dump)
signal.signal(signal.SIGTERM, on_term)
T.Client = QwenClient
T.BUDGET = int(sys.argv[1]); script = sys.argv[2]
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
