#!/usr/bin/env python3
"""Round 99 (PREREG_r99): run t1_run_s2 / t1_run_sig unchanged against the glm-5.3-flash API.
Only the HTTP client changes: same messages, temperature 0.8 (t1_run.TEMP), max_tokens 40. This model cannot turn thinking off,
so the lowest setting (reasoning_effort=low) is used and reasoning tokens are recorded per call. No seed (not in the API docs).
Usage goes to <out>/api_usage*.json (one file per run). After a fatal API error, or past TOKEN_CAP tokens in this arm, all later
calls fail at once. The API key is read from the first line of stdin (never argv, env file or logs).
  <key on stdin> | python t1_run_api_glm.py 15 t1_run_sig.py --base-url <url> --model glm-5.3-flash --out <dir> ..."""
import atexit, collections, json, os, runpy, signal, statistics, sys, threading, time, urllib.error, urllib.request
import t1_run as T

KEY = sys.stdin.readline().strip()
assert KEY, "no API key on stdin"
OUT = sys.argv[sys.argv.index("--out") + 1]
TOKEN_CAP = int(os.environ.get("TOKEN_CAP", "0")); MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "40"))
FATAL = {400, 401, 403, 404}
LOCK = threading.Lock(); ABORT = threading.Event(); T0 = time.time(); DUMPED = []
S = dict(calls=0, tin=0, tout=0, cached=0, reasoning_tokens=0, calls_with_reasoning_tokens=0, reasoning_nonempty=0, empty_content=0, retries=0,
         max_tokens=MAX_TOKENS, content_tokens_max=0, calls_content_over_40=0, reasoning_per_call=[],
         http=collections.Counter(), finish=collections.Counter(), models=collections.Counter(), lat=[], samples=[])


def clean(text):
    return str(text).replace(KEY, "<key>")[:300]


class GlmClient(T.Client):
    def __init__(self, base, model, key):
        super().__init__(base, model, KEY)

    def chat(self, sys_p, user, seed):
        body = {"model": self.model, "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                "temperature": T.TEMP, "max_tokens": MAX_TOKENS, "reasoning_effort": "low"}
        for attempt in range(12):
            if ABORT.is_set():
                raise RuntimeError("stopped after an earlier fatal API error or the token cap")
            t = time.time()
            try:
                req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    j = json.loads(r.read())
                u = j.get("usage") or {}; c = j["choices"][0]; m = c["message"]; text = m.get("content") or ""
                rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                with self.lock:
                    self.calls += 1; self.tin += u.get("prompt_tokens") or 0; self.tout += u.get("completion_tokens") or 0
                with LOCK:
                    S["calls"] += 1; S["tin"] += u.get("prompt_tokens") or 0; S["tout"] += u.get("completion_tokens") or 0
                    S["cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                    S["reasoning_tokens"] += rt; S["calls_with_reasoning_tokens"] += rt > 0
                    ct = (u.get("completion_tokens") or 0) - rt; S["content_tokens_max"] = max(S["content_tokens_max"], ct); S["calls_content_over_40"] += ct > 40
                    if rt:
                        S["reasoning_per_call"].append(rt)
                    S["reasoning_nonempty"] += bool(m.get("reasoning_content")); S["empty_content"] += not text.strip()
                    S["finish"][str(c.get("finish_reason"))] += 1; S["models"][str(j.get("model"))] += 1; S["lat"].append(time.time() - t)
                    if len(S["samples"]) < 30:
                        S["samples"].append(text[:120])
                    if TOKEN_CAP and S["tin"] + S["tout"] > TOKEN_CAP and not ABORT.is_set():
                        S["fatal"] = f"token cap {TOKEN_CAP} reached"; ABORT.set()
                return text
            except urllib.error.HTTPError as ex:
                detail = clean(ex.read().decode("utf-8", "replace"))
                with self.lock:
                    self.errors += 1
                with LOCK:
                    S["http"][str(ex.code)] += 1; S["retries"] += 1
                if ex.code in FATAL or '"1113"' in detail or attempt == 11:
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
T.Client = GlmClient
T.BUDGET = int(sys.argv[1]); script = sys.argv[2]
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
