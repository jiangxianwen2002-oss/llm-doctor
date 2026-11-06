#!/usr/bin/env python3
"""Round 96 (PREREG_r96): run t1_run_s2 / t1_run_sig / t1_run_bad unchanged against the DeepSeek API instead of vLLM.
Only the HTTP client changes: same messages, temperature 0.8 (t1_run.TEMP), max_tokens 40; thinking disabled the way
round 92 did it ({"thinking": {"type": "disabled"}}); no seed (the API does not take one), no chat_template_kwargs.
The API key is read from the first line of stdin (never argv, env file or logs).
  <key on stdin> | python t1_run_api.py 15 t1_run_sig.py --base-url https://api.deepseek.com --model deepseek-flash --out ds_sig0_96 ..."""
import json, runpy, sys, time, urllib.request
import t1_run as T
KEY = sys.stdin.readline().strip()
assert KEY, "no API key on stdin"


class DSClient(T.Client):
    def __init__(self, base, model, key):
        super().__init__(base, model, KEY)

    def chat(self, sys_p, user, seed):
        body = {"model": self.model, "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                "temperature": T.TEMP, "max_tokens": 40, "thinking": {"type": "disabled"}}
        for attempt in range(12):
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
                if attempt == 11:
                    raise RuntimeError(f"API not answering: {type(ex).__name__}")
                time.sleep(min(60, 3 * (attempt + 1)))


T.Client = DSClient
T.BUDGET = int(sys.argv[1]); script = sys.argv[2]
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
