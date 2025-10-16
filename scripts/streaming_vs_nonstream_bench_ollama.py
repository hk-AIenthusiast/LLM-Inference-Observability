import time, json, csv, requests
from pathlib import Path

# ---------------- CONFIG ----------------
MODEL = "llama3"
HOST = "http://localhost:11434"
PROMPT = "Explain RAG in 4–5 sentences."
RUNS = 20
KEEP_ALIVE = "10m"
OUT_CSV = Path("../data/streaming_vs_nonstream.csv")
TIMEOUT_S = 600
# ----------------------------------------

FIELDS = ["mode","run_idx","ttft_s","e2e_s","prompt_tokens","output_tokens","error"]

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def chat_once(stream: bool):
    url = f"{HOST}/api/chat"
    payload = {
        "model": MODEL,
        "messages": [{"role":"user","content":PROMPT}],
        "keep_alive": KEEP_ALIVE,
        "stream": stream,
    }
    try:
        if stream:
            start = time.perf_counter()
            ttft = None
            last = {}
            with requests.post(url, json=payload, stream=True, timeout=TIMEOUT_S) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line: continue
                    try: data = json.loads(line)
                    except json.JSONDecodeError: continue
                    last = data
                    if data.get("message",{}).get("content") and ttft is None:
                        ttft = time.perf_counter() - start
                    if data.get("done"): break
            e2e = time.perf_counter() - start
            return {"mode":"stream","ttft_s":ttft,"e2e_s":e2e,
                    "prompt_tokens":int(last.get("prompt_eval_count",0)),
                    "output_tokens":int(last.get("eval_count",0))}
        else:
            start = time.perf_counter()
            r = requests.post(url, json=payload, timeout=TIMEOUT_S)
            r.raise_for_status()
            e2e = time.perf_counter() - start
            data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
            return {"mode":"nonstream","ttft_s":None,"e2e_s":e2e,
                    "prompt_tokens":int(data.get("prompt_eval_count",0)),
                    "output_tokens":int(data.get("eval_count",0))}
    except Exception as e:
        return {"mode":"stream" if stream else "nonstream","error":str(e)}

def run():
    ensure_parent(OUT_CSV)

    # Always overwrite (write mode 'w')
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        _ = chat_once(True)  # warm-up

        # Stream runs
        for i in range(1, RUNS+1):
            r = chat_once(True)
            r["run_idx"] = i
            for k in FIELDS:
                r.setdefault(k,"")
            writer.writerow(r)

        # Non-stream runs
        for i in range(1, RUNS+1):
            r = chat_once(False)
            r["run_idx"] = i
            for k in FIELDS:
                r.setdefault(k,"")
            writer.writerow(r)

    print(f"Wrote {2*RUNS} rows to: {OUT_CSV.resolve()}")
    print("Header columns:", ", ".join(FIELDS))

if __name__ == "__main__":
    run()
