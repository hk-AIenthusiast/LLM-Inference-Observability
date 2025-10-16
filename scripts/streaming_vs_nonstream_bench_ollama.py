import time, json, csv, requests
from pathlib import Path
import numpy as np

MODEL = "llama3"
HOST = "http://localhost:11434"
PROMPT = "Explain RAG in 4–5 sentences."
RUNS = 20
KEEP_ALIVE = "10m"
OUT_CSV = Path("streaming_vs_nonstream.csv")

def chat_stream(stream: bool):
    url = f"{HOST}/api/chat"
    payload = {"model": MODEL, "messages":[{"role":"user","content":PROMPT}], "keep_alive": KEEP_ALIVE, "stream": stream}
    if stream:
        start = time.perf_counter(); ttft = None; last = {}
        with requests.post(url, json=payload, stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line: continue
                try: data = json.loads(line)
                except: continue
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
        r = requests.post(url, json=payload, timeout=600)  # full response (no token stream)
        r.raise_for_status()
        e2e = time.perf_counter() - start
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
        # No true TTFT in non-stream; report None
        return {"mode":"nonstream","ttft_s":None,"e2e_s":e2e,
                "prompt_tokens":int(data.get("prompt_eval_count",0)),
                "output_tokens":int(data.get("eval_count",0))}

def run():
    new = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["mode","run_idx","ttft_s","e2e_s","prompt_tokens","output_tokens"])
        # warm-up
        _ = chat_stream(True)
        for i in range(1, RUNS+1):
            s = chat_stream(True)
            w.writerow([s["mode"], i, s["ttft_s"], s["e2e_s"], s["prompt_tokens"], s["output_tokens"]])
        for i in range(1, RUNS+1):
            s = chat_stream(False)
            w.writerow([s["mode"], i, s["ttft_s"], s["e2e_s"], s["prompt_tokens"], s["output_tokens"]])

if __name__ == "__main__":
    run()
