import time, json, csv, requests
from pathlib import Path

MODEL = "llama3"
HOST = "http://localhost:11434"
PROMPT = "Say 'OK' and nothing else."
IDLE_SECONDS = [0, 15, 30, 60, 120, 300, 600]   # adjust as needed
KEEP_ALIVE_AFTER = "10m"                        # keep warm after probe
OUT_CSV = Path("keepalive_decay.csv")

def stream_once():
    url=f"{HOST}/api/chat"
    payload={"model":MODEL,"messages":[{"role":"user","content":PROMPT}],"stream":True,"keep_alive":KEEP_ALIVE_AFTER}
    start=time.perf_counter(); ttft=None; last={}
    with requests.post(url,json=payload,stream=True,timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line: continue
            try: data=json.loads(line)
            except: continue
            last=data
            if data.get("message",{}).get("content") and ttft is None:
                ttft=time.perf_counter()-start
            if data.get("done"): break
    e2e=time.perf_counter()-start
    return ttft, e2e

def main():
    new=not OUT_CSV.exists()
    with OUT_CSV.open("a",newline="") as f:
        w=csv.writer(f)
        if new: w.writerow(["idle_s","ttft_s","e2e_s"])
        # prime
        _=stream_once()
        for idle in IDLE_SECONDS:
            print(f"Sleeping {idle}s...")
            time.sleep(idle)
            ttft,e2e = stream_once()
            w.writerow([idle, f"{ttft:.6f}", f"{e2e:.6f}"])
            print(f"idle={idle}s  TTFT={ttft:.3f}s  E2E={e2e:.3f}s")

if __name__=="__main__":
    main()
