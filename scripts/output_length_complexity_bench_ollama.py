import time, json, csv, requests
from pathlib import Path
import numpy as np

MODEL = "llama3"
HOST = "http://localhost:11434"
KEEP_ALIVE = "10m"
RUNS = 15
OUT_CSV = Path("output_length_complexity.csv")

TASKS = {
    "factual":   "List 10 key facts about the Eiffel Tower.",
    "reasoning": "Solve step-by-step: If a factory produces 120 widgets per hour and demand increases by 35%, how many widgets are needed per 8-hour shift? Explain your reasoning briefly.",
    "summarize": "Summarize the following concept in simple terms: Retrieval-Augmented Generation (RAG), covering ingestion, chunking, embeddings, vector search, and reranking."
}
NUM_PREDICT = [64, 256, 512]   # cap output length

def chat_stream(prompt, num_predict):
    url=f"{HOST}/api/chat"
    payload={
        "model":MODEL,
        "messages":[{"role":"user","content":prompt}],
        "keep_alive":KEEP_ALIVE,
        "stream":True,
        "options":{"num_predict": num_predict}
    }
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
    pt=int(last.get("prompt_eval_count",0)); ot=int(last.get("eval_count",0))
    ev=float(last.get("eval_duration",0)/1e9); tps=(ot/ev) if ev>0 else 0.0
    return ttft,e2e,pt,ot,tps

def main():
    new=not OUT_CSV.exists()
    with OUT_CSV.open("a",newline="") as f:
        w=csv.writer(f)
        if new: w.writerow(["task","num_predict","run_idx","ttft_s","e2e_s","prompt_tokens","output_tokens","decode_tps"])
        # warm-up
        _=chat_stream("Warm up, say OK.", 32)
        for task, prompt in TASKS.items():
            for n in NUM_PREDICT:
                for i in range(1, RUNS+1):
                    ttft,e2e,pt,ot,tps = chat_stream(prompt, n)
                    w.writerow([task,n,i,f"{ttft:.6f}",f"{e2e:.6f}",pt,ot,f"{tps:.3f}"])
                    print(f"[{task} n={n}] {i}/{RUNS} TTFT={ttft:.3f}s E2E={e2e:.3f}s out={ot} tok tps={tps:.1f}")

if __name__=="__main__":
    main()
