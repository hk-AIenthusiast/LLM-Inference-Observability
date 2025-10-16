import time
import json
import csv
from pathlib import Path
from statistics import median
import numpy as np
import threading
import requests

# -------------------- CONFIG --------------------
MODEL = "llama3"                       # your local Ollama model
HOST = "http://localhost:11434"
KEEP_ALIVE = "10m"                     # keep the model warm
PROMPT = "In one short paragraph, explain Retrieval-Augmented Generation (RAG) and when to use it."
CONCURRENCY_LEVELS = [1, 2, 4, 8]      # number of parallel clients
RUNS_PER_WORKER = 10                   # requests each worker performs
INTER_CALL_SLEEP = 0.05                # per-worker delay between its calls
OUT_CSV = Path("../data/concurrency_runs.csv") # raw per-request data
# ------------------------------------------------


def chat_stream_ollama(model: str, prompt: str, keep_alive):
    """
    Streaming chat to Ollama with TTFT/E2E and token metrics.
    Returns dict: ttft_s, e2e_s, prompt_tokens, output_tokens, decode_tps
    """
    url = f"{HOST}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "keep_alive": keep_alive,
    }

    start = time.perf_counter()
    first_token_time = None
    last_data = {}

    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            last_data = data  # final 'done' carries metrics

            # first streamed content -> TTFT
            msg = data.get("message", {})
            content = msg.get("content", "")
            if content and first_token_time is None:
                first_token_time = time.perf_counter()

            if data.get("done"):
                break

    end = time.perf_counter()
    if first_token_time is None:
        first_token_time = end

    prompt_tokens = int(last_data.get("prompt_eval_count", 0))
    output_tokens = int(last_data.get("eval_count", 0))
    eval_duration_s = float(last_data.get("eval_duration", 0) / 1e9)
    decode_tps = (output_tokens / eval_duration_s) if eval_duration_s > 0 else 0.0

    return {
        "ttft_s": first_token_time - start,
        "e2e_s": end - start,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "decode_tps": decode_tps,
    }


def percentile(arr, p):
    return float(np.percentile(np.array(arr, dtype=float), p)) if arr else 0.0


def run_concurrency_level(conc: int):
    """
    Launch 'conc' parallel workers; each performs RUNS_PER_WORKER requests.
    Returns (rows, wall_seconds) where rows is a list of dicts (per request).
    """
    rows = []
    rows_lock = threading.Lock()
    start_barrier = threading.Barrier(conc)

    # one-time warm-up (not measured)
    _ = chat_stream_ollama(MODEL, "Warm up, say OK.", keep_alive=KEEP_ALIVE)

    def worker(worker_id: int):
        nonlocal rows
        start_barrier.wait()  # synchronize start
        for i in range(1, RUNS_PER_WORKER + 1):
            try:
                m = chat_stream_ollama(MODEL, PROMPT, keep_alive=KEEP_ALIVE)
                rec = {
                    "concurrency": conc,
                    "worker_id": worker_id,
                    "run_idx": i,
                    **m,
                }
                with rows_lock:
                    rows.append(rec)
            except Exception as e:
                with rows_lock:
                    rows.append({
                        "concurrency": conc, "worker_id": worker_id, "run_idx": i,
                        "ttft_s": float("nan"), "e2e_s": float("nan"),
                        "prompt_tokens": 0, "output_tokens": 0, "decode_tps": 0.0,
                        "error": str(e),
                    })
            time.sleep(INTER_CALL_SLEEP)

    threads = [threading.Thread(target=worker, args=(wid,)) for wid in range(conc)]
    wall_start = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall_end = time.perf_counter()

    return rows, (wall_end - wall_start)


def summarize_conc(rows, conc):
    # only rows for that concurrency (ignore NaNs for percentiles)
    r = [x for x in rows if x["concurrency"] == conc and np.isfinite(x.get("e2e_s", float("nan")))]
    if not r:
        return None
    ttfts = [x["ttft_s"] for x in r]
    e2es = [x["e2e_s"] for x in r]
    out_tps = [x["decode_tps"] for x in r]
    return {
        "concurrency": conc,
        "p50_ttft_s": percentile(ttfts, 50),
        "p95_ttft_s": percentile(ttfts, 95),
        "p50_e2e_s": percentile(e2es, 50),
        "p95_e2e_s": percentile(e2es, 95),
        "p50_decode_tps": percentile(out_tps, 50),
        "p95_decode_tps": percentile(out_tps, 95),
        "n_requests": len(r),
    }


def main():
    # Prepare CSV header
    new_file = not OUT_CSV.exists()
    if new_file:
        with OUT_CSV.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "concurrency","worker_id","run_idx",
                "ttft_s","e2e_s","prompt_tokens","output_tokens","decode_tps","error"
            ])

    all_rows = []
    print(f"Running concurrency levels: {CONCURRENCY_LEVELS}  |  runs/worker={RUNS_PER_WORKER}\n")

    for conc in CONCURRENCY_LEVELS:
        print(f"=== Concurrency {conc} ===")
        rows, wall_s = run_concurrency_level(conc)
        all_rows.extend(rows)

        # Persist rows immediately
        with OUT_CSV.open("a", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow([
                    r.get("concurrency"), r.get("worker_id"), r.get("run_idx"),
                    r.get("ttft_s"), r.get("e2e_s"), r.get("prompt_tokens"),
                    r.get("output_tokens"), r.get("decode_tps"), r.get("error", "")
                ])

        # Throughput achieved (successful requests only)
        ok = [r for r in rows if np.isfinite(r.get("e2e_s", float("nan")))]
        achieved = len(ok) / wall_s if wall_s > 0 else 0.0

        summary = summarize_conc(rows, conc)
        if summary:
            print(
                f"TTFT p50={summary['p50_ttft_s']:.3f}s  p95={summary['p95_ttft_s']:.3f}s | "
                f"E2E p50={summary['p50_e2e_s']:.3f}s  p95={summary['p95_e2e_s']:.3f}s | "
                f"Decode p50={summary['p50_decode_tps']:.0f} tok/s | "
                f"Throughput ≈ {achieved:.2f} req/s over {wall_s:.2f}s ({summary['n_requests']} ok)"
            )
        else:
            print("No successful rows to summarize.")

    print(f"\nRaw per-request data saved to: {OUT_CSV.resolve()}")
    print("Tip: feed this CSV to your chart script to plot latency vs concurrency.")
    

if __name__ == "__main__":
    main()
