import time
import json
import csv
from pathlib import Path
from statistics import median
import numpy as np
import requests

# -------------------- CONFIG --------------------
MODEL = "llama3"              # any local Ollama model (e.g., "llama3", "phi3", "mistral")
HOST = "http://localhost:11434"
OUT_CSV = Path("latency_runs.csv")

# Prompts for your short vs medium test (edit as you like)
PROMPTS = {
    "short":  "Say hi in one short sentence.",
    "medium": "Explain, in ~5 sentences, what retrieval-augmented generation (RAG) is and when to use it."
}

# Experiment plan
COLD_RUNS = 10
WARM_RUNS = 50

# Small delay between calls to avoid overlap (seconds)
INTER_CALL_SLEEP = 0.15
# ------------------------------------------------


def chat_stream_ollama(model: str, prompt: str, keep_alive) -> tuple[float, float]:
    """
    Send a streaming chat request to Ollama and measure:
      - TTFT (time to first token)
      - E2E (end-to-end) time
    keep_alive:
      - 0        -> unload immediately (forces "cold" each time)
      - '10m'    -> keep model hot for 10 minutes (good for warm batches)
    Returns:
      (ttft_seconds, e2e_seconds)
    """
    url = f"{HOST}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "keep_alive": keep_alive
    }

    # Start the clock
    start = time.perf_counter()
    first_token_time = None

    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Occasionally a partial line can appear; skip it.
                continue

            # The first content chunk marks TTFT
            msg = data.get("message", {})
            content = msg.get("content", "")
            if content and first_token_time is None:
                first_token_time = time.perf_counter()

            # The stream ends when "done": true arrives
            if data.get("done"):
                break

    end = time.perf_counter()
    if first_token_time is None:
        # No tokens? Treat TTFT as E2E to avoid crashing; usually means empty response.
        first_token_time = end

    ttft = first_token_time - start
    e2e = end - start
    return ttft, e2e


def percentile(arr, p):
    # numpy.percentile handles small arrays well
    return float(np.percentile(np.array(arr, dtype=float), p))


def run_batch(label: str, prompt: str, runs: int, keep_alive):
    """
    Run N requests and collect TTFT / E2E for a given prompt & keep_alive mode.
    """
    ttfts, e2es = [], []
    for i in range(1, runs + 1):
        ttft, e2e = chat_stream_ollama(MODEL, prompt, keep_alive)
        ttfts.append(ttft)
        e2es.append(e2e)
        print(f"[{label}] Run {i}/{runs}  TTFT={ttft:.3f}s  E2E={e2e:.3f}s")
        time.sleep(INTER_CALL_SLEEP)
    return ttfts, e2es


def summarize(label: str, ttfts, e2es):
    p50_ttft = percentile(ttfts, 50)
    p95_ttft = percentile(ttfts, 95)
    p50_e2e  = percentile(e2es, 50)
    p95_e2e  = percentile(e2es, 95)
    print(f"\n=== {label} ===")
    print(f"TTFT  p50={p50_ttft:.3f}s  p95={p95_ttft:.3f}s")
    print(f"E2E   p50={p50_e2e:.3f}s   p95={p95_e2e:.3f}s")
    return p50_ttft, p95_ttft, p50_e2e, p95_e2e


def main():
    # Ensure output CSV exists with header
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["phase","prompt_len","run_idx","ttft_s","e2e_s"])

    overall_summary = []

    # ---------- COLD runs (force unload every call) ----------
    for name, prompt in PROMPTS.items():
        label = f"cold/{name}"
        ttfts, e2es = run_batch(label, prompt, COLD_RUNS, keep_alive=0)

        # Append raw runs to CSV
        with OUT_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            for i, (t, e) in enumerate(zip(ttfts, e2es), start=1):
                writer.writerow(["cold", name, i, f"{t:.6f}", f"{e:.6f}"])

        p50_ttft, p95_ttft, p50_e2e, p95_e2e = summarize(label, ttfts, e2es)
        overall_summary.append((label, p50_ttft, p95_ttft, p50_e2e, p95_e2e))

    # ---------- WARM runs (keep model in memory between calls) ----------
    # Warm-up priming call (not counted)
    _ = chat_stream_ollama(MODEL, "Warm up, just say OK.", keep_alive="10m")

    for name, prompt in PROMPTS.items():
        label = f"warm/{name}"
        ttfts, e2es = run_batch(label, prompt, WARM_RUNS, keep_alive="10m")

        # Append raw runs to CSV
        with OUT_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            for i, (t, e) in enumerate(zip(ttfts, e2es), start=1):
                writer.writerow(["warm", name, i, f"{t:.6f}", f"{e:.6f}"])

        p50_ttft, p95_ttft, p50_e2e, p95_e2e = summarize(label, ttfts, e2es)
        overall_summary.append((label, p50_ttft, p95_ttft, p50_e2e, p95_e2e))

    # ---------- Pretty comparison rows you can paste into your UX table ----------
    print("\n=== UX-ready summary rows ===")
    # Build side-by-side cold vs warm statements for TTFT and E2E, per prompt size
    # Find helpers
    def get(label):
        for row in overall_summary:
            if row[0] == label:
                return row
        return None

    for name in PROMPTS.keys():
        cold = get(f"cold/{name}")
        warm = get(f"warm/{name}")
        if not (cold and warm):
            continue

        cold_p50_ttft = cold[1]; warm_p50_ttft = warm[1]
        cold_p50_e2e  = cold[3]; warm_p50_e2e  = warm[3]

        ttft_ratio = (cold_p50_ttft / warm_p50_ttft) if warm_p50_ttft > 0 else float("inf")
        e2e_ratio  = (cold_p50_e2e / warm_p50_e2e) if warm_p50_e2e > 0 else float("inf")

        print(
            f"{name} prompt — "
            f"TTFT p50 cold = {cold_p50_ttft:.2f}s, warm = {warm_p50_ttft:.2f}s → cold is ~{ttft_ratio:.1f}× slower. "
            f"E2E p50 cold = {cold_p50_e2e:.2f}s, warm = {warm_p50_e2e:.2f}s → cold is ~{e2e_ratio:.1f}× slower."
        )

    print(f"\nRaw run data saved to: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
