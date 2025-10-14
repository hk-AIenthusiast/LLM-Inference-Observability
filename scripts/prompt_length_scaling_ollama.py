import time
import json
import csv
from pathlib import Path
import numpy as np
import requests

# -------------------- CONFIG --------------------
MODEL = "llama3"                 # any local Ollama model you’ve pulled
HOST = "http://localhost:11434"
OUT_CSV = Path("prompt_length_scaling.csv")

# Prompts for scaling. Replace LONG/XLONG with your real long texts when ready.
PROMPTS = {
    "short":  "Say hi in one short sentence.",
    "medium": "Explain retrieval-augmented generation (RAG) in 3–4 sentences, focusing on why it reduces hallucinations and how retrieval + generation work together.",
    "long":   "Provide a detailed explanation of RAG covering ingestion, chunking, embedding models, vector databases, hybrid search, reranking, and evaluation metrics such as Recall@K and faithfulness. Include typical pitfalls and mitigation strategies in 6–8 sentences.",
    "xlong":  (
        "Provide a comprehensive, multi-paragraph overview of Retrieval-Augmented Generation (RAG). "
        "Discuss data pipelines (ingestion, chunking strategies like 200–500 tokens with overlap), embedding model selection "
        "(domain-specific vs general), indexing in vector stores (FAISS/Chroma/Pinecone), hybrid lexical + dense retrieval, "
        "reranking with cross-encoders, and multi-hop/agentic retrieval. Explain latency/throughput tradeoffs, caching, and "
        "how prompt size/response size impact TTFT and E2E. Include guidelines for guardrails, citations, and when RAG is "
        "overkill. Conclude with deployment tips (warm pools, keep-alive) and evaluation approaches (human eval, groundedness). "
        * 3  # repeated to lengthen — replace with real long text for production
    )
}

RUNS_PER_SIZE = 30               # warm runs per prompt size
INTER_CALL_SLEEP = 0.12          # small gap between requests
KEEP_ALIVE = "10m"               # keep model warm across calls
# ------------------------------------------------


def chat_stream_ollama(model: str, prompt: str, keep_alive):
    """
    Streaming chat request to Ollama with TTFT/E2E and token/latency metrics.

    Returns a dict with:
      ttft_s, e2e_s,
      prompt_tokens, output_tokens,
      decode_tps (output tokens / decode time),
      total_duration_s, prompt_eval_duration_s, eval_duration_s
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

    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Save last event (final 'done' event carries metrics)
            last_data = data

            # First content chunk → TTFT
            msg = data.get("message", {})
            content = msg.get("content", "")
            if content and first_token_time is None:
                first_token_time = time.perf_counter()

            if data.get("done"):
                break

    end = time.perf_counter()
    if first_token_time is None:
        first_token_time = end

    # Metrics from final event (ns → s)
    prompt_tokens = int(last_data.get("prompt_eval_count", 0))
    output_tokens = int(last_data.get("eval_count", 0))
    total_duration_s = float(last_data.get("total_duration", 0) / 1e9)
    prompt_eval_duration_s = float(last_data.get("prompt_eval_duration", 0) / 1e9)
    eval_duration_s = float(last_data.get("eval_duration", 0) / 1e9)

    # Throughput: decoding only (tokens/sec when generating output)
    decode_tps = (output_tokens / eval_duration_s) if eval_duration_s > 0 else 0.0

    return {
        "ttft_s": first_token_time - start,
        "e2e_s": end - start,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "decode_tps": decode_tps,
        "total_duration_s": total_duration_s,
        "prompt_eval_duration_s": prompt_eval_duration_s,
        "eval_duration_s": eval_duration_s,
    }


def percentile(arr, p):
    return float(np.percentile(np.array(arr, dtype=float), p)) if arr else 0.0


def run_batch(size_label: str, prompt: str, runs: int):
    # Warm-up (not measured)
    _ = chat_stream_ollama(MODEL, "Warm up please. Say OK.", keep_alive=KEEP_ALIVE)

    rows = []
    for i in range(1, runs + 1):
        m = chat_stream_ollama(MODEL, prompt, keep_alive=KEEP_ALIVE)
        rows.append(m)
        print(
            f"[{size_label}] {i}/{runs}  "
            f"TTFT={m['ttft_s']:.3f}s  E2E={m['e2e_s']:.3f}s  "
            f"prompt={m['prompt_tokens']} tok  output={m['output_tokens']} tok  "
            f"decode={m['decode_tps']:.1f} tok/s"
        )
        time.sleep(INTER_CALL_SLEEP)
    return rows


def summarize(size_label: str, rows):
    ttfts = [r["ttft_s"] for r in rows]
    e2es = [r["e2e_s"] for r in rows]
    dec_tps = [r["decode_tps"] for r in rows]
    in_tok = [r["prompt_tokens"] for r in rows]
    out_tok = [r["output_tokens"] for r in rows]

    summary = {
        "label": size_label,
        "p50_ttft_s": percentile(ttfts, 50),
        "p95_ttft_s": percentile(ttfts, 95),
        "p50_e2e_s": percentile(e2es, 50),
        "p95_e2e_s": percentile(e2es, 95),
        "p50_decode_tps": percentile(dec_tps, 50),
        "p95_decode_tps": percentile(dec_tps, 95),
        "avg_prompt_tokens": float(np.mean(in_tok)) if in_tok else 0.0,
        "avg_output_tokens": float(np.mean(out_tok)) if out_tok else 0.0,
    }

    print(
        f"\n=== {size_label} (avg prompt tok ~{summary['avg_prompt_tokens']:.0f}) ===\n"
        f"TTFT   p50={summary['p50_ttft_s']:.3f}s   p95={summary['p95_ttft_s']:.3f}s\n"
        f"E2E    p50={summary['p50_e2e_s']:.3f}s    p95={summary['p95_e2e_s']:.3f}s\n"
        f"Decode throughput   p50={summary['p50_decode_tps']:.1f} tok/s   p95={summary['p95_decode_tps']:.1f} tok/s\n"
    )
    return summary


def main():
    # Prepare CSV
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "size","run_idx","ttft_s","e2e_s",
                "prompt_tokens","output_tokens","decode_tps",
                "total_duration_s","prompt_eval_duration_s","eval_duration_s"
            ])

    all_summaries = []

    for size, prompt in PROMPTS.items():
        rows = run_batch(size, prompt, RUNS_PER_SIZE)

        # append raw rows
        with OUT_CSV.open("a", newline="") as f:
            w = csv.writer(f)
            for i, r in enumerate(rows, start=1):
                w.writerow([
                    size, i, f"{r['ttft_s']:.6f}", f"{r['e2e_s']:.6f}",
                    r["prompt_tokens"], r["output_tokens"], f"{r['decode_tps']:.3f}",
                    f"{r['total_duration_s']:.6f}", f"{r['prompt_eval_duration_s']:.6f}", f"{r['eval_duration_s']:.6f}",
                ])

        summary = summarize(size, rows)
        all_summaries.append(summary)

    # UX-ready comparison lines
    print("=== UX-ready summary ===")
    for s in all_summaries:
        print(
            f"{s['label']}: TTFT p50 {s['p50_ttft_s']:.2f}s (p95 {s['p95_ttft_s']:.2f}s), "
            f"E2E p50 {s['p50_e2e_s']:.2f}s (p95 {s['p95_e2e_s']:.2f}s), "
            f"Decode ~{s['p50_decode_tps']:.0f} tok/s (p50), prompt≈{s['avg_prompt_tokens']:.0f} tok."
        )

    print(f"\nRaw data saved to: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
