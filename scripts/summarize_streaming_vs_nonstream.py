import pandas as pd, numpy as np
from pathlib import Path

IN  = Path("../data/streaming_vs_nonstream.csv")
OUT = Path("../data/streaming_vs_nonstream_summary.csv")

def p(series, q):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(np.percentile(s, q)) if len(s) else float("nan")

def main():
    if not IN.exists():
        print(f"Missing CSV: {IN.resolve()}"); return

    df = pd.read_csv(IN)
    # normalize headers just in case
    df.columns = [c.strip().lower() for c in df.columns]

    # Coerce numerics
    for c in ("ttft_s","e2e_s","prompt_tokens","output_tokens","run_idx"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Summaries ---
    groups = []
    for mode, g in df.groupby("mode"):
        row = {
            "mode": mode,
            "n_runs": len(g),
            "p50_e2e_s": p(g["e2e_s"], 50),
            "p95_e2e_s": p(g["e2e_s"], 95),
            "avg_prompt_tokens": float(g["prompt_tokens"].mean()),
            "avg_output_tokens": float(g["output_tokens"].mean()),
        }
        # TTFT only meaningful for streaming rows
        if mode == "stream":
            row.update({
                "p50_ttft_s": p(g["ttft_s"], 50),
                "p95_ttft_s": p(g["ttft_s"], 95),
            })
        else:
            row.update({"p50_ttft_s": float("nan"), "p95_ttft_s": float("nan")})
        groups.append(row)

    summary = pd.DataFrame(groups).sort_values("mode")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)

    # --- Print table + UX lines ---
    print("\n=== Streaming vs Non-Streaming Summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote: {OUT.resolve()}")

    # UX one-liners
    s = {r["mode"]: r for _, r in summary.iterrows()}
    if "stream" in s and "nonstream" in s:
        e2e_ratio = s["nonstream"]["p50_e2e_s"] / s["stream"]["p50_e2e_s"] if s["stream"]["p50_e2e_s"] > 0 else float("nan")
        print("\nUX lines:")
        if pd.notna(s["stream"]["p50_ttft_s"]):
            print(f"Streaming: TTFT p50 {s['stream']['p50_ttft_s']:.2f}s (p95 {s['stream']['p95_ttft_s']:.2f}s).")
        print(f"E2E: non-stream p50 {s['nonstream']['p50_e2e_s']:.2f}s vs stream {s['stream']['p50_e2e_s']:.2f}s "
              f"→ non-stream ≈ {e2e_ratio:.1f}× slower to first visible token (no incremental streaming).")

if __name__ == "__main__":
    main()
