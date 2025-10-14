#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "cold_warm.jsonl"
OUT_PATH = Path(__file__).parent.parent / "data" / "cold_warm_summary.csv"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_json(DATA_PATH, lines=True)

    quantiles = [0.5, 0.95]
    q_labels = {0.5: "p50", 0.95: "p95"}

    results = []

    for metric in ["ttft_ms", "latency_ms"]:
        if metric not in df.columns:
            continue
        # Group by prompt × cold_start
        grouped = df.groupby(["prompt_id", "cold_start"])[metric]
        for (prompt_id, cold_start), series in grouped:
            q_vals = series.quantile(quantiles)
            results.append({
                "metric": metric,
                "prompt_id": prompt_id,
                "cold_start": cold_start,
                "p50": round(q_vals[0.5], 1),
                "p95": round(q_vals[0.95], 1)
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"✅ Summary written to: {OUT_PATH}\n")
    print(out_df)

if __name__ == "__main__":
    main()
