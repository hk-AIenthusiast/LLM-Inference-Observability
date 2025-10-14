import pandas as pd

# Load data (adjust path if needed)
df = pd.read_json("../data/cold_warm.jsonl", lines=True)

def q(series): 
    return series.quantile([0.5, 0.95]).rename({0.5:"p50", 0.95:"p95"})

print("TTFT (ms) by cold_start:")
print(df.groupby(["prompt_id","cold_start"])["ttft_ms"].apply(q))

print("\nE2E latency (ms) by cold_start:")
print(df.groupby(["prompt_id","cold_start"])["latency_ms"].apply(q))
