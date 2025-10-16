import pandas as pd, numpy as np
from pathlib import Path

IN = Path("streaming_vs_nonstream.csv")

def p(s,q): s=pd.to_numeric(s,errors="coerce").dropna(); return float(np.percentile(s,q)) if len(s) else float("nan")

if not IN.exists(): print("Missing CSV:", IN.resolve()); raise SystemExit
df = pd.read_csv(IN)
sum_ = df.groupby("mode").agg(
    p50_ttft_s = ("ttft_s", lambda s: p(s,50)),
    p95_ttft_s = ("ttft_s", lambda s: p(s,95)),
    p50_e2e_s  = ("e2e_s",  lambda s: p(s,50)),
    p95_e2e_s  = ("e2e_s",  lambda s: p(s,95)),
    avg_prompt_tokens=("prompt_tokens","mean"),
    avg_output_tokens=("output_tokens","mean"),
).reset_index()

print("\n=== Streaming vs Non-Streaming ===")
print(sum_.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

s = {r["mode"]: r for _,r in sum_.iterrows()}
if "stream" in s and "nonstream" in s:
    e2e_ratio = s["nonstream"]["p50_e2e_s"] / s["stream"]["p50_e2e_s"] if s["stream"]["p50_e2e_s"]>0 else float("nan")
    print("\nUX lines:")
    print(f"Streaming TTFT p50 {s['stream']['p50_ttft_s']:.2f}s (p95 {s['stream']['p95_ttft_s']:.2f}s).")
    print(f"Non-stream E2E p50 {s['nonstream']['p50_e2e_s']:.2f}s vs stream {s['stream']['p50_e2e_s']:.2f}s → non-stream is ~{e2e_ratio:.1f}× slower to first visible token.")

