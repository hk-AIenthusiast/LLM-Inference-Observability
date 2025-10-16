import pandas as pd, numpy as np
from pathlib import Path

IN = Path("output_length_complexity.csv")
def p(s,q): s=pd.to_numeric(s,errors="coerce").dropna(); return float(np.percentile(s,q)) if len(s) else float("nan")

if not IN.exists(): print("Missing CSV:", IN.resolve()); raise SystemExit
df = pd.read_csv(IN)

g = df.groupby(["task","num_predict"]).agg(
    p50_ttft_s=("ttft_s", lambda s: p(s,50)),
    p95_ttft_s=("ttft_s", lambda s: p(s,95)),
    p50_e2e_s =("e2e_s",  lambda s: p(s,50)),
    p95_e2e_s =("e2e_s",  lambda s: p(s,95)),
    p50_tps   =("decode_tps", lambda s: p(s,50)),
    avg_out_tokens=("output_tokens","mean"),
).reset_index().sort_values(["task","num_predict"])

print("\n=== Output Length & Task Complexity ===")
print(g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\nUX lines:")
for _,r in g.iterrows():
    print(f"{r['task']} (num_predict={int(r['num_predict'])}): "
          f"E2E p50 {r['p50_e2e_s']:.2f}s (p95 {r['p95_e2e_s']:.2f}s), "
          f"TTFT p50 {r['p50_ttft_s']:.2f}s, "
          f"~{r['avg_out_tokens']:.0f} output tokens at ~{r['p50_tps']:.0f} tok/s.")
