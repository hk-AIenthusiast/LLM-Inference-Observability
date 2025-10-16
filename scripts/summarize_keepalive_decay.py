import pandas as pd
from pathlib import Path

IN = Path("keepalive_decay.csv")
if not IN.exists(): print("Missing CSV:", IN.resolve()); raise SystemExit
df = pd.read_csv(IN)
df["idle_s"]=pd.to_numeric(df["idle_s"]); df["ttft_s"]=pd.to_numeric(df["ttft_s"]); df["e2e_s"]=pd.to_numeric(df["e2e_s"])

print("\n=== Keep-Alive / Idle Decay ===")
print(df.sort_values("idle_s").to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Heuristic: find first idle where TTFT jumps >3× the minimum (indicates unload)
baseline = df["ttft_s"].min()
threshold = baseline * 3.0
candidates = df[df["ttft_s"] > threshold].sort_values("idle_s")
if len(candidates):
    t = int(candidates.iloc[0]["idle_s"])
    print(f"\nUX line: Model appears to unload after ~{t}s of inactivity (TTFT jumps from ~{baseline:.2f}s to {candidates.iloc[0]['ttft_s']:.2f}s).")
else:
    print("\nUX line: Model remained warm across tested idle windows; TTFT stayed near baseline.")
