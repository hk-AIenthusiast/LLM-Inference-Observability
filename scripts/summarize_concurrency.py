import math
import pandas as pd
import numpy as np
from pathlib import Path

IN_CSV = Path("../data/concurrency_runs.csv")
OUT_CSV = Path("../data/concurrency_summary.csv")

def pct(x, y):
    return 0.0 if y == 0 else (100.0 * x / y)

def p(series, q):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(np.percentile(s, q)) if len(s) else float("nan")

def main():
    if not IN_CSV.exists():
        print(f"Input CSV not found: {IN_CSV.resolve()}")
        return

    df = pd.read_csv(IN_CSV)

    # successful rows = finite e2e/ttft
    df["ok"] = np.isfinite(pd.to_numeric(df["e2e_s"], errors="coerce"))

    groups = []
    for conc, g in df.groupby("concurrency"):
        ok = g[g["ok"]]
        err = g[~g["ok"]]
        total = len(g)

        row = {
            "concurrency": int(conc),
            "requests_total": total,
            "requests_ok": len(ok),
            "requests_error": len(err),
            "success_rate_pct": pct(len(ok), total),

            "p50_ttft_s": p(ok["ttft_s"], 50),
            "p95_ttft_s": p(ok["ttft_s"], 95),

            "p50_e2e_s":  p(ok["e2e_s"], 50),
            "p95_e2e_s":  p(ok["e2e_s"], 95),

            "p50_decode_tps": p(ok["decode_tps"], 50),
            "p95_decode_tps": p(ok["decode_tps"], 95),

            "avg_prompt_tokens": ok.get("prompt_tokens", pd.Series(dtype=float)).mean() if len(ok) else float("nan"),
            "avg_output_tokens": ok.get("output_tokens", pd.Series(dtype=float)).mean() if len(ok) else float("nan"),
        }
        groups.append(row)

    summary = pd.DataFrame(groups).sort_values("concurrency")
    summary.to_csv(OUT_CSV, index=False)

    # ---- Pretty print + UX-ready one-liners ----
    print("\n=== Concurrency Summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== UX-ready lines (paste into report) ===")
    for _, r in summary.iterrows():
        conc = int(r["concurrency"])
        sr   = r["success_rate_pct"]
        ttft50, ttft95 = r["p50_ttft_s"], r["p95_ttft_s"]
        e2e50, e2e95   = r["p50_e2e_s"], r["p95_e2e_s"]
        tps50          = r["p50_decode_tps"]

        print(
            f"@{conc} concurrent: TTFT p50 {ttft50:.2f}s (p95 {ttft95:.2f}s); "
            f"E2E p50 {e2e50:.2f}s (p95 {e2e95:.2f}s); "
            f"decode ~{tps50:.0f} tok/s (p50); success {sr:.1f}%."
        )

    print(f"\nWrote: {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
