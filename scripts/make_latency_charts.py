import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = Path("charts")
OUTDIR.mkdir(exist_ok=True)

def savefig(name: str):
    path = OUTDIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path.resolve()}")

# ------------- Prompt-length scaling -------------
def charts_prompt_length_scaling(csv_path="prompt_length_scaling.csv"):
    if not Path(csv_path).exists():
        print(f"[prompt-length] {csv_path} not found. Skipping.")
        return

    df = pd.read_csv(csv_path)
    req_cols = {"size","run_idx","ttft_s","e2e_s","prompt_tokens","output_tokens","decode_tps"}
    if not req_cols.issubset(df.columns):
        print(f"[prompt-length] CSV missing required columns; found: {df.columns.tolist()}")
        return

    # Summary stats
    g = df.groupby("size")
    summary = pd.DataFrame({
        "p50_ttft_s": g["ttft_s"].median(),
        "p95_ttft_s": g["ttft_s"].quantile(0.95),
        "p50_e2e_s":  g["e2e_s"].median(),
        "p95_e2e_s":  g["e2e_s"].quantile(0.95),
        "p50_decode_tps": g["decode_tps"].median(),
        "p95_decode_tps": g["decode_tps"].quantile(0.95),
        "avg_prompt_tokens": g["prompt_tokens"].mean(),
        "avg_output_tokens": g["output_tokens"].mean(),
    }).reset_index()

    # Bar: p50/p95 TTFT
    x = np.arange(len(summary["size"]))
    width = 0.35
    plt.figure(figsize=(8,4.5))
    plt.bar(x - width/2, summary["p50_ttft_s"], width, label="p50")
    plt.bar(x + width/2, summary["p95_ttft_s"], width, label="p95")
    plt.xticks(x, summary["size"])
    plt.ylabel("TTFT (s)")
    plt.title("TTFT by Prompt Size (p50/p95)")
    plt.legend()
    savefig("scaling_ttft_p50_p95.png")

    # Bar: p50/p95 E2E
    plt.figure(figsize=(8,4.5))
    plt.bar(x - width/2, summary["p50_e2e_s"], width, label="p50")
    plt.bar(x + width/2, summary["p95_e2e_s"], width, label="p95")
    plt.xticks(x, summary["size"])
    plt.ylabel("E2E latency (s)")
    plt.title("E2E by Prompt Size (p50/p95)")
    plt.legend()
    savefig("scaling_e2e_p50_p95.png")

    # Boxplots: TTFT / E2E by size
    order = summary["size"].tolist()
    plt.figure(figsize=(8,4.5))
    df.boxplot(column="ttft_s", by="size", grid=False)
    plt.suptitle("")
    plt.title("TTFT distribution by size")
    plt.xlabel("size"); plt.ylabel("TTFT (s)")
    savefig("scaling_ttft_box.png")

    plt.figure(figsize=(8,4.5))
    df.boxplot(column="e2e_s", by="size", grid=False)
    plt.suptitle("")
    plt.title("E2E distribution by size")
    plt.xlabel("size"); plt.ylabel("E2E (s)")
    savefig("scaling_e2e_box.png")

    # Scatter: prompt_tokens vs TTFT / E2E
    plt.figure(figsize=(8,4.5))
    for s in order:
        sub = df[df["size"]==s]
        plt.scatter(sub["prompt_tokens"], sub["ttft_s"], alpha=0.7, label=s)
    plt.xlabel("Prompt tokens"); plt.ylabel("TTFT (s)")
    plt.title("TTFT vs Prompt Tokens")
    plt.legend()
    savefig("scaling_scatter_prompt_tokens_ttft.png")

    plt.figure(figsize=(8,4.5))
    for s in order:
        sub = df[df["size"]==s]
        plt.scatter(sub["prompt_tokens"], sub["e2e_s"], alpha=0.7, label=s)
    plt.xlabel("Prompt tokens"); plt.ylabel("E2E (s)")
    plt.title("E2E vs Prompt Tokens")
    plt.legend()
    savefig("scaling_scatter_prompt_tokens_e2e.png")

    # Boxplot: decode throughput by size
    plt.figure(figsize=(8,4.5))
    df.boxplot(column="decode_tps", by="size", grid=False)
    plt.suptitle("")
    plt.title("Decode Throughput by Size")
    plt.xlabel("size"); plt.ylabel("tokens/sec")
    savefig("scaling_decode_tps_box.png")

    # Save summary CSV for your report
    summary.to_csv(OUTDIR / "scaling_summary.csv", index=False)
    print(f"Saved: {(OUTDIR / 'scaling_summary.csv').resolve()}")

# ------------- Cold vs Warm -------------
def charts_cold_warm():
    path_raw = Path("latency_runs.csv")   # from the earlier harness
    path_sum = Path("cold_warm_summary.csv")  # optional: your summarized file
    path_alt = Path("results_summary.csv")    # or the custom one you showed

    if path_raw.exists():
        df = pd.read_csv(path_raw)
        # Expecting columns: phase (cold/warm), prompt_len (short/medium), run_idx, ttft_s, e2e_s
        if {"phase","prompt_len","run_idx","ttft_s","e2e_s"}.issubset(df.columns):
            g = df.groupby(["prompt_len","phase"])
            summary = g.agg(
                p50_ttft_s=("ttft_s","median"),
                p95_ttft_s=("ttft_s",lambda s: s.quantile(0.95)),
                p50_e2e_s=("e2e_s","median"),
                p95_e2e_s=("e2e_s",lambda s: s.quantile(0.95)),
            ).reset_index()

            # Bar charts: TTFT and E2E
            # Pivot for plotting
            piv_ttft = summary.pivot(index="prompt_len", columns="phase", values="p50_ttft_s")
            piv_e2e  = summary.pivot(index="prompt_len", columns="phase", values="p50_e2e_s")

            ax = piv_ttft.plot(kind="bar", figsize=(8,4.5))
            ax.set_ylabel("TTFT p50 (s)")
            ax.set_title("Cold vs Warm TTFT (p50) by Prompt Length")
            savefig("coldwarm_ttft_p50_bar.png")

            ax = piv_e2e.plot(kind="bar", figsize=(8,4.5))
            ax.set_ylabel("E2E p50 (s)")
            ax.set_title("Cold vs Warm E2E (p50) by Prompt Length")
            savefig("coldwarm_e2e_p50_bar.png")

            # Boxplots: distributions if desired
            plt.figure(figsize=(8,4.5))
            df["phase_len"] = df["phase"] + " / " + df["prompt_len"]
            df.boxplot(column="ttft_s", by="phase_len", grid=False)
            plt.suptitle("")
            plt.title("TTFT distribution (cold/warm × length)")
            plt.xlabel("group"); plt.ylabel("TTFT (s)")
            savefig("coldwarm_ttft_box.png")

            plt.figure(figsize=(8,4.5))
            df.boxplot(column="e2e_s", by="phase_len", grid=False)
            plt.suptitle("")
            plt.title("E2E distribution (cold/warm × length)")
            plt.xlabel("group"); plt.ylabel("E2E (s)")
            savefig("coldwarm_e2e_box.png")

            summary.to_csv(OUTDIR / "coldwarm_summary_from_raw.csv", index=False)
            print(f"Saved: {(OUTDIR / 'coldwarm_summary_from_raw.csv').resolve()}")
            return

    # If we get here, try summarized format like the one you pasted
    candidate = None
    for p in [path_sum, path_alt, Path("summary.csv")]:
        if p.exists():
            candidate = p; break
    if candidate is None:
        print("[cold-warm] No cold/warm CSV found. Skipping.")
        return

    df = pd.read_csv(candidate)
    if {"metric","prompt_id","cold_start","p50","p95"}.issubset(df.columns):
        # Create separate charts for ttft_ms and latency_ms
        for metric in ["ttft_ms","latency_ms"]:
            sub = df[df["metric"]==metric].copy()
            sub["group"] = sub["prompt_id"].astype(str) + " / " + sub["cold_start"].map({True:"cold", False:"warm"})
            sub = sub.sort_values("group")

            # p50 and p95 bars (values are ms in this file)
            x = np.arange(len(sub["group"]))
            width = 0.35

            plt.figure(figsize=(9,4.5))
            plt.bar(x - width/2, sub["p50"]/1000.0, width, label="p50")
            plt.bar(x + width/2, sub["p95"]/1000.0, width, label="p95")
            plt.xticks(x, sub["group"], rotation=20, ha="right")
            ylabel = "TTFT (s)" if metric=="ttft_ms" else "E2E (s)"
            title  = "TTFT p50/p95 (cold vs warm)" if metric=="ttft_ms" else "E2E p50/p95 (cold vs warm)"
            plt.ylabel(ylabel); plt.title(title); plt.legend()
            fname = "coldwarm_ttft_p50p95_from_summary.png" if metric=="ttft_ms" else "coldwarm_e2e_p50p95_from_summary.png"
            savefig(fname)
    else:
        print(f"[cold-warm] Unrecognized columns in {candidate}: {df.columns.tolist()}")

def main():
    charts_prompt_length_scaling()
    charts_cold_warm()
    print("\nAll done. Charts saved under:", OUTDIR.resolve())

if __name__ == "__main__":
    main()
