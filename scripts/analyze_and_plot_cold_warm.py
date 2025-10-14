#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).parent.parent / "data" / "cold_warm.jsonl"   # change if needed
OUT_DIR = Path(__file__).parent.parent / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    # Normalize types a bit
    if "cold_start" in df:
        df["cold_start"] = df["cold_start"].astype(bool)
    if "prompt_id" not in df:
        df["prompt_id"] = "default"
    return df

def print_quantiles(df: pd.DataFrame, col: str, title: str):
    if col not in df:
        print(f"[WARN] Column {col} not found; skipping quantiles for {title}")
        return
    def q(series):
        return series.quantile([0.5, 0.95]).rename({0.5: "p50", 0.95: "p95"})
    print(f"\n{title} by prompt_id × cold_start:")
    print(df.groupby(["prompt_id", "cold_start"])[col].apply(q))

def boxplot_by_cold(df: pd.DataFrame, col: str, fname: str, title: str, ylabel: str):
    """One chart per metric. X-axis grouped by prompt_id & cold/warm."""
    if col not in df:
        print(f"[WARN] Column {col} not found; skipping plot {fname}")
        return
    # If column has invalid sentinel (-1), drop them for plotting
    plot_df = df[df[col] >= 0].copy()
    if plot_df.empty:
        print(f"[WARN] No valid data to plot for {col}")
        return

    # Build categories like "short (cold)" / "short (warm)"
    plot_df["cat"] = plot_df.apply(
        lambda r: f"{r.get('prompt_id','prompt')}\n({'cold' if r['cold_start'] else 'warm'})", axis=1
    )

    # Order: group by prompt, cold then warm
    order = []
    for pid in plot_df["prompt_id"].unique():
        order.extend([f"{pid}\n(cold)", f"{pid}\n(warm)"])

    data = [plot_df.loc[plot_df["cat"] == k, col].values for k in order if k in plot_df["cat"].unique()]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, showfliers=False)
    plt.xticks(range(1, len(data) + 1), [k for k in order if k in plot_df["cat"].unique()])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Prompt × State")
    plt.tight_layout()
    out_path = OUT_DIR / fname
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = load_df(DATA_PATH)

    # Print summary quantiles
    print_quantiles(df, "ttft_ms", "TTFT (ms)")
    print_quantiles(df, "latency_ms", "E2E latency (ms)")

    # Plots (skip TTFT if it looks like a non-stream dataset)
    has_ttft = "ttft_ms" in df.columns and (df["ttft_ms"] >= 0).any()

    if has_ttft:
        boxplot_by_cold(
            df, col="ttft_ms",
            fname="cold_warm_ttft_boxplot.png",
            title="TTFT (ms): Cold vs Warm (by Prompt)",
            ylabel="TTFT (ms)"
        )

    boxplot_by_cold(
        df, col="latency_ms",
        fname="cold_warm_e2e_boxplot.png",
        title="E2E Latency (ms): Cold vs Warm (by Prompt)",
        ylabel="E2E Latency (ms)"
    )

if __name__ == "__main__":
    main()
