"""
plot_scaling.py

Plot DDP scaling results from the Snakefile_scaling sweep.

The benchmark CSV has one row per run with columns:
    architecture, world_size, base_width, depth, batch_size,
    per_rank_batch_size, tf32, precision, epochs, epoch_time_s,
    total_time_s, accuracy

(plus cnn_width/cnn_depth/tf_width/tf_depth/tf_heads for dual-branch
runs -- those columns will be NaN for resnet1d rows, and the
base_width/depth columns will be NaN for dual rows).

Usage in a notebook:

    from src.plot_scaling import (
        load_results, plot_scaling, plot_efficiency, plot_all_scaling,
    )
    df = load_results("path/to/benchmark_combined_h100.csv")
    plot_all_scaling(df)

Each individual panel function accepts an optional `ax=` argument so
you can compose a custom layout.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(*paths):
    """Load one or more benchmark CSVs and concatenate them.

    Useful when results came in across multiple files (e.g. one CSV per
    Snakemake rule that you cat together). Also normalises boolean
    columns in case pandas read them as strings.
    """
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        for col in ["tf32"]:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].map({"True": True, "False": False}).astype(bool)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    # Drop duplicated header rows that can sneak in when concatenating
    # raw CSVs with `cat` (each file's header becomes a data row).
    if "architecture" in combined.columns:
        combined = combined[combined["architecture"] != "architecture"].copy()
    # Coerce numeric columns back after the header-row purge
    for col in ["world_size", "base_width", "depth", "batch_size",
                "per_rank_batch_size", "epochs", "epoch_time_s",
                "total_time_s", "accuracy"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined.reset_index(drop=True)


def _resnet_label(row):
    return f"ResNet1d (bw={int(row['base_width'])}, d={int(row['depth'])})"


def _config_groups(df):
    """Group rows by (architecture, model config), each group representing
    one scaling curve to plot. Returns a list of (label, sub_df) tuples
    where sub_df is sorted by world_size.
    """
    groups = []
    if "architecture" in df.columns:
        for arch, arch_df in df.groupby("architecture"):
            if arch == "resnet1d":
                key_cols = ["base_width", "depth"]
                label_fn = _resnet_label
            elif arch == "dual":
                key_cols = ["cnn_width", "cnn_depth", "tf_width", "tf_depth", "tf_heads"]
                label_fn = lambda r: (f"Dual (cnn={int(r['cnn_width'])}/{int(r['cnn_depth'])}, "
                                      f"tf={int(r['tf_width'])}/{int(r['tf_depth'])})")
            else:
                continue
            available = [c for c in key_cols if c in arch_df.columns]
            for keys, sub in arch_df.groupby(available):
                sub_sorted = sub.sort_values("world_size")
                label = label_fn(sub_sorted.iloc[0])
                groups.append((label, sub_sorted))
    return groups


def plot_scaling(df, ax=None):
    """Panel 1: epoch time vs world_size for each (architecture, config).

    Lower is better. Adds an ideal-scaling reference line based on the
    world_size=1 baseline of each curve.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    groups = _config_groups(df)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))

    for (label, sub), color in zip(groups, colors):
        ax.plot(sub["world_size"], sub["epoch_time_s"], "o-",
               label=label, color=color, linewidth=2, markersize=8)

        # Ideal scaling reference (only if we have a world_size=1 baseline)
        baseline = sub[sub["world_size"] == 1]
        if len(baseline) > 0:
            t1 = baseline["epoch_time_s"].iloc[0]
            ws = sub["world_size"].values
            ideal = t1 / ws
            ax.plot(ws, ideal, "--", color=color, alpha=0.4, linewidth=1)

    ax.set_xlabel("GPUs (world_size)")
    ax.set_ylabel("Epoch time (s)")
    ax.set_title("Strong scaling: epoch time vs GPU count\n(dashed = ideal linear scaling)",
                fontsize=10, loc="left")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_speedup(df, ax=None):
    """Panel 2: speedup vs world_size, where speedup = t(1) / t(N).

    Ideal scaling is the diagonal y=x. Above the line = super-linear
    (rare but possible due to memory effects); below = sub-linear (the
    common case, due to communication overhead).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    groups = _config_groups(df)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
    max_ws = 1

    for (label, sub), color in zip(groups, colors):
        baseline = sub[sub["world_size"] == 1]
        if len(baseline) == 0:
            continue
        t1 = baseline["epoch_time_s"].iloc[0]
        speedup = t1 / sub["epoch_time_s"]
        ax.plot(sub["world_size"], speedup, "o-",
               label=label, color=color, linewidth=2, markersize=8)
        max_ws = max(max_ws, sub["world_size"].max())

    # Ideal speedup line
    ws_range = np.array([1, max_ws])
    ax.plot(ws_range, ws_range, "k--", alpha=0.4, label="ideal (linear)")

    ax.set_xlabel("GPUs (world_size)")
    ax.set_ylabel("Speedup vs 1 GPU")
    ax.set_title("Strong-scaling speedup\n(above dashed line = super-linear; below = sub-linear)",
                fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_efficiency(df, ax=None):
    """Panel 3: parallel efficiency = speedup / world_size, in percent.

    100% = perfect linear scaling. Lower = communication overhead is
    eating into the speedup. Above 100% = super-linear (something
    unexpected is helping, often a memory or cache effect).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    groups = _config_groups(df)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))

    for (label, sub), color in zip(groups, colors):
        baseline = sub[sub["world_size"] == 1]
        if len(baseline) == 0:
            continue
        t1 = baseline["epoch_time_s"].iloc[0]
        efficiency = 100 * (t1 / sub["epoch_time_s"]) / sub["world_size"]
        ax.plot(sub["world_size"], efficiency, "o-",
               label=label, color=color, linewidth=2, markersize=8)

    ax.axhline(100, color="k", linestyle="--", alpha=0.4,
              label="ideal (100%)")
    ax.set_xlabel("GPUs (world_size)")
    ax.set_ylabel("Parallel efficiency (%)")
    ax.set_title("Parallel efficiency\n(how much of each GPU's potential we use)",
                fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_all_scaling(df, figsize=(8, 14)):
    """Render the three scaling panels in one stacked figure.

    Returns the Figure so it can be saved/tweaked further in a notebook.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    plot_scaling(df, ax=axes[0])
    plot_speedup(df, ax=axes[1])
    plot_efficiency(df, ax=axes[2])
    fig.tight_layout()
    return fig


def summary_table(df):
    """Tidy DataFrame summarising results (one row per (config, world_size)).

    Convenient for displaying inline in the notebook above the plots.
    """
    rows = []
    for label, sub in _config_groups(df):
        baseline = sub[sub["world_size"] == 1]
        t1 = baseline["epoch_time_s"].iloc[0] if len(baseline) else None
        for _, r in sub.iterrows():
            t = r["epoch_time_s"]
            ws = int(r["world_size"])
            speedup = (t1 / t) if t1 is not None else None
            efficiency = (100 * speedup / ws) if speedup is not None else None
            rows.append({
                "config": label,
                "world_size": ws,
                "epoch_time_s": round(t, 2),
                "speedup": round(speedup, 2) if speedup else None,
                "efficiency_%": round(efficiency, 1) if efficiency else None,
                "accuracy": round(r["accuracy"], 4) if pd.notna(r["accuracy"]) else None,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["benchmark_combined.csv"]
    df = load_results(*paths)
    print(summary_table(df).to_string(index=False))
    fig = plot_all_scaling(df)
    out = "scaling_plots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {out}")

