#!/usr/bin/env python3
"""Generate paper tables and figures from MCLTA evaluation artifacts.

Usage:
    uv run python scripts/generate_mclta_artifacts.py --results-dir results/mclta --output-dir paper/mclta_artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
matplotlib.use("Agg")


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all CSV run files from results directory."""
    csv_files = list(results_dir.glob("*_runs.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_runs.csv files found in {results_dir}")
    dfs = [pd.read_csv(f) for f in csv_files]
    return pd.concat(dfs, ignore_index=True)


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-dataset summary table for the paper."""
    rows = []
    for ds, grp in df.groupby("dataset"):
        completed = grp[~grp["status"].str.startswith("ERROR") & (grp["status"] != "INVALID")]
        n_total = len(grp)
        n_done = len(completed)
        min_cert = completed[completed["status"] == "MINIMUM_SOLVER_CERTIFIED"]
        near_cert = completed[
            completed["status"].isin(["MINIMUM_SOLVER_CERTIFIED", "NEAR_MINIMUM_SOLVER_CERTIFIED"])
        ]
        rows.append({
            "Dataset": ds,
            "Runs": n_total,
            "Completion": f"{n_done / n_total:.0%}" if n_total > 0 else "—",
            "Min-cert": f"{len(min_cert) / n_done:.0%}" if n_done > 0 else "—",
            "Near-min-cert": f"{len(near_cert) / n_done:.0%}" if n_done > 0 else "—",
            "Med. selected": f"{completed['n_selected'].median():.1f}" if n_done > 0 else "—",
            "Med. atoms": f"{completed['n_atoms'].median():.1f}" if n_done > 0 else "—",
            "Med. runtime (s)": f"{completed['runtime_s'].median():.1f}" if n_done > 0 else "—",
        })
    return pd.DataFrame(rows)


def make_compression_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table comparing selected clauses vs total atoms."""
    rows = []
    for ds, grp in df.groupby("dataset"):
        completed = grp[
            ~grp["status"].str.startswith("ERROR") &
            (grp["status"] != "INVALID") &
            (grp["n_atoms"] > 0)
        ]
        if len(completed) == 0:
            continue
        compression = completed["n_selected"] / completed["n_atoms"]
        rows.append({
            "Dataset": ds,
            "Med. n_atoms": completed["n_atoms"].median(),
            "Med. n_selected": completed["n_selected"].median(),
            "Med. compression": compression.median(),
        })
    return pd.DataFrame(rows)


def plot_direction_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of direction signature distribution."""
    # Count statuses
    status_counts = df["status"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    status_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_xlabel("Atlas Status")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Atlas Statuses")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_runtime_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Histogram of per-run runtimes."""
    fig, ax = plt.subplots(figsize=(6, 4))
    runtimes = df["runtime_s"].dropna()
    ax.hist(runtimes, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(runtimes.median(), color="red", linestyle="--", label=f"Median={runtimes.median():.1f}s")
    ax.set_xlabel("Runtime per Atlas (s)")
    ax.set_ylabel("Count")
    ax.set_title("MCLTA Runtime Distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_compression_ratio(df: pd.DataFrame, output_path: Path) -> None:
    """Box plot of compression ratio (n_selected / n_atoms) per dataset."""
    valid = df[(df["n_atoms"] > 0) & ~df["status"].str.startswith("ERROR")].copy()
    valid["compression"] = valid["n_selected"] / valid["n_atoms"]

    datasets = valid["dataset"].unique()
    data_by_ds = [valid[valid["dataset"] == ds]["compression"].values for ds in datasets]

    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.5), 4))
    ax.boxplot(data_by_ds, tick_labels=datasets)
    ax.set_ylabel("Compression ratio (n_selected / n_atoms)")
    ax.set_title("MCLTA Clause Compression vs Exhaustive Atoms")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="No compression")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_latex_table(df: pd.DataFrame, output_path: Path, caption: str, label: str) -> None:
    """Save a DataFrame as a LaTeX table."""
    latex = df.to_latex(index=False, caption=caption, label=label, escape=True)
    output_path.write_text(latex)
    print(f"Saved LaTeX table: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate MCLTA paper artifacts.")
    parser.add_argument("--results-dir", default="results/mclta",
                        help="Directory containing *_runs.csv files")
    parser.add_argument("--output-dir", default="paper/mclta_artifacts",
                        help="Directory for generated artifacts")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    df = load_results(results_dir)
    print(f"Loaded {len(df)} run records from {df['dataset'].nunique()} datasets")

    # Summary table
    summary = make_summary_table(df)
    print("\nSummary table:")
    print(summary.to_string())
    summary.to_csv(output_dir / "summary_table.csv", index=False)
    save_latex_table(
        summary, output_dir / "table_mclta_summary.tex",
        caption="MCLTA evaluation summary across datasets.",
        label="tab:mclta_summary",
    )

    # Compression table
    compression = make_compression_table(df)
    if not compression.empty:
        compression.to_csv(output_dir / "compression_table.csv", index=False)
        save_latex_table(
            compression, output_dir / "table_mclta_compression.tex",
            caption="MCLTA clause compression relative to exhaustive atom enumeration.",
            label="tab:mclta_compression",
        )

    # Figures
    plot_direction_distribution(df, output_dir / "fig_mclta_status_dist.pdf")
    plot_runtime_distribution(df, output_dir / "fig_mclta_runtime.pdf")
    if not compression.empty:
        plot_compression_ratio(df, output_dir / "fig_mclta_compression.pdf")

    print(f"\nArtifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
