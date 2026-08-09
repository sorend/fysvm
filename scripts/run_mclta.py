#!/usr/bin/env python3
"""MCLTA experiment CLI.

Usage:
    uv run python scripts/run_mclta.py --dataset mammographic_mass --n-splits 5
    uv run python scripts/run_mclta.py --dataset all --n-splits 5 --output-dir results/mclta

Primary datasets: mammographic_mass, haberman_survival, pima_diabetes,
                  breast_cancer_original, heart_cleveland
Stress test: breast_cancer_diagnostic (WDBC)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


PRIMARY_DATASETS = [
    "mammographic_mass",
    "haberman_survival",
    "pima_diabetes",
    "breast_cancer_original",
    "heart_cleveland",
]

STRESS_DATASETS = ["breast_cancer_diagnostic"]


def main():
    parser = argparse.ArgumentParser(
        description="Run MCLTA evaluation on one or more datasets."
    )
    parser.add_argument(
        "--dataset", default="mammographic_mass",
        help="Dataset slug, 'primary' for primary 5, 'all' for primary + stress",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--n-repeats", type=int, default=1, help="Number of CV repeats")
    parser.add_argument(
        "--output-dir", default="results/mclta",
        help="Directory for output JSON files",
    )
    parser.add_argument("--time-limit", type=float, default=30.0,
                        help="MILP time limit per atom (seconds)")
    parser.add_argument("--set-cover-time-limit", type=float, default=60.0,
                        help="Time limit for set-cover MILP (seconds)")
    parser.add_argument("--max-context-features", type=int, default=5,
                        help="Maximum number of grammar context features")
    parser.add_argument("--max-clause-literals", type=int, default=2,
                        help="Maximum literals per context clause")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # Resolve dataset list
    if args.dataset == "primary":
        datasets = PRIMARY_DATASETS
    elif args.dataset == "all":
        datasets = PRIMARY_DATASETS + STRESS_DATASETS
    else:
        datasets = [args.dataset]

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from fysvm.datasets import load_dataset
    from fysvm.mclta_evaluation import evaluate_dataset_clte, results_to_dataframe, summarise_results
    from fysvm.transition_envelopes import MilpConfig

    solver = MilpConfig(
        time_limit_seconds=args.time_limit,
        relative_gap=1e-6,
    )

    all_results = []
    t_global_start = time.perf_counter()

    for slug in datasets:
        print(f"\n=== Dataset: {slug} ===")
        try:
            ds = load_dataset(slug)
        except Exception as exc:
            print(f"  SKIP: Could not load dataset: {exc}")
            continue

        X = ds.X.astype(np.float64)
        y = ds.y

        print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")

        t_ds = time.perf_counter()
        result = evaluate_dataset_clte(
            X, y,
            dataset_name=slug,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            envelope_solver=solver,
            set_cover_time_limit=args.set_cover_time_limit,
            max_context_features=args.max_context_features,
            max_clause_literals=args.max_clause_literals,
            random_state=args.random_state,
        )

        t_ds_end = time.perf_counter()
        print(f"  Done in {t_ds_end - t_ds:.1f}s. Runs: {result.n_runs}")
        print(f"  Summary: {result.summary}")

        all_results.append(result)

        # Save per-dataset results
        df = results_to_dataframe(result)
        csv_path = output_dir / f"{slug}_runs.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path}")

    # Save summary table
    if all_results:
        summary_df = summarise_results(all_results)
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
        print(summary_df.to_string())

    print(f"\nTotal runtime: {time.perf_counter() - t_global_start:.1f}s")


if __name__ == "__main__":
    main()
