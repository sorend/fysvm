"""Bootstrap stability and finite C-grid near-optimal validation analysis.

Runs three empirical studies on five biomedical datasets:
  1. Bootstrap prediction stability (H3.1)
  2. Finite C-grid near-optimal validation-set analysis (H3.3)
  3. Bootstrap monotonicity certificate-status analysis (H3.2)

Outputs
-------
runs/finite_c_grid_stability/stability_results.json
runs/finite_c_grid_stability/summary_bootstrap.csv
runs/finite_c_grid_stability/summary_finite_c_grid.csv
runs/finite_c_grid_stability/summary_monotonicity_certificate_retention.csv

Usage
-----
    uv run python scripts/run_rashomon_stability.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure the src directory is on the path (for development installs)
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fysvm.datasets import load_dataset
from fysvm.rule_svm import FuzzyRuleSVM
from fysvm.run_metadata import write_run_metadata
from fysvm.stability import (
    bootstrap_prediction_stability,
    certificate_retention_bootstrap,
    rashomon_stability,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRIMARY_DATASETS = [
    "pima_diabetes",
    "heart_cleveland",
    "breast_cancer_diagnostic",
    "mammographic_mass",
    "parkinsons",
]

# Reference model configuration (fixed)
_CLF_KWARGS = dict(
    penalty="l1",
    and_operator="min",
    max_rule_length=2,
    max_rules=256,
    min_rule_coverage=0.01,
    rule_length_penalty=0.35,
    class_weight="balanced",
    rule_generation="enumeration",
    softmin_temperature=0.1,
    random_state=0,
    max_iter=20000,
)

# Finite C-grid for near-optimal validation analysis
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

# Number of bootstrap iterations
N_BOOTSTRAP = 200

# Additive balanced-accuracy tolerance
DELTA_BACC = 0.02

# Primary feature for certificate retention per dataset
# (feature name to search for by case-insensitive substring match)
CERT_FEATURES: dict[str, tuple[str, str]] = {
    # slug: (feature_name_pattern, direction)
    "pima_diabetes":            ("glucose",      "positive"),
    "heart_cleveland":          ("oldpeak",      "negative"),   # negatively annotated
    "breast_cancer_diagnostic": ("worst radius", "positive"),
    "mammographic_mass":        ("birads",       "positive"),
    "parkinsons":               ("MDVP:Fo",      "negative"),   # lower Fo → Parkinson's
}


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_clf_factory(feature_names: list[str] | None = None):
    """Return a zero-arg factory for a fresh FuzzyRuleSVM at C=1.0."""
    def factory():
        return FuzzyRuleSVM(C=1.0, feature_names=feature_names, **_CLF_KWARGS)
    return factory


def make_finite_c_grid_factory(feature_names: list[str] | None = None):
    """Return a factory (C: float) → fresh FuzzyRuleSVM."""
    def factory(C: float):
        return FuzzyRuleSVM(C=C, feature_names=feature_names, **_CLF_KWARGS)
    return factory


# ---------------------------------------------------------------------------
# Feature index lookup
# ---------------------------------------------------------------------------


def _find_feature_index(feature_names: list[str], pattern: str) -> int | None:
    """Case-insensitive substring match to find a feature index."""
    p_norm = pattern.lower().replace("_", "").replace("-", "").replace(" ", "")
    for i, fn in enumerate(feature_names):
        fn_norm = fn.lower().replace("_", "").replace("-", "").replace(" ", "")
        if p_norm in fn_norm or fn_norm in p_norm:
            return i
    return None


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_dataset(slug: str) -> dict:
    """Run all stability analyses for one dataset.

    Returns bootstrap, finite C-grid, and monotonicity certificate results.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {slug}")
    print(f"{'='*60}")

    # --- Load data ---
    ds = load_dataset(slug)
    X, y = ds.X, ds.y
    feature_names = list(ds.feature_names)
    print(f"  Shape: {X.shape}, Classes: {np.unique(y)}")

    # --- Single 80/20 stratified split ---
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    # --- 75/25 train/val split from the 80% (for finite C-grid selection) ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=0
    )

    print(
        f"  Split: train={len(X_train_full)}, test={len(X_test)}"
        f" (finite C-grid: train={len(X_train)}, val={len(X_val)})"
    )

    # --- 1. Bootstrap prediction stability ---
    print(f"\n  [1/3] Bootstrap stability (n={N_BOOTSTRAP})...")
    t0 = time.time()
    boot_result = bootstrap_prediction_stability(
        make_clf_factory(feature_names),
        X_train_full,
        y_train_full,
        X_test,
        n_bootstrap=N_BOOTSTRAP,
        random_state=42,
        dataset_name=slug,
        y_test=y_test,
    )
    boot_time = time.time() - t0
    print(
        f"    mean_agreement={boot_result.mean_prediction_agreement:.4f}"
        f"  std={boot_result.std_prediction_agreement:.4f}"
        f"  [{boot_time:.1f}s]"
    )
    bootstrap_dict = {
        "mean_prediction_agreement": boot_result.mean_prediction_agreement,
        "std_prediction_agreement": boot_result.std_prediction_agreement,
        "q05_prediction_agreement": boot_result.q05_prediction_agreement,
        "q50_prediction_agreement": boot_result.q50_prediction_agreement,
        "q95_prediction_agreement": boot_result.q95_prediction_agreement,
        "n_bootstrap": boot_result.n_bootstrap,
        "per_bootstrap_agreement": boot_result.per_bootstrap_agreement.tolist(),
        "per_sample_agreement": boot_result.per_sample_agreement.tolist(),
        "reference_test_balanced_accuracy": boot_result.reference_test_balanced_accuracy,
        "details": boot_result.details,
    }

    # --- 2. Finite C-grid stability ---
    print(f"\n  [2/3] Finite C-grid near-optimal validation analysis...")
    t0 = time.time()
    grid_result = rashomon_stability(
        make_finite_c_grid_factory(feature_names),
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        c_grid=C_GRID,
        delta_bacc=DELTA_BACC,
        dataset_name=slug,
        y_test=y_test,
    )
    grid_time = time.time() - t0
    print(
        f"    near_optimal_size={grid_result.near_optimal_size}"
        f"  agreement={grid_result.test_prediction_agreement:.4f}"
        f"  best_C={grid_result.details['best_c']}"
        f"  [{grid_time:.1f}s]"
    )
    finite_c_grid_dict = {
        "c_grid": grid_result.c_grid,
        "val_balanced_accuracies": {
            str(c): v for c, v in grid_result.val_balanced_accuracies.items()
        },
        "near_optimal_set": grid_result.near_optimal_set,
        "near_optimal_size": grid_result.near_optimal_size,
        "delta_bacc": grid_result.delta_bacc,
        "test_prediction_agreement": grid_result.test_prediction_agreement,
        "best_model_test_balanced_accuracy": grid_result.best_model_test_balanced_accuracy,
        "details": grid_result.details,
    }

    # --- 3. Certificate retention ---
    print(f"\n  [3/3] Monotonicity certificate-status bootstrap...")
    feat_pattern, feat_direction = CERT_FEATURES[slug]
    feat_idx = _find_feature_index(feature_names, feat_pattern)
    if feat_idx is None:
        raise ValueError(f"Feature '{feat_pattern}' not found for dataset '{slug}'.")
    feat_name = feature_names[feat_idx]
    print(f"    Feature: '{feat_name}' (idx={feat_idx}), direction={feat_direction}")
    t0 = time.time()
    cert_result = certificate_retention_bootstrap(
        make_clf_factory(feature_names),
        X_train_full,
        y_train_full,
        feature_index=feat_idx,
        direction=feat_direction,
        n_bootstrap=N_BOOTSTRAP,
        random_state=42,
    )
    cert_time = time.time() - t0
    print(
        f"    reference_status={cert_result.reference_status}"
        f"  reference_status_retention={cert_result.reference_status_retention_rate:.4f}"
        f"  certified_rate={cert_result.certified_rate:.4f}"
        f"  [{cert_time:.1f}s]"
    )
    cert_dict = {
        "feature_name": feat_name,
        "feature_index": feat_idx,
        "direction": feat_direction,
        "n_bootstrap": cert_result.n_bootstrap,
        "reference_status": cert_result.reference_status,
        "reference_status_retention_rate": cert_result.reference_status_retention_rate,
        "certified_rate": cert_result.certified_rate,
        "status_counts": cert_result.status_counts,
        "status_fractions": cert_result.status_fractions,
        "certificate_type": cert_result.certificate_type,
        "details": cert_result.details,
    }

    return {
        "dataset": slug,
        "n_train": int(len(X_train_full)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "bootstrap": bootstrap_dict,
        "finite_c_grid": finite_c_grid_dict,
        "monotonicity_certificate_retention": cert_dict,
    }


def print_summary(all_results: dict[str, dict]) -> None:
    """Print a formatted summary table."""
    header = (
        f"{'Dataset':<30} | {'mean_agree':>10} | {'near_opt_sz':>11} | "
        f"{'grid_agree':>10} | {'cert_rate':>9}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("  BOOTSTRAP AND FINITE C-GRID STABILITY SUMMARY")
    print(sep)
    print(header)
    print(sep)

    for slug, res in all_results.items():
        boot = res["bootstrap"]
        grid = res["finite_c_grid"]
        cert = res["monotonicity_certificate_retention"]

        mean_agree = f"{boot['mean_prediction_agreement']:.4f}"
        near_optimal_size = str(grid["near_optimal_size"])
        grid_agree = f"{grid['test_prediction_agreement']:.4f}"
        certified_rate = f"{cert['certified_rate']:.4f}"

        print(
            f"{slug:<30} | {mean_agree:>10} | {near_optimal_size:>11} | "
            f"{grid_agree:>10} | {certified_rate:>9}"
        )

    print(sep)


def main() -> None:
    out_dir = Path("runs/finite_c_grid_stability")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Bootstrap Stability + Finite C-Grid Near-Optimal Validation Analysis")
    print(f"N_BOOTSTRAP={N_BOOTSTRAP}, C_GRID={C_GRID}")
    print(f"DELTA_BACC={DELTA_BACC}")
    print("=" * 60)

    all_results: dict[str, dict] = {}
    for slug in PRIMARY_DATASETS:
        all_results[slug] = run_dataset(slug)

    # --- Save JSON ---
    json_path = out_dir / "stability_results.json"
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2, allow_nan=False)
    write_run_metadata(
        out_dir,
        command=["uv", "run", "python", *sys.argv],
        config={
            "datasets": PRIMARY_DATASETS,
            "n_bootstrap": N_BOOTSTRAP,
            "n_jobs": 1,
            "c_grid": C_GRID,
            "delta_bacc": DELTA_BACC,
            "classifier": _CLF_KWARGS,
            "certificate_analysis": "secondary monotonicity status analysis",
        },
    )
    print(f"\nResults saved to {json_path}")

    # --- Summary tables ---
    print_summary(all_results)

    # Bootstrap CSV
    boot_rows = []
    for slug, res in all_results.items():
        boot = res["bootstrap"]
        boot_rows.append({
            "dataset": slug,
            "n_train": res["n_train"],
            "n_test": res["n_test"],
            "n_bootstrap": boot["n_bootstrap"],
            "mean_agreement": boot["mean_prediction_agreement"],
            "std_agreement": boot["std_prediction_agreement"],
            "q05_agreement": boot["q05_prediction_agreement"],
            "q50_agreement": boot["q50_prediction_agreement"],
            "q95_agreement": boot["q95_prediction_agreement"],
            "reference_test_balanced_accuracy": boot["reference_test_balanced_accuracy"],
        })
    pd.DataFrame(boot_rows).to_csv(out_dir / "summary_bootstrap.csv", index=False)

    # Finite C-grid CSV
    grid_rows = []
    for slug, res in all_results.items():
        grid = res["finite_c_grid"]
        grid_rows.append({
            "dataset": slug,
            "near_optimal_size": grid["near_optimal_size"],
            "near_optimal_set": str(grid["near_optimal_set"]),
            "test_prediction_agreement": grid["test_prediction_agreement"],
            "delta_bacc": grid["delta_bacc"],
            "best_c": grid["details"]["best_c"],
            "best_validation_balanced_accuracy": grid["details"]["best_bacc"],
            "best_model_test_balanced_accuracy": grid["best_model_test_balanced_accuracy"],
        })
    pd.DataFrame(grid_rows).to_csv(out_dir / "summary_finite_c_grid.csv", index=False)

    # Certificate retention CSV
    cert_rows = []
    for slug, res in all_results.items():
        cert = res["monotonicity_certificate_retention"]
        cert_rows.append({
            "dataset": slug,
            "certificate_type": "monotonicity",
            "feature_name": cert["feature_name"],
            "direction": cert["direction"],
            "reference_status": cert["reference_status"],
            "reference_status_retention_rate": cert["reference_status_retention_rate"],
            "certified_rate": cert["certified_rate"],
            "status_counts": json.dumps(cert["status_counts"], sort_keys=True),
            "status_fractions": json.dumps(cert["status_fractions"], sort_keys=True),
            "n_bootstrap": cert["n_bootstrap"],
        })
    pd.DataFrame(cert_rows).to_csv(
        out_dir / "summary_monotonicity_certificate_retention.csv", index=False
    )

    print(f"\nCSV summaries saved to {out_dir}/")


if __name__ == "__main__":
    main()
