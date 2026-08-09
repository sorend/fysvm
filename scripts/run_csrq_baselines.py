"""Baseline comparison and sensitivity analysis for CSRQClassifier (M10).

Runs:
  1. Baseline comparison on 5 real datasets (nested CV 3x5 folds)
     - CSRQClassifier (complete mode, r=2)
     - FuzzyRuleSVM product L1
     - FuzzyRuleSVM product L2
     - MembershipSVM L1
     - LinearSVC on raw imputed features
  2. Sensitivity analyses on 5 real datasets
     - Vary degree_penalty: {0, 0.35, 1.0}
     - Vary intercept_penalty: {0.1, 1.0, 10.0}
     - Vary max_rule_length: {1, 2}

Usage:
    uv run python scripts/run_csrq_baselines.py [--smoke] [--baselines] [--sensitivity] [--all]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from fysvm.csrq import CSRQClassifier
from fysvm.membership import MembershipSVM
from fysvm.rule_svm import FuzzyRuleSVM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear_svc_baseline(C: float = 1.0) -> LinearSVC:
    """Vanilla LinearSVC on imputed raw features."""
    return LinearSVC(
        C=C,
        penalty="l2",
        loss="squared_hinge",
        dual=True,
        class_weight="balanced",
        max_iter=10000,
        tol=1e-4,
    )


def _inner_cv_select_C(
    estimator_factory,
    C_grid,
    X_train,
    y_train,
    n_inner_folds,
    repeat,
    fold_idx,
) -> float:
    """Select C by inner cross-validation. Returns best C."""
    best_C = C_grid[0]
    best_score = -np.inf
    inner_cv = StratifiedKFold(
        n_splits=n_inner_folds, shuffle=True,
        random_state=repeat * 100 + fold_idx,
    )
    for C_val in C_grid:
        scores = []
        for i_train, i_val in inner_cv.split(X_train, y_train):
            Xi_tr = X_train[i_train]
            yi_tr = y_train[i_train]
            Xi_val = X_train[i_val]
            yi_val = y_train[i_val]
            imp = SimpleImputer(strategy="median")
            Xi_tr = imp.fit_transform(Xi_tr)
            Xi_val = imp.transform(Xi_val)
            clf = estimator_factory(C_val)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(Xi_tr, yi_tr)
                preds = clf.predict(Xi_val)
                scores.append(balanced_accuracy_score(yi_val, preds))
            except Exception:
                scores.append(float("nan"))
        valid = [s for s in scores if not np.isnan(s)]
        mean_score = float(np.mean(valid)) if valid else float("nan")
        if not np.isnan(mean_score) and mean_score > best_score:
            best_score = mean_score
            best_C = C_val
    return best_C


def _run_nested_cv(
    estimator_factory,
    X,
    y,
    n_repeats=3,
    n_outer_folds=5,
    n_inner_folds=3,
    C_grid=None,
) -> dict:
    """Run nested cross-validation for a given estimator factory.

    estimator_factory(C) -> fitted estimator (unfitted).
    Returns dict with BA, AUC, fold results.
    """
    if C_grid is None:
        C_grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

    fold_results = []
    for repeat in range(n_repeats):
        outer_cv = StratifiedKFold(
            n_splits=n_outer_folds, shuffle=True, random_state=repeat * 100
        )
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_tr_raw)
            X_te = imp.transform(X_te_raw)

            best_C = _inner_cv_select_C(
                estimator_factory, C_grid, X_tr, y_tr, n_inner_folds, repeat, fold_idx,
            )
            clf = estimator_factory(best_C)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(X_tr, y_tr)
                preds = clf.predict(X_te)
                ba = balanced_accuracy_score(y_te, preds)
                try:
                    df = clf.decision_function(X_te)
                    auc = roc_auc_score(y_te, df)
                except Exception:
                    auc = float("nan")
                fold_results.append({
                    "repeat": repeat,
                    "fold": fold_idx,
                    "best_C": best_C,
                    "balanced_accuracy": ba,
                    "roc_auc": auc,
                })
            except Exception as exc:
                fold_results.append({
                    "repeat": repeat,
                    "fold": fold_idx,
                    "best_C": best_C,
                    "balanced_accuracy": float("nan"),
                    "roc_auc": float("nan"),
                    "error": str(exc),
                })

    bas = [r["balanced_accuracy"] for r in fold_results if not np.isnan(r["balanced_accuracy"])]
    aucs = [r["roc_auc"] for r in fold_results if not np.isnan(r["roc_auc"])]
    return {
        "mean_balanced_accuracy": float(np.mean(bas)) if bas else float("nan"),
        "std_balanced_accuracy": float(np.std(bas)) if bas else float("nan"),
        "mean_roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_folds": len(fold_results),
        "n_valid": len(bas),
        "fold_results": fold_results,
    }


# ---------------------------------------------------------------------------
# Estimator factories
# ---------------------------------------------------------------------------

def csrq_factory(r=2, degree_penalty=0.35, intercept_penalty=1.0):
    def factory(C):
        return CSRQClassifier(
            C=C,
            max_rule_length=r,
            partition_quantiles=(0.05, 0.5, 0.95),
            degree_penalty=degree_penalty,
            intercept_penalty=intercept_penalty,
            class_weight="balanced",
            strict_anchor_policy="drop",
            max_semantic_terms=4096,
        )
    return factory


def fuzzy_rule_svm_factory(penalty="l1"):
    def factory(C):
        return FuzzyRuleSVM(
            C=C,
            penalty=penalty,
            and_operator="product",
            max_rule_length=2,
            partition_quantiles=(0.05, 0.5, 0.95),
            class_weight="balanced",
        )
    return factory


def membership_svm_factory():
    def factory(C):
        return MembershipSVM(
            C=C,
            penalty="l1",
            partition_quantiles=(0.05, 0.5, 0.95),
            class_weight="balanced",
        )
    return factory


def linear_svc_factory():
    def factory(C):
        return LinearSVC(
            C=C,
            penalty="l2",
            loss="squared_hinge",
            dual=True,
            class_weight="balanced",
            max_iter=10000,
            tol=1e-4,
        )
    return factory


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

BASELINES = [
    ("CSRQClassifier r=2", csrq_factory(r=2)),
    ("FuzzyRuleSVM product L1", fuzzy_rule_svm_factory("l1")),
    ("FuzzyRuleSVM product L2", fuzzy_rule_svm_factory("l2")),
    ("MembershipSVM L1", membership_svm_factory()),
    ("LinearSVC raw", linear_svc_factory()),
]


def run_baselines(datasets, smoke=False) -> list[dict]:
    """Run baseline comparison on all datasets."""
    n_repeats = 1 if smoke else 3
    n_outer = 3 if smoke else 5
    results = []

    for ds_spec in datasets:
        print(f"\n  Dataset: {ds_spec.name} (n={len(ds_spec.y)}, d={ds_spec.X.shape[1]})")
        X, y = ds_spec.X, ds_spec.y
        ds_results = {"dataset": ds_spec.name, "n": int(len(y)), "d": int(X.shape[1])}

        for name, factory in BASELINES:
            t0 = time.perf_counter()
            print(f"    {name}...", end=" ", flush=True)
            result = _run_nested_cv(
                factory, X, y,
                n_repeats=n_repeats,
                n_outer_folds=n_outer,
                n_inner_folds=3,
            )
            elapsed = time.perf_counter() - t0
            print(f"BA={result['mean_balanced_accuracy']:.3f}±{result['std_balanced_accuracy']:.3f} "
                  f"AUC={result['mean_roc_auc']:.3f} ({elapsed:.1f}s)")
            ds_results[name] = result

        results.append(ds_results)

    return results


# ---------------------------------------------------------------------------
# Sensitivity analyses
# ---------------------------------------------------------------------------

SENSITIVITY_PARAMS = {
    "degree_penalty": [
        ("eta=0", csrq_factory(r=2, degree_penalty=0.0)),
        ("eta=0.35", csrq_factory(r=2, degree_penalty=0.35)),
        ("eta=1.0", csrq_factory(r=2, degree_penalty=1.0)),
    ],
    "intercept_penalty": [
        ("p0=0.1", csrq_factory(r=2, intercept_penalty=0.1)),
        ("p0=1.0", csrq_factory(r=2, intercept_penalty=1.0)),
        ("p0=10.0", csrq_factory(r=2, intercept_penalty=10.0)),
    ],
    "max_rule_length": [
        ("r=1", csrq_factory(r=1)),
        ("r=2", csrq_factory(r=2)),
    ],
}


def run_sensitivity(datasets, smoke=False) -> dict:
    """Run sensitivity analyses for CSRQ parameters."""
    n_repeats = 1 if smoke else 3
    n_outer = 3 if smoke else 5
    all_results = {}

    for param_name, variants in SENSITIVITY_PARAMS.items():
        print(f"\n  Parameter: {param_name}")
        param_results = {}

        for ds_spec in datasets:
            print(f"    Dataset: {ds_spec.name}")
            X, y = ds_spec.X, ds_spec.y
            ds_entry = {}

            for variant_name, factory in variants:
                print(f"      {variant_name}...", end=" ", flush=True)
                t0 = time.perf_counter()
                result = _run_nested_cv(
                    factory, X, y,
                    n_repeats=n_repeats,
                    n_outer_folds=n_outer,
                    n_inner_folds=3,
                )
                elapsed = time.perf_counter() - t0
                print(f"BA={result['mean_balanced_accuracy']:.3f}±{result['std_balanced_accuracy']:.3f} "
                      f"({elapsed:.1f}s)")
                ds_entry[variant_name] = {
                    "mean_balanced_accuracy": result["mean_balanced_accuracy"],
                    "std_balanced_accuracy": result["std_balanced_accuracy"],
                    "mean_roc_auc": result["mean_roc_auc"],
                }

            param_results[ds_spec.name] = ds_entry

        all_results[param_name] = param_results

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class _DsSpec:
    """Minimal dataset spec container."""
    def __init__(self, name, X, y):
        self.name = name
        self.X = X
        self.y = y


def main():
    parser = argparse.ArgumentParser(description="CSRQ baseline comparison and sensitivity analysis.")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (fast).")
    parser.add_argument("--baselines", action="store_true", help="Run baseline comparison.")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analyses.")
    parser.add_argument("--all", dest="run_all", action="store_true", help="Run everything.")
    parser.add_argument("--output", default="results/csrq_baselines.json", help="Output file.")
    args = parser.parse_args()

    if args.run_all:
        args.baselines = True
        args.sensitivity = True

    if not (args.baselines or args.sensitivity):
        args.baselines = True
        args.sensitivity = True

    # Load datasets
    try:
        from fysvm.datasets import list_datasets, load_dataset
        PRIMARY_SLUGS = [
            "pima_diabetes",
            "heart_cleveland",
            "breast_cancer_diagnostic",
            "mammographic_mass",
            "parkinsons",
        ]
        all_ds = {ds.slug: ds for ds in list_datasets()}
        datasets = []
        for slug in PRIMARY_SLUGS:
            if slug in all_ds:
                try:
                    ds = load_dataset(slug)
                    datasets.append(_DsSpec(all_ds[slug].name, ds.X, ds.y))
                except Exception as e:
                    print(f"  Skipping {slug}: {e}")
    except Exception as exc:
        print(f"Dataset loading failed: {exc}")
        return {}

    if not datasets:
        print("No datasets available. Exiting.")
        return {}

    print(f"\nLoaded {len(datasets)} datasets.")

    all_results = {}

    if args.baselines:
        print("\n=== Baseline Comparison ===")
        all_results["baselines"] = run_baselines(datasets, smoke=args.smoke)

    if args.sensitivity:
        print("\n=== Sensitivity Analyses ===")
        all_results["sensitivity"] = run_sensitivity(datasets, smoke=args.smoke)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    return all_results


if __name__ == "__main__":
    main()
