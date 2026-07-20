"""High-dimensional failure mode analysis: rule-budget sensitivity sweep.

Evaluates FuzzyRuleSVM with varying rule budgets (max_rules) and rule lengths
(max_rule_length) on a representative set of datasets spanning low to very high
dimensionality.  Tracks not only predictive accuracy but also candidate rule
counts (before truncation) and explanation compactness to characterise how
rule-budget truncation degrades the conjunctive-rule story in high-dimensional
settings.

Datasets studied:
  pima_diabetes               d=8    (low-dim reference)
  spectf_heart                d=44   (moderate-dim, max_rule_length capped to 1)
  arrhythmia_binary           d=279  (high-dim, max_rule_length=1 only)
  parkinsons_disease_classification d=754 (very high-dim, max_rule_length=1 only)

Rule budgets swept: 64, 128, 256, 512, 1024
Rule lengths: max_rule_length=1 always; max_rule_length=2 added for d<=32

Usage:
    uv run python scripts/analyze_highdim.py
    uv run python scripts/analyze_highdim.py --datasets pima_diabetes spectf_heart
    uv run python scripts/analyze_highdim.py --max-samples 400
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from fysvm.datasets import PreparedDataset, load_dataset
from fysvm.evaluation import (
    _json_default,
    _stable_dataset_seed,
    _predictive_metrics,
    _stratified_subset,
    _fuzzy_rule_metrics,
)
from fysvm.rule_svm import FuzzyRuleSVM
from fysvm.run_metadata import write_run_metadata


# ---------------------------------------------------------------------------
# Datasets and budget configurations for the sweep
# ---------------------------------------------------------------------------

#: Datasets included in the budget sweep, grouped by dimensionality regime.
SWEEP_DATASETS: list[str] = [
    "pima_diabetes",                      # d=8  low-dim
    "spectf_heart",                       # d=44 moderate-dim
    "arrhythmia_binary",                  # d=279 high-dim
    "parkinsons_disease_classification",  # d=754 very high-dim
]

#: Rule budgets (max_rules) to sweep.
RULE_BUDGETS: list[int] = [64, 128, 256, 512, 1024]

#: Maximum antecedent length threshold: length-2 rules only for d <= this value.
_MAX_LEN2_DIM_THRESHOLD: int = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _suppressed_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        yield


def _build_frs(
    dataset: PreparedDataset,
    random_state: int,
    max_rules: int,
    max_rule_length: int,
    C: float = 1.0,
    penalty: Literal["l1", "l2"] = "l1",
) -> Any:
    """Build a FuzzyRuleSVM (wrapped in OneVsRestClassifier if multiclass)."""
    base = FuzzyRuleSVM(
        C=C,
        penalty=penalty,
        max_rule_length=max_rule_length,
        max_rules=max_rules,
        min_rule_coverage=0.01,
        rule_length_penalty=0.35,
        feature_names=dataset.feature_names,
        class_weight="balanced",
        random_state=random_state,
        max_iter=20000,
    )
    if dataset.spec.task == "multiclass":
        return OneVsRestClassifier(base)
    return base


def _frs_param_grid(max_rules: int, max_rule_length: int) -> list[dict[str, Any]]:
    """Inner-CV parameter grid for a given (max_rules, max_rule_length) setting."""
    return [
        {"C": c, "penalty": penalty}
        for c in (0.3, 1.0, 3.0)
        for penalty in ("l1", "l2")
    ]


def _select_params_inner_cv(
    dataset: PreparedDataset,
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_rules: int,
    max_rule_length: int,
    inner_splits: int,
    random_state: int,
) -> dict[str, Any]:
    """Select (C, penalty) by inner-CV balanced accuracy for given budget/length."""
    grid = _frs_param_grid(max_rules, max_rule_length)

    class_counts = np.unique(y_train, return_counts=True)[1]
    splits = min(inner_splits, int(np.min(class_counts)))
    if splits < 2:
        return {"C": 1.0, "penalty": "l1"}

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for i, params in enumerate(grid):
        bal_scores: list[float] = []
        for fold_i, (inner_train, inner_val) in enumerate(cv.split(X_train, y_train)):
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", _build_frs(dataset, random_state + fold_i, max_rules,
                                     max_rule_length, **params)),
            ])
            with _suppressed_warnings():
                pipe.fit(X_train[inner_train], y_train[inner_train])
            y_pred = pipe.predict(X_train[inner_val])
            bal_scores.append(float(balanced_accuracy_score(y_train[inner_val], y_pred)))
        scored.append((float(np.mean(bal_scores)), -i, params))

    _, _, best_params = max(scored)
    return best_params


def _extract_candidate_rules(model: Any) -> float:
    """Return mean n_candidate_rules_ across binary sub-estimators."""
    if isinstance(model, OneVsRestClassifier):
        counts = [
            getattr(est, "n_candidate_rules_", float("nan"))
            for est in model.estimators_
        ]
        valid = [c for c in counts if not (isinstance(c, float) and np.isnan(c))]
        return float(np.mean(valid)) if valid else float("nan")
    return float(getattr(model, "n_candidate_rules_", float("nan")))


def _run_budget_sweep(
    dataset: PreparedDataset,
    budgets: list[int],
    outer_splits: int,
    inner_splits: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Run the nested-CV budget sweep for one dataset.

    For each (max_rules, max_rule_length) configuration, runs outer_splits-fold
    stratified CV with inner_splits-fold inner selection of (C, penalty).
    Returns a list of per-fold result rows.
    """
    X = np.asarray(dataset.X)
    y = np.asarray(dataset.y)
    n_features = X.shape[1]

    # Determine which rule-length configurations to test
    lengths_to_test: list[int] = [1]
    if n_features <= _MAX_LEN2_DIM_THRESHOLD:
        lengths_to_test = [1, 2]

    class_counts = np.unique(y, return_counts=True)[1]
    n_splits = min(outer_splits, int(np.min(class_counts)))
    if n_splits < 2:
        raise ValueError(f"Dataset {dataset.spec.slug}: too few samples per class.")

    rows: list[dict[str, Any]] = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    configs: list[tuple[int, int]] = [
        (max_rules, length)
        for length in lengths_to_test
        for max_rules in budgets
    ]
    total = len(configs)

    for cfg_i, (max_rules, max_rule_length) in enumerate(configs):
        label = f"FRS_budget{max_rules}_len{max_rule_length}"
        print(
            f"    [{cfg_i + 1}/{total}] {label} ...",
            flush=True,
        )

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            seed = random_state + fold * 100 + cfg_i
            best_params = _select_params_inner_cv(
                dataset, X_train, y_train, max_rules, max_rule_length,
                inner_splits=inner_splits, random_state=seed,
            )

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", _build_frs(dataset, seed, max_rules, max_rule_length,
                                     **best_params)),
            ])
            t0 = time.perf_counter()
            with _suppressed_warnings():
                pipe.fit(X_train, y_train)
            fit_t = time.perf_counter() - t0

            X_test_prepared = pipe.named_steps["imputer"].transform(X_test)
            y_pred = pipe.predict(X_test)
            pred_metrics = _predictive_metrics(pipe, X_test, y_test, y_pred)

            model = pipe.named_steps["model"]
            rule_metrics = _fuzzy_rule_metrics(model, X_test_prepared, y_test)
            n_candidates = _extract_candidate_rules(model)

            rows.append({
                "dataset": dataset.spec.slug,
                "n_features": n_features,
                "max_rules": max_rules,
                "max_rule_length": max_rule_length,
                "config_label": label,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "fit_seconds": fit_t,
                "n_candidate_rules": n_candidates,
                "selected_C": best_params.get("C"),
                "selected_penalty": best_params.get("penalty"),
                **pred_metrics,
                **rule_metrics,  # already prefixed with "rule_" by _fuzzy_rule_metrics
            })

    return rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-fold rows to per-(dataset, max_rules, max_rule_length) summary."""
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["max_rules"], row["max_rule_length"])
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for (dataset, max_rules, max_rule_length), fold_rows in sorted(grouped.items()):
        first = fold_rows[0]
        num_keys = [
            k for k, v in first.items()
            if isinstance(v, int | float | np.integer | np.floating)
            and k not in {"fold", "n_train", "n_test", "max_rules", "max_rule_length", "n_features"}
        ]
        s: dict[str, Any] = {
            "dataset": dataset,
            "n_features": first["n_features"],
            "max_rules": max_rules,
            "max_rule_length": max_rule_length,
            "config_label": first["config_label"],
            "n_folds": len(fold_rows),
        }
        for k in num_keys:
            vals = np.asarray([r[k] for r in fold_rows], dtype=float)
            s[f"{k}_mean"] = float(np.nanmean(vals))
            s[f"{k}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            s[f"{k}_median"] = float(np.nanmedian(vals))
        summaries.append(s)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    all_keys: list[str] = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary_table(summaries: list[dict[str, Any]]) -> None:
    """Print a compact summary table of key results."""
    print("\n" + "=" * 80)
    print("Rule-budget sensitivity summary")
    print("=" * 80)
    print(f"{'Dataset':<40} {'d':>5} {'Budget':>8} {'Len':>4} {'BalAcc':>8} {'Candidates':>12} {'Active':>8}")
    print("-" * 80)
    for s in summaries:
        ba = s.get("balanced_accuracy_mean", float("nan"))
        cands = s.get("n_candidate_rules_mean", float("nan"))
        active = s.get("rule_support_rule_count_mean", float("nan"))
        print(
            f"{s['dataset']:<40} {s['n_features']:>5d} "
            f"{s['max_rules']:>8d} {s['max_rule_length']:>4d} "
            f"{ba:>8.4f} {cands:>12.0f} {active:>8.1f}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=SWEEP_DATASETS,
        help="Dataset slugs to include in the sweep (default: %(default)s).",
    )
    parser.add_argument("--data-dir", default="datasets/prepared")
    parser.add_argument("--output-dir", default="runs/highdim-analysis")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=RULE_BUDGETS,
        help="Rule budgets (max_rules) to sweep (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir, config=vars(args))
    data_dir = Path(args.data_dir)

    all_rows: list[dict[str, Any]] = []
    for i, slug in enumerate(args.datasets):
        print(f"\n[{i + 1}/{len(args.datasets)}] {slug} ...", flush=True)
        dataset = load_dataset(slug, data_dir)
        dataset_seed = _stable_dataset_seed(args.random_state, slug)
        if args.max_samples and dataset.X.shape[0] > args.max_samples:
            dataset = _stratified_subset(dataset, args.max_samples, dataset_seed)
        rows = _run_budget_sweep(
            dataset,
            budgets=args.budgets,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            random_state=dataset_seed,
        )
        all_rows.extend(rows)
        print(f"  Done: {len(rows)} fold-config rows for {slug}.", flush=True)

    summaries = _summarize(all_rows)
    _print_summary_table(summaries)

    _write_csv(output_dir / "fold_metrics.csv", all_rows)
    _write_csv(output_dir / "metrics.csv", summaries)
    (output_dir / "metrics.json").write_text(
        json.dumps(summaries, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults written to: {output_dir}")


if __name__ == "__main__":
    main()
