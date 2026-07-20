"""Broader ablation suite for FuzzyRuleSVM design choices.

Tests eight design dimensions of FuzzyRuleSVM against a raw-features baseline:

  frs_default       -- Standard FuzzyRuleSVM (L1/L2 tuned, min t-norm,
                       length_penalty=0.35, quantiles=0.05/0.5/0.95)
  frs_no_penalty    -- Same but rule_length_penalty=0.0
  frs_l2_forced     -- Force L2 penalty (C tuned only)
  frs_product_tnorm -- Product t-norm instead of min
  frs_softmin_tnorm -- Smooth-min aggregation instead of min
  frs_wide_quantiles-- Quantile anchors (0.25/0.5/0.75) vs default (0.05/0.5/0.95)
  frs_1term_only    -- max_rule_length=1 for ALL datasets (removes 2-antecedent rules)
  membership_svm    -- No rule generation; L1/L2 LinearSVC on 3*d membership features
  linear_svc_raw    -- Standardised LinearSVC on original features (strongest sparse baseline)

Design decisions (documented):
  - Each variant fixes the ablated parameter and tunes C (and penalty where applicable)
    via inner 3-fold CV on balanced accuracy.  This cleanly isolates each design choice.
  - frs_default and all FRS variants keep max_rules=min(256, max(24, 3*d)) and the
    same max_rule_length heuristic as the main evaluation harness (except frs_1term_only
    which forces length=1 everywhere).
  - frs_l2_forced fixes penalty="l2" and tunes only C in {0.3, 1.0, 3.0}.
  - linear_svc_raw uses StandardScaler + LinearSVC tuned over C in {0.01, 0.1, 1.0, 10.0}.
  - All other variants tune C in {0.3, 1.0, 3.0} and penalty in {l1, l2} (same grid
    as frs_default).

Usage:
    uv run python scripts/ablation_suite.py
    uv run python scripts/ablation_suite.py --max-samples 400
    uv run python scripts/ablation_suite.py pima_diabetes statlog_heart
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from fysvm.datasets import DATASET_SPECS, PreparedDataset, load_dataset
from fysvm.evaluation import _json_default, _predictive_metrics, _stable_dataset_seed, _stratified_subset
from fysvm.rule_svm import FuzzyRuleSVM, _FuzzyPartition


# ---------------------------------------------------------------------------
# MembershipSVM (inline copy -- avoids circular import from ablation_membership_svm)
# ---------------------------------------------------------------------------

class MembershipSVM(ClassifierMixin, BaseEstimator):
    """L1/L2-sparse linear SVM on 3*d fuzzy membership features (no conjunctions)."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        penalty: str = "l1",
        partition_quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95),
        class_weight: Any = None,
        random_state: int | None = None,
        max_iter: int = 10000,
        tol: float = 1e-4,
    ) -> None:
        self.C = C
        self.penalty = penalty
        self.partition_quantiles = partition_quantiles
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MembershipSVM":
        X, y = check_X_y(X, y, dtype=np.float64)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        quantiles = np.quantile(X, self.partition_quantiles, axis=0)
        self.partitions_ = [
            _FuzzyPartition(
                low=float(quantiles[0, j]),
                medium=float(quantiles[1, j]),
                high=float(quantiles[2, j]),
            )
            for j in range(X.shape[1])
        ]
        M = self._memberships(X)
        svc = LinearSVC(
            C=self.C,
            penalty=self.penalty,
            loss="squared_hinge",
            dual=False,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        svc.fit(M, y)
        self.estimator_ = svc
        return self

    def _memberships(self, X: np.ndarray) -> np.ndarray:
        parts = [p.transform(X[:, j]) for j, p in enumerate(self.partitions_)]
        return np.clip(np.hstack(parts), 0.0, 1.0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self.estimator_.decision_function(self._memberships(check_array(X, dtype=np.float64)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self.estimator_.predict(self._memberships(check_array(X, dtype=np.float64)))


# ---------------------------------------------------------------------------
# Model spec framework
# ---------------------------------------------------------------------------

ModelBuilder = Callable[[PreparedDataset, int, dict[str, Any]], Any]
ParamGridBuilder = Callable[[PreparedDataset], list[dict[str, Any]]]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    builder: ModelBuilder
    param_grid: ParamGridBuilder
    uses_standard_scaler: bool = False


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

def _frs_c_and_penalty_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    """Standard FuzzyRuleSVM grid: C x penalty."""
    n_features = dataset.X.shape[1]
    max_rule_length = 2 if n_features <= 32 else 1
    max_rules = min(256, max(24, 3 * n_features))
    return [
        {
            "C": c,
            "penalty": penalty,
            "max_rule_length": max_rule_length,
            "max_rules": max_rules,
            "min_rule_coverage": 0.01,
            "rule_length_penalty": 0.35,
        }
        for c in (0.3, 1.0, 3.0)
        for penalty in ("l1", "l2")
    ]


def _frs_no_penalty_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    """FuzzyRuleSVM with rule_length_penalty=0: C x penalty."""
    n_features = dataset.X.shape[1]
    max_rule_length = 2 if n_features <= 32 else 1
    max_rules = min(256, max(24, 3 * n_features))
    return [
        {
            "C": c,
            "penalty": penalty,
            "max_rule_length": max_rule_length,
            "max_rules": max_rules,
            "min_rule_coverage": 0.01,
            "rule_length_penalty": 0.0,   # ablated: no length penalty
        }
        for c in (0.3, 1.0, 3.0)
        for penalty in ("l1", "l2")
    ]


def _frs_l2_forced_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    """FuzzyRuleSVM with L2 penalty forced; tunes only C."""
    n_features = dataset.X.shape[1]
    max_rule_length = 2 if n_features <= 32 else 1
    max_rules = min(256, max(24, 3 * n_features))
    return [
        {
            "C": c,
            "penalty": "l2",              # ablated: force L2
            "max_rule_length": max_rule_length,
            "max_rules": max_rules,
            "min_rule_coverage": 0.01,
            "rule_length_penalty": 0.35,
        }
        for c in (0.3, 1.0, 3.0)
    ]


def _frs_product_tnorm_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    """FuzzyRuleSVM with product t-norm."""
    n_features = dataset.X.shape[1]
    max_rule_length = 2 if n_features <= 32 else 1
    max_rules = min(256, max(24, 3 * n_features))
    return [
        {
            "C": c,
            "penalty": penalty,
            "max_rule_length": max_rule_length,
            "max_rules": max_rules,
            "min_rule_coverage": 0.01,
            "rule_length_penalty": 0.35,
            "and_operator": "product",    # ablated: product t-norm
        }
        for c in (0.3, 1.0, 3.0)
        for penalty in ("l1", "l2")
    ]


def _frs_softmin_tnorm_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    """FuzzyRuleSVM with smooth-min aggregation."""
    n_features = dataset.X.shape[1]
    max_rule_length = 2 if n_features <= 32 else 1
    max_rules = min(256, max(24, 3 * n_features))
    return [
        {
            "C": c,
            "penalty": penalty,
            "max_rule_length": max_rule_length,
            "max_rules": max_rules,
            "min_rule_coverage": 0.01,
            "rule_length_penalty": 0.35,
            "and_operator": "softmin",    # ablated: smooth-min aggregation
        }
        for c in (0.3, 1.0, 3.0)
        for penalty in ("l1", "l2")
    ]


def _frs_wide_quantiles_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    """FuzzyRuleSVM with wider quantile anchors (0.25/0.5/0.75)."""
    n_features = dataset.X.shape[1]
    max_rule_length = 2 if n_features <= 32 else 1
    max_rules = min(256, max(24, 3 * n_features))
    return [
        {
            "C": c,
            "penalty": penalty,
            "max_rule_length": max_rule_length,
            "max_rules": max_rules,
            "min_rule_coverage": 0.01,
            "rule_length_penalty": 0.35,
            "partition_quantiles": (0.25, 0.5, 0.75),  # ablated: wider anchors
        }
        for c in (0.3, 1.0, 3.0)
        for penalty in ("l1", "l2")
    ]


def _frs_1term_only_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    """FuzzyRuleSVM with max_rule_length=1 for ALL datasets."""
    n_features = dataset.X.shape[1]
    max_rules = min(256, max(24, 3 * n_features))
    return [
        {
            "C": c,
            "penalty": penalty,
            "max_rule_length": 1,         # ablated: single-antecedent rules only
            "max_rules": max_rules,
            "min_rule_coverage": 0.01,
            "rule_length_penalty": 0.35,
        }
        for c in (0.3, 1.0, 3.0)
        for penalty in ("l1", "l2")
    ]


def _membership_svm_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    del dataset
    return [
        {"C": c, "penalty": penalty}
        for c in (0.3, 1.0, 3.0)
        for penalty in ("l1", "l2")
    ]


def _linear_svc_raw_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    del dataset
    return [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_frs(dataset: PreparedDataset, random_state: int, params: dict[str, Any]) -> Any:
    base = FuzzyRuleSVM(
        **params,
        feature_names=dataset.feature_names,
        class_weight="balanced",
        random_state=random_state,
        max_iter=20000,
    )
    if dataset.spec.task == "multiclass":
        return OneVsRestClassifier(base)
    return base


def _build_membership_svm(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Any:
    base = MembershipSVM(
        **params,
        class_weight="balanced",
        random_state=random_state,
        max_iter=20000,
    )
    if dataset.spec.task == "multiclass":
        return OneVsRestClassifier(base)
    return base


def _build_linear_svc_raw(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Any:
    del dataset
    return LinearSVC(
        **params,
        class_weight="balanced",
        random_state=random_state,
        max_iter=20000,
        dual="auto",
    )


# ---------------------------------------------------------------------------
# Model spec list
# ---------------------------------------------------------------------------

def _model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "frs_default",
            "FRS-Default",
            _build_frs,
            _frs_c_and_penalty_grid,
            uses_standard_scaler=False,
        ),
        ModelSpec(
            "frs_no_penalty",
            "FRS-NoPenalty",
            _build_frs,
            _frs_no_penalty_grid,
            uses_standard_scaler=False,
        ),
        ModelSpec(
            "frs_l2_forced",
            "FRS-L2Forced",
            _build_frs,
            _frs_l2_forced_grid,
            uses_standard_scaler=False,
        ),
        ModelSpec(
            "frs_product_tnorm",
            "FRS-ProductTnorm",
            _build_frs,
            _frs_product_tnorm_grid,
            uses_standard_scaler=False,
        ),
        ModelSpec(
            "frs_softmin_tnorm",
            "FRS-SoftminTnorm",
            _build_frs,
            _frs_softmin_tnorm_grid,
            uses_standard_scaler=False,
        ),
        ModelSpec(
            "frs_wide_quantiles",
            "FRS-WideQuantiles",
            _build_frs,
            _frs_wide_quantiles_grid,
            uses_standard_scaler=False,
        ),
        ModelSpec(
            "frs_1term_only",
            "FRS-1TermOnly",
            _build_frs,
            _frs_1term_only_grid,
            uses_standard_scaler=False,
        ),
        ModelSpec(
            "membership_svm",
            "MembershipSVM",
            _build_membership_svm,
            _membership_svm_grid,
            uses_standard_scaler=False,
        ),
        ModelSpec(
            "linear_svc_raw",
            "LinearSVC-Raw",
            _build_linear_svc_raw,
            _linear_svc_raw_grid,
            uses_standard_scaler=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@contextmanager
def _suppressed_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        yield


def _build_pipeline(spec: ModelSpec, dataset: PreparedDataset, seed: int, params: dict) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if spec.uses_standard_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", spec.builder(dataset, seed, params)))
    return Pipeline(steps)


def _select_params(
    spec: ModelSpec,
    dataset: PreparedDataset,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    inner_splits: int,
    random_state: int,
) -> tuple[dict[str, Any], float]:
    grid = spec.param_grid(dataset)
    if len(grid) == 1:
        return grid[0], float("nan")

    class_counts = np.unique(y_train, return_counts=True)[1]
    splits = min(inner_splits, int(np.min(class_counts)))
    if splits < 2:
        return grid[0], float("nan")

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for i, params in enumerate(grid):
        bal_scores = []
        for fold_i, (inner_train, inner_valid) in enumerate(cv.split(X_train, y_train)):
            pipe = _build_pipeline(spec, dataset, random_state + fold_i, params)
            with _suppressed_warnings():
                pipe.fit(X_train[inner_train], y_train[inner_train])
            y_pred = pipe.predict(X_train[inner_valid])
            bal_scores.append(float(balanced_accuracy_score(y_train[inner_valid], y_pred)))
        scored.append((float(np.mean(bal_scores)), -i, params))

    best_score, _, best_params = max(scored)
    return best_params, best_score


def _evaluate_dataset(
    dataset: PreparedDataset,
    models: list[ModelSpec],
    *,
    outer_splits: int,
    inner_splits: int,
    random_state: int,
) -> list[dict[str, Any]]:
    y = np.asarray(dataset.y)
    class_counts = np.unique(y, return_counts=True)[1]
    n_splits = min(outer_splits, int(np.min(class_counts)))
    if n_splits < 2:
        raise ValueError(f"Dataset {dataset.spec.slug}: too few samples per class.")

    rows = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, test_idx) in enumerate(cv.split(dataset.X, y), start=1):
        X_train, X_test = dataset.X[train_idx], dataset.X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for model_i, spec in enumerate(models):
            seed = random_state + fold * 100 + model_i
            params, _ = _select_params(spec, dataset, X_train, y_train,
                                       inner_splits=inner_splits, random_state=seed)
            pipe = _build_pipeline(spec, dataset, seed, params)

            t0 = time.perf_counter()
            with _suppressed_warnings():
                pipe.fit(X_train, y_train)
            fit_t = time.perf_counter() - t0

            y_pred = pipe.predict(X_test)
            metrics = _predictive_metrics(pipe, X_test, y_test, y_pred)

            rows.append({
                "dataset": dataset.spec.slug,
                "model_key": spec.key,
                "model": spec.name,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_features": dataset.X.shape[1],
                "selected_params": json.dumps(params, sort_keys=True),
                "fit_seconds": fit_t,
                **metrics,
            })
    return rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model_key"])].append(row)

    summaries = []
    for (dataset, model_key), fold_rows in sorted(grouped.items()):
        first = fold_rows[0]
        num_keys = [
            k for k, v in first.items()
            if isinstance(v, int | float | np.integer | np.floating)
            and k not in {"fold", "n_train", "n_test", "n_features"}
        ]
        s: dict[str, Any] = {
            "dataset": dataset,
            "model_key": model_key,
            "model": first["model"],
            "n_folds": len(fold_rows),
            "n_features": first["n_features"],
        }
        for k in num_keys:
            vals = np.asarray([r[k] for r in fold_rows], dtype=float)
            s[f"{k}_mean"] = float(np.nanmean(vals))
            s[f"{k}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        summaries.append(s)
    return summaries


def _wilcoxon_p(diffs: np.ndarray) -> float:
    if np.all(np.isclose(diffs, 0.0)):
        return 1.0
    try:
        return float(wilcoxon(diffs, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    corrected = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        corrected[idx] = min(1.0, p_values[idx] * (n - rank + 1))
        running_max = max(running_max, corrected[idx])
        corrected[idx] = running_max
    return corrected


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

def _run_ablation(
    dataset_slugs: list[str],
    data_dir: Path,
    output_dir: Path,
    report_path: Path,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 0,
    max_samples: int | None = None,
) -> None:
    models = _model_specs()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for i, slug in enumerate(dataset_slugs):
        print(f"  [{i+1}/{len(dataset_slugs)}] {slug} ...", flush=True)
        dataset = load_dataset(slug, data_dir)
        dataset_seed = _stable_dataset_seed(random_state, slug)
        if max_samples and dataset.X.shape[0] > max_samples:
            dataset = _stratified_subset(dataset, max_samples, dataset_seed)
        rows = _evaluate_dataset(
            dataset, models,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            random_state=dataset_seed,
        )
        all_rows.extend(rows)

    summaries = _summarize(all_rows)

    # Organise by model
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in summaries:
        by_model[row["model_key"]][row["dataset"]] = row

    common_datasets = sorted(
        set.intersection(*(set(v.keys()) for v in by_model.values()))
    )

    # Pairwise stats vs frs_default
    ref_key = "frs_default"
    ref = by_model[ref_key]
    metric = "balanced_accuracy_mean"

    pairwise: list[dict[str, Any]] = []
    p_raw: list[float] = []
    for spec in models:
        if spec.key == ref_key:
            continue
        comp = by_model[spec.key]
        ref_vals = np.array([ref[d][metric] for d in common_datasets])
        comp_vals = np.array([comp[d][metric] for d in common_datasets])
        diffs = ref_vals - comp_vals
        p = _wilcoxon_p(diffs)
        wins = int(np.sum(diffs > 0))
        losses = int(np.sum(diffs < 0))
        ties = int(np.sum(np.isclose(diffs, 0)))
        pairwise.append({
            "comparison": f"FRS-Default vs {spec.name}",
            "model_key": spec.key,
            "ref_mean": float(np.mean(ref_vals)),
            "comp_mean": float(np.mean(comp_vals)),
            "mean_delta": float(np.mean(diffs)),
            "median_delta": float(np.median(diffs)),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "wilcoxon_p": p,
        })
        p_raw.append(p)

    # Holm correction across pairwise tests
    corrected = _holm_bonferroni(p_raw)
    for row, cp in zip(pairwise, corrected):
        row["wilcoxon_p_holm"] = cp

    # Write outputs
    _write_csv(output_dir / "fold_metrics.csv", all_rows)
    _write_csv(output_dir / "metrics.csv", summaries)
    _write_csv(output_dir / "pairwise_tests.csv", pairwise)
    (output_dir / "metrics.json").write_text(
        json.dumps(summaries, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pairwise_tests.json").write_text(
        json.dumps(pairwise, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    # Build markdown report
    ref_mean = float(np.mean([ref[d][metric] for d in common_datasets]))
    lines = [
        "# Broader Ablation Suite: FuzzyRuleSVM Design Choices",
        "",
        "Tests eight design dimensions: length penalty, L2 vs L1, aggregation,",
        "quantile anchoring, rule length cap, membership-only baseline,",
        "and raw-features sparse linear baseline.",
        "",
        f"- Datasets evaluated: {len(common_datasets)}",
        f"- Outer CV: {outer_splits}-fold  |  Inner CV: {inner_splits}-fold",
        f"- Reference: FRS-Default (mean balanced accuracy {ref_mean:.3f})",
        "",
        "## Summary: Mean Balanced Accuracy",
        "",
        "| Variant | Mean Bal. Acc. | Δ vs Default | W/T/L | p | p_Holm |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for spec in models:
        comp = by_model[spec.key]
        comp_mean = float(np.mean([comp[d][metric] for d in common_datasets if d in comp]))
        if spec.key == ref_key:
            lines.append(f"| **{spec.name}** | **{comp_mean:.3f}** | --- | --- | --- | --- |")
            continue
        row = next(r for r in pairwise if r["model_key"] == spec.key)
        delta_str = f"{-row['mean_delta']:+.3f}"
        p_str = "<0.001" if row["wilcoxon_p"] < 0.001 else f"{row['wilcoxon_p']:.3f}"
        ph_str = "<0.001" if row["wilcoxon_p_holm"] < 0.001 else f"{row['wilcoxon_p_holm']:.3f}"
        wtl = f"{row['losses']}/{row['ties']}/{row['wins']}"  # wins/ties/losses FOR the variant
        lines.append(f"| {spec.name} | {comp_mean:.3f} | {delta_str} | {wtl} | {p_str} | {ph_str} |")

    lines += [
        "",
        "*W/T/L counts wins/ties/losses FOR the variant (positive Δ = variant beats Default)*",
        "",
        "## Per-Dataset Results (Balanced Accuracy)",
        "",
    ]
    # Header
    col_keys = [spec.key for spec in models]
    col_names = [spec.name for spec in models]
    lines.append("| Dataset | " + " | ".join(col_names) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(col_names)) + "|")
    for d in common_datasets:
        vals = []
        for key in col_keys:
            v = by_model[key].get(d, {}).get(metric, float("nan"))
            vals.append(f"{v:.3f}" if not np.isnan(v) else "---")
        best_idx = int(np.argmax([by_model[k].get(d, {}).get(metric, float("-inf")) for k in col_keys]))
        vals[best_idx] = f"**{vals[best_idx]}**"
        lines.append(f"| {d} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Pairwise Tests (FRS-Default vs each variant, Balanced Accuracy)",
        "",
        "| Comparison | Ref Mean | Var Mean | Δ | Median Δ | W/T/L | p | p_Holm |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in pairwise:
        comp_mean = float(np.mean([by_model[row["model_key"]][d][metric] for d in common_datasets if d in by_model[row["model_key"]]]))
        delta_str = f"{row['mean_delta']:+.3f}"
        med_str = f"{row['median_delta']:+.3f}"
        p_str = "<0.001" if row["wilcoxon_p"] < 0.001 else f"{row['wilcoxon_p']:.3f}"
        ph_str = "<0.001" if row["wilcoxon_p_holm"] < 0.001 else f"{row['wilcoxon_p_holm']:.3f}"
        wtl = f"{row['wins']}/{row['ties']}/{row['losses']}"
        lines.append(
            f"| {row['comparison']} | {row['ref_mean']:.3f} | {comp_mean:.3f} | "
            f"{delta_str} | {med_str} | {wtl} | {p_str} | {ph_str} |"
        )

    lines += [
        "",
        "*Positive Δ = FRS-Default beats the variant; W/T/L counts datasets where Default wins/ties/loses.*",
        "*p_Holm = Holm-Bonferroni corrected p-value across the 8-test family.*",
        "",
        "## Design Choice Interpretation",
        "",
        "- **FRS-NoPenalty**: removing the length penalty (0.35 → 0.0) shows whether"
        " biasing toward shorter rules helps.",
        "- **FRS-L2Forced**: forcing L2 shows whether L1 sparsity is critical.",
        "- **FRS-ProductTnorm / FRS-SoftminTnorm**: replacing minimum aggregation shows"
        " whether the fuzzy AND approximation matters.",
        "- **FRS-WideQuantiles**: using (0.25/0.5/0.75) instead of (0.05/0.5/0.95)"
        " shows whether narrow vs wide anchoring of membership functions matters.",
        "- **FRS-1TermOnly**: forcing single-antecedent rules everywhere shows the value"
        " of two-antecedent conjunctions on low-dimensional datasets.",
        "- **MembershipSVM**: removes rule generation entirely.",
        "- **LinearSVC-Raw**: operates on original standardised features — the strongest"
        " non-fuzzy sparse linear baseline.",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {report_path}")
    print(f"Artifacts: {output_dir}")

    # Console summary
    print("\n=== Mean balanced accuracy per variant ===")
    for spec in models:
        comp = by_model[spec.key]
        comp_mean = float(np.mean([comp[d][metric] for d in common_datasets if d in comp]))
        marker = " <-- reference" if spec.key == ref_key else ""
        print(f"  {spec.name:25s}  {comp_mean:.4f}{marker}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    all_keys: list[str] = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*",
                        help="Dataset slugs (defaults to all).")
    parser.add_argument("--data-dir", default="datasets/prepared")
    parser.add_argument("--output-dir", default="runs/ablation-suite")
    parser.add_argument("--report", default="docs/ablation_suite.md")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)

    slugs = args.datasets or [spec.slug for spec in DATASET_SPECS]
    print(f"Running broader ablation on {len(slugs)} datasets ({len(_model_specs())} variants)...")
    _run_ablation(
        dataset_slugs=slugs,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        report_path=Path(args.report),
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        random_state=args.random_state,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
