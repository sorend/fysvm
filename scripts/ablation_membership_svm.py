"""Ablation: MembershipSVM (no rule generation) vs FuzzyRuleSVM.

Tests whether the conjunctive rule generation step in FuzzyRuleSVM adds
value over simply training L1-LinearSVC directly on the 3×d fuzzy
membership degrees (MembershipSVM).

Usage:
    uv run python scripts/ablation_membership_svm.py
    uv run python scripts/ablation_membership_svm.py --max-samples 400
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
from sklearn.linear_model import LogisticRegression
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
from fysvm.profiling import timed_peak_memory
from fysvm.rule_svm import FuzzyRuleSVM, _FuzzyPartition
from fysvm.run_metadata import write_run_metadata


# ---------------------------------------------------------------------------
# MembershipSVM: L1-LinearSVC on raw fuzzy membership features (no rules)
# ---------------------------------------------------------------------------

class MembershipSVM(ClassifierMixin, BaseEstimator):
    """L1-sparse linear SVM on raw 3×d fuzzy membership features.

    Each numeric feature is mapped to (low, medium, high) membership degrees
    using the same data-adaptive trapezoid partition as FuzzyRuleSVM.
    A LinearSVC with the chosen penalty is then trained on the resulting
    3×n_features activation matrix.  No conjunctive rule generation takes place.

    This ablation isolates the contribution of the rule generation step in
    FuzzyRuleSVM: any performance difference between MembershipSVM and
    FuzzyRuleSVM is attributable solely to the conjunction of features.
    """

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
        if len(self.classes_) == 2:
            svc.fit(M, y)
            self.estimator_ = svc
        else:
            # multiclass handled externally via OneVsRestClassifier
            svc.fit(M, y)
            self.estimator_ = svc

        return self

    def _memberships(self, X: np.ndarray) -> np.ndarray:
        """Compute 3×d membership matrix of shape (n_samples, 3*n_features)."""
        parts = [p.transform(X[:, j]) for j, p in enumerate(self.partitions_)]
        return np.clip(np.hstack(parts), 0.0, 1.0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        M = self._memberships(X)
        return self.estimator_.decision_function(M)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        M = self._memberships(X)
        return self.estimator_.predict(M)

    def nonzero_coef_count(self) -> int:
        """Return the number of nonzero membership-feature coefficients.

        For binary models: counts nonzero entries in the (1, 3d) coefficient
        vector.  For multiclass wrapped in OneVsRestClassifier the caller is
        responsible for aggregating across estimators; this method handles
        only the binary (single-estimator) case.
        """
        check_is_fitted(self)
        coef = self.estimator_.coef_
        if coef.ndim == 2:
            return int(np.any(coef != 0, axis=0).sum())
        return int((coef != 0).sum())


class MembershipLogisticL1(MembershipSVM):
    """L1-logistic regression on the same raw fuzzy membership features."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        partition_quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95),
        class_weight: Any = None,
        random_state: int | None = None,
        max_iter: int = 5000,
        tol: float = 1e-3,
    ) -> None:
        super().__init__(
            C=C,
            penalty="l1",
            partition_quantiles=partition_quantiles,
            class_weight=class_weight,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MembershipLogisticL1":
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

        clf = LogisticRegression(
            C=self.C,
            l1_ratio=1.0,
            solver="saga",
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        clf.fit(M, y)
        self.estimator_ = clf
        return self


# ---------------------------------------------------------------------------
# Complexity helper
# ---------------------------------------------------------------------------

def _membership_svm_complexity(model: Any) -> dict[str, float]:
    """Count nonzero membership-feature coefficients in a fitted MembershipSVM.

    Handles both the binary case (MembershipSVM directly) and the multiclass
    case (OneVsRestClassifier wrapping MembershipSVM).  For multiclass, a
    feature is counted as "active" if it has a nonzero coefficient in *any*
    class-vs-rest binary problem.

    Returns a dict with key ``membership_nonzero_coefs`` (float).
    """
    # Unwrap OneVsRestClassifier if present
    if hasattr(model, "estimators_"):
        coef_rows: list[np.ndarray] = []
        for est in model.estimators_:
            # Each estimator is a MembershipSVM
            inner = est
            if hasattr(inner, "estimator_") and hasattr(inner.estimator_, "coef_"):
                coef = inner.estimator_.coef_
                # Flatten to 1-D per estimator
                if coef.ndim == 2:
                    coef_rows.append((np.any(coef != 0, axis=0)).astype(float))
                else:
                    coef_rows.append((coef != 0).astype(float))
        if coef_rows:
            union_active = np.any(np.vstack(coef_rows), axis=0)
            return {"membership_nonzero_coefs": float(union_active.sum())}
    elif hasattr(model, "estimator_") and hasattr(model.estimator_, "coef_"):
        coef = model.estimator_.coef_
        if coef.ndim == 2:
            n = int(np.any(coef != 0, axis=0).sum())
        else:
            n = int((coef != 0).sum())
        return {"membership_nonzero_coefs": float(n)}
    return {}


# ---------------------------------------------------------------------------
# Comparison framework (minimal, adapted from compare_recommendations.py)
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


def _fuzzy_rule_svm_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
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


def _membership_svm_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    del dataset
    return [
        {"C": c, "penalty": penalty}
        for c in (0.3, 1.0, 3.0)
        for penalty in ("l1", "l2")
    ]


def _membership_logistic_l1_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    del dataset
    return [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]


def _build_fuzzy_rule_svm(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Any:
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


def _build_membership_logistic_l1(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Any:
    base = MembershipLogisticL1(
        **params,
        class_weight="balanced",
        random_state=random_state,
        max_iter=5000,
    )
    if dataset.spec.task == "multiclass":
        return OneVsRestClassifier(base)
    return base


def _model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("fuzzy_rule_svm", "FuzzyRuleSVM", _build_fuzzy_rule_svm, _fuzzy_rule_svm_grid),
        ModelSpec("membership_svm", "MembershipSVM", _build_membership_svm, _membership_svm_grid),
        ModelSpec(
            "membership_logistic_l1",
            "MembershipLogisticL1",
            _build_membership_logistic_l1,
            _membership_logistic_l1_grid,
        ),
    ]


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

            _, fit_t, fit_peak_memory_mb = timed_peak_memory(
                lambda: _fit_with_warnings(pipe, X_train, y_train)
            )
            y_pred, predict_t, predict_peak_memory_mb = timed_peak_memory(
                lambda: pipe.predict(X_test)
            )
            metrics = _predictive_metrics(pipe, X_test, y_test, y_pred)

            extra: dict[str, Any] = {}
            if spec.key in {"membership_svm", "membership_logistic_l1"}:
                extra.update(_membership_svm_complexity(pipe.named_steps["model"]))

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
                "predict_seconds": predict_t,
                "fit_peak_memory_mb": fit_peak_memory_mb,
                "predict_peak_memory_mb": predict_peak_memory_mb,
                **metrics,
                **extra,
            })
    return rows


def _fit_with_warnings(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    with _suppressed_warnings():
        pipe.fit(X, y)
    return pipe


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
    """Return monotone Holm-adjusted p-values in input order."""

    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    corrected = [0.0] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, p_values[index] * (len(p_values) - rank))
        running_max = max(running_max, adjusted)
        corrected[index] = running_max
    return corrected


def _run_comparison(
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
    write_run_metadata(
        output_dir,
        config={
            "datasets": dataset_slugs,
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "random_state": random_state,
            "max_samples": max_samples,
        },
    )

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

    # Statistical tests
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in summaries:
        by_model[row["model_key"]][row["dataset"]] = row

    datasets = sorted(
        set.intersection(*(set(v.keys()) for v in by_model.values()))
    )

    stat_rows: list[dict[str, Any]] = []
    model_pairs = [
        ("fuzzy_rule_svm", "membership_svm"),
        ("fuzzy_rule_svm", "membership_logistic_l1"),
        ("membership_svm", "membership_logistic_l1"),
    ]
    for metric in ["balanced_accuracy", "f1_macro", "accuracy"]:
        key = f"{metric}_mean"
        for left_key, right_key in model_pairs:
            left = by_model[left_key]
            right = by_model[right_key]
            left_vals = np.array([left[d][key] for d in datasets])
            right_vals = np.array([right[d][key] for d in datasets])
            diffs = left_vals - right_vals
            p = _wilcoxon_p(diffs)
            wins = int(np.sum(diffs > 0))
            losses = int(np.sum(diffs < 0))
            ties = int(np.sum(np.isclose(diffs, 0)))
            row = {
                "metric": metric,
                "left_model_key": left_key,
                "right_model_key": right_key,
                "left_model": left[datasets[0]]["model"],
                "right_model": right[datasets[0]]["model"],
                "left_mean": float(np.mean(left_vals)),
                "right_mean": float(np.mean(right_vals)),
                "mean_delta": float(np.mean(diffs)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "wilcoxon_p": p,
            }
            stat_rows.append(row)
            print(
                f"  {row['left_model']} vs {row['right_model']} [{metric}]: "
                f"left={row['left_mean']:.4f} right={row['right_mean']:.4f} "
                f"Δ={row['mean_delta']:+.4f} W/T/L={wins}/{ties}/{losses} p={p:.4f}"
            )

    for metric in ["balanced_accuracy", "f1_macro", "accuracy"]:
        group = [row for row in stat_rows if row["metric"] == metric]
        adjusted = _holm_bonferroni([row["wilcoxon_p"] for row in group])
        for row, p_holm in zip(group, adjusted, strict=True):
            row["wilcoxon_p_holm"] = p_holm

    # Write outputs
    _write_csv(output_dir / "fold_metrics.csv", all_rows)
    _write_csv(output_dir / "metrics.csv", summaries)
    _write_csv(output_dir / "statistical_tests.csv", stat_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(summaries, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    # Build markdown report
    frs = by_model["fuzzy_rule_svm"]
    mem = by_model["membership_svm"]
    log = by_model["membership_logistic_l1"]
    lines = [
        "# Ablation: Membership-Space Linear Baselines vs FuzzyRuleSVM",
        "",
        "Tests whether conjunctive rule generation and the regularised max-margin objective",
        "add value over raw 3×d fuzzy membership features with either LinearSVC or",
        "L1-logistic regression.",
        "",
        f"- Datasets: {len(datasets)}",
        f"- Outer CV: {outer_splits}-fold  |  Inner CV: {inner_splits}-fold",
        "",
        "## Per-Dataset Balanced Accuracy",
        "",
        "| Dataset | FuzzyRuleSVM | MembershipSVM | MembershipLogisticL1 | Δ vs SVM | Δ vs LogReg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for d in datasets:
        fv = frs[d]["balanced_accuracy_mean"]
        mv = mem[d]["balanced_accuracy_mean"]
        lv = log[d]["balanced_accuracy_mean"]
        best = max(fv, mv, lv)
        ftxt = f"**{fv:.3f}**" if np.isclose(fv, best) else f"{fv:.3f}"
        mtxt = f"**{mv:.3f}**" if np.isclose(mv, best) else f"{mv:.3f}"
        ltxt = f"**{lv:.3f}**" if np.isclose(lv, best) else f"{lv:.3f}"
        lines.append(f"| {d} | {ftxt} | {mtxt} | {ltxt} | {fv-mv:+.3f} | {fv-lv:+.3f} |")

    frs_all = np.array([frs[d]["balanced_accuracy_mean"] for d in datasets])
    mem_all = np.array([mem[d]["balanced_accuracy_mean"] for d in datasets])
    log_all = np.array([log[d]["balanced_accuracy_mean"] for d in datasets])

    lines += [
        (
            f"| **Mean** | **{np.mean(frs_all):.3f}** | {np.mean(mem_all):.3f} | "
            f"{np.mean(log_all):.3f} | {np.mean(frs_all - mem_all):+.3f} | "
            f"{np.mean(frs_all - log_all):+.3f} |"
        ),
        "",
        "## Wilcoxon Signed-Rank Test (Balanced Accuracy)",
        "",
        "| Comparison | Left Mean | Right Mean | Δ | W/T/L | p_raw | p_Holm |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in stat_rows:
        if row["metric"] != "balanced_accuracy":
            continue
        lines.append(
            f"| {row['left_model']} vs {row['right_model']} | {row['left_mean']:.3f} | "
            f"{row['right_mean']:.3f} | {row['mean_delta']:+.3f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {row['wilcoxon_p']:.4f} | "
            f"{row['wilcoxon_p_holm']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- **Δ > 0**: the left model outperforms the right model.",
        "- **Δ ≈ 0**: the two models are empirically tied under this protocol.",
        "- **Δ < 0**: the right model is stronger under this protocol.",
        "",
        "A significant positive FuzzyRuleSVM--MembershipLogisticL1 result would support",
        "a max-margin contribution beyond regularised classification on membership features.",
        "A non-significant result limits the accuracy claim to the fuzzy feature mapping",
        "and leaves the exact rule-space decomposition as the primary contribution.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {report_path}")
    print(f"Artifacts: {output_dir}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    # Collect all keys across all rows so model-specific columns are preserved.
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
    parser.add_argument("--output-dir", default="runs/ablation-membership")
    parser.add_argument("--report", default="docs/ablation_membership.md")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)

    slugs = args.datasets or [spec.slug for spec in DATASET_SPECS]
    print(f"Running ablation on {len(slugs)} datasets...")
    _run_comparison(
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
