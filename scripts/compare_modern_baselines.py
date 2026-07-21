"""Compare FuzzyRuleSVM against modern interpretable tabular baselines.

Baselines included:
- EBM (Explainable Boosting Machine, InterpretML / Nori et al. 2019)
- RuleFit (Friedman & Popescu 2008) via the imodels library

All models run under the same 5-fold outer / 3-fold inner nested-CV protocol
used in the main standard-baselines comparison. Missing values are imputed
with the training-fold median. No StandardScaler is applied to any model
(EBM and tree-based rule generation are scale-invariant; FuzzyRuleSVM builds
its own fuzzy partitions from training quantiles).

EBM is trained with balanced sample weights when the installed InterpretML
version accepts ``sample_weight``; balanced_accuracy remains the primary metric.
EBM on high-dimensional datasets (>100 features) uses interactions=0 to limit
compute.

RuleFit is wrapped in OneVsRestClassifier for multiclass problems, matching
the treatment of FuzzyRuleSVM.

Complexity metrics collected:
- EBM: term_count  (number of feature terms selected, including interactions)
- RuleFit: nonzero_rules  (number of rules with nonzero coefficient)
"""

from __future__ import annotations

import argparse
import csv
import json
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from fysvm.datasets import DATASET_SPECS, PreparedDataset, load_dataset
from fysvm.evaluation import (
    _fuzzy_rule_metrics,
    _json_default,
    _mean_pairwise_jaccard,
    _predictive_metrics,
    _stable_dataset_seed,
    _stratified_subset,
    _support_rule_set,
)
from fysvm.profiling import timed_peak_memory
from fysvm.rule_svm import FuzzyRuleSVM
from fysvm.run_metadata import write_run_metadata


ModelBuilder = Callable[[PreparedDataset, int, dict[str, Any]], Any]
ParamGridBuilder = Callable[[PreparedDataset], list[dict[str, Any]]]


@dataclass(frozen=True)
class ModelSpec:
    """One comparable estimator family."""

    key: str
    name: str
    builder: ModelBuilder
    param_grid: ParamGridBuilder
    has_rule_metrics: bool = False
    uses_standard_scaler: bool = False


@dataclass(frozen=True)
class ComparisonResult:
    """Files and in-memory rows produced by a comparison run."""

    output_dir: Path
    report_path: Path
    fold_metrics: list[dict[str, Any]]
    summary_metrics: list[dict[str, Any]]
    statistical_tests: list[dict[str, Any]]


def run_modern_comparison(
    *,
    dataset_slugs: Iterable[str] | None = None,
    data_dir: str | Path = "datasets/prepared",
    output_dir: str | Path = "runs/modern-baselines-comparison",
    report_path: str | Path = "docs/modern_baselines_comparison.md",
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 0,
    max_samples: int | None = None,
) -> ComparisonResult:
    """Run nested-CV comparisons and write CSV, JSON, and Markdown artifacts."""

    if outer_splits < 2:
        raise ValueError("outer_splits must be at least 2.")
    if inner_splits < 2:
        raise ValueError("inner_splits must be at least 2.")

    selected = list(dataset_slugs) if dataset_slugs is not None else [
        spec.slug for spec in DATASET_SPECS
    ]
    models = _model_specs()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_run_metadata(
        output_path,
        config={
            "datasets": selected,
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "random_state": random_state,
            "max_samples": max_samples,
            "ebm_imbalance_policy": "model__sample_weight=balanced when supported",
        },
    )

    fold_metrics: list[dict[str, Any]] = []
    rule_sets: dict[tuple[str, str], list[set[str]]] = defaultdict(list)

    for dataset_index, slug in enumerate(selected):
        dataset = load_dataset(slug, data_dir)
        dataset_seed = _stable_dataset_seed(random_state, slug)
        if max_samples is not None and dataset.X.shape[0] > max_samples:
            dataset = _stratified_subset(dataset, max_samples, dataset_seed)
        print(f"  [{dataset_index+1}/{len(selected)}] {dataset.spec.name} "
              f"({dataset.X.shape[0]} x {dataset.X.shape[1]}, {dataset.spec.task})")
        rows = _evaluate_dataset(
            dataset,
            models,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            random_state=dataset_seed,
            rule_sets=rule_sets,
        )
        fold_metrics.extend(rows)

    summary_metrics = _summarize_metrics(fold_metrics, rule_sets)
    statistical_tests = _statistical_tests(summary_metrics, models)

    _write_csv(output_path / "fold_metrics.csv", fold_metrics)
    _write_csv(output_path / "metrics.csv", summary_metrics)
    _write_csv(output_path / "statistical_tests.csv", statistical_tests)
    (output_path / "metrics.json").write_text(
        json.dumps(
            {
                "fold_metrics": fold_metrics,
                "summary_metrics": summary_metrics,
                "statistical_tests": statistical_tests,
            },
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )

    report = _build_markdown_report(
        summary_metrics,
        statistical_tests,
        selected,
        models,
        output_path,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        random_state=random_state,
        max_samples=max_samples,
    )
    report_output_path = Path(report_path)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.write_text(report, encoding="utf-8")

    return ComparisonResult(
        output_path,
        report_output_path,
        fold_metrics,
        summary_metrics,
        statistical_tests,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Prepared dataset slugs to evaluate. Defaults to all datasets.",
    )
    parser.add_argument("--data-dir", default="datasets/prepared")
    parser.add_argument("--output-dir", default="runs/modern-baselines-comparison")
    parser.add_argument("--report", default="docs/modern_baselines_comparison.md")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional stratified cap for quick smoke runs.",
    )
    args = parser.parse_args(argv)

    result = run_modern_comparison(
        dataset_slugs=args.datasets or None,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        random_state=args.random_state,
        max_samples=args.max_samples,
    )
    print(result.report_path)
    print(result.output_dir)


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------

def _model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "fuzzy_rule_svm",
            "FuzzyRuleSVM",
            _build_fuzzy_rule_svm,
            _fuzzy_rule_svm_grid,
            has_rule_metrics=True,
            uses_standard_scaler=False,
        ),
        ModelSpec(
            "ebm",
            "EBM",
            _build_ebm,
            _ebm_grid,
            has_rule_metrics=False,
            uses_standard_scaler=False,
        ),
        ModelSpec(
            "rulefit",
            "RuleFit",
            _build_rulefit,
            _rulefit_grid,
            has_rule_metrics=False,
            uses_standard_scaler=False,
        ),
    ]


# ---------------------------------------------------------------------------
# FuzzyRuleSVM
# ---------------------------------------------------------------------------

def _fuzzy_rule_svm_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    n_features = dataset.X.shape[1]
    if n_features > 100:
        return [
            {
                "C": 1.0,
                "penalty": "l1",
                "max_rule_length": 2,
                "max_rules": max_rules,
                "min_rule_coverage": 0.01,
                "rule_length_penalty": 0.35,
                "feature_screening": "anova",
                "screen_top_k": screen_top_k,
            }
            for max_rules in (64, 128, 256, 512, 1024, 2048)
            for screen_top_k in (16, 32)
        ]
    max_rule_length = 2 if n_features <= 32 else 1
    max_rules_options = sorted({min(256, max(24, 3 * n_features)), 512})
    return [
        {
            "C": c,
            "penalty": penalty,
            "max_rule_length": max_rule_length,
            "max_rules": max_rules,
            "min_rule_coverage": 0.01,
            "rule_length_penalty": 0.35,
        }
        for max_rules in max_rules_options
        for c in (0.3, 1.0, 3.0)
        for penalty in ("l1", "l2")
    ]


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


# ---------------------------------------------------------------------------
# EBM (Explainable Boosting Machine)
# ---------------------------------------------------------------------------

def _ebm_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    """Two-point grid: with and without pairwise interactions.

    High-dimensional datasets (> 100 features) use interactions=0 only
    because pairwise interaction search over 100+ features is prohibitively slow.
    """
    n_features = dataset.X.shape[1]
    if n_features > 100:
        # High-dimensional: main effects only, reduced bags
        return [{"interactions": 0, "max_rounds": 200, "outer_bags": 4, "inner_bags": 0}]
    return [
        {"interactions": 0, "max_rounds": 300, "outer_bags": 8, "inner_bags": 0},
        {"interactions": 5, "max_rounds": 300, "outer_bags": 8, "inner_bags": 0},
    ]


def _build_ebm(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Any:
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(
        **params,
        n_jobs=1,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# RuleFit
# ---------------------------------------------------------------------------

def _rulefit_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    """Two-point grid: moderate and generous rule budgets."""
    n_features = dataset.X.shape[1]
    # On high-dimensional datasets reduce n_estimators to limit rule explosion
    n_estimators = 50 if n_features > 100 else 100
    return [
        {"n_estimators": n_estimators, "max_rules": 100, "tree_size": 4},
        {"n_estimators": n_estimators, "max_rules": 200, "tree_size": 4},
    ]


def _build_rulefit(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Any:
    from imodels import RuleFitClassifier
    base = _RuleFitWrapper(
        feature_names=dataset.feature_names,
        random_state=random_state,
        **params,
    )
    if dataset.spec.task == "multiclass":
        return OneVsRestClassifier(base)
    return base


class _RuleFitWrapper:
    """sklearn-compatible wrapper around imodels.RuleFitClassifier.

    Implements the minimal sklearn interface needed for Pipeline compatibility
    with sklearn >= 1.8 (get_params, set_params, __sklearn_tags__).
    Stores feature_names for fitting and exposes a nonzero_rules_ attribute
    that counts rules with non-zero coefficients after L1-regularised fitting.
    """

    def __init__(
        self,
        *,
        feature_names: list[str] | None = None,
        n_estimators: int = 100,
        max_rules: int = 100,
        tree_size: int = 4,
        random_state: int = 0,
    ) -> None:
        self.feature_names = feature_names
        self.n_estimators = n_estimators
        self.max_rules = max_rules
        self.tree_size = tree_size
        self.random_state = random_state

    def _make_model(self) -> Any:
        from imodels import RuleFitClassifier
        return RuleFitClassifier(
            n_estimators=self.n_estimators,
            max_rules=self.max_rules,
            tree_size=self.tree_size,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RuleFitWrapper":
        self._model = self._make_model()
        with _suppressed_warnings():
            self._model.fit(X, y, feature_names=self.feature_names)
        # Count nonzero rule coefficients
        coefs = self._model.coef
        self.nonzero_rules_: int = int(sum(abs(c) > 1e-10 for c in coefs))
        self.total_rules_: int = len(coefs)
        self.classes_ = self._model.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def __sklearn_tags__(self) -> Any:
        from sklearn.utils._tags import ClassifierTags, Tags, TargetTags
        return Tags(
            estimator_type="classifier",
            target_tags=TargetTags(required=True),
            classifier_tags=ClassifierTags(),
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "n_estimators": self.n_estimators,
            "max_rules": self.max_rules,
            "tree_size": self.tree_size,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "_RuleFitWrapper":
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _evaluate_dataset(
    dataset: PreparedDataset,
    models: list[ModelSpec],
    *,
    outer_splits: int,
    inner_splits: int,
    random_state: int,
    rule_sets: dict[tuple[str, str], list[set[str]]],
) -> list[dict[str, Any]]:
    y = np.asarray(dataset.y)
    class_counts = np.unique(y, return_counts=True)[1]
    splits = min(outer_splits, int(np.min(class_counts)))
    if splits < 2:
        raise ValueError(
            f"Dataset {dataset.spec.slug} has a class with fewer than two samples."
        )

    rows: list[dict[str, Any]] = []
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    for fold, (train_index, test_index) in enumerate(cv.split(dataset.X, y), start=1):
        X_train, X_test = dataset.X[train_index], dataset.X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for model_index, model_spec in enumerate(models):
            model_seed = random_state + fold * 100 + model_index
            selected_params, inner_score, inner_f1 = _select_params(
                model_spec,
                dataset,
                X_train,
                y_train,
                inner_splits=inner_splits,
                random_state=model_seed,
            )
            pipeline = _build_pipeline(model_spec, dataset, model_seed, selected_params)

            ebm_weighted = False
            _, fit_seconds, fit_peak_memory_mb = timed_peak_memory(
                lambda: _fit_pipeline(
                    pipeline,
                    X_train,
                    y_train,
                    use_balanced_sample_weight=model_spec.key == "ebm",
                )
            )
            if model_spec.key == "ebm":
                ebm_weighted = bool(getattr(pipeline.named_steps["model"], "_fysvm_weighted_fit", False))

            y_pred, predict_seconds, predict_peak_memory_mb = timed_peak_memory(
                lambda: pipeline.predict(X_test)
            )

            metrics: dict[str, Any] = {
                "dataset": dataset.spec.slug,
                "dataset_name": dataset.spec.name,
                "task": dataset.spec.task,
                "model_key": model_spec.key,
                "model": model_spec.name,
                "fold": fold,
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                "n_features": int(dataset.X.shape[1]),
                "n_classes": int(len(np.unique(y))),
                "inner_balanced_accuracy": inner_score,
                "inner_f1_macro": inner_f1,
                "selected_params": json.dumps(selected_params, sort_keys=True),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "fit_peak_memory_mb": fit_peak_memory_mb,
                "predict_peak_memory_mb": predict_peak_memory_mb,
                "ebm_balanced_sample_weight": ebm_weighted,
                **_predictive_metrics(pipeline, X_test, y_test, y_pred),
            }

            if model_spec.has_rule_metrics:
                X_test_prepared = pipeline.named_steps["imputer"].transform(X_test)
                model = pipeline.named_steps["model"]
                metrics.update(_fuzzy_rule_metrics(model, X_test_prepared, y_test))
                rule_sets[(dataset.spec.slug, model_spec.key)].append(_support_rule_set(model))

            # EBM-specific complexity metric
            if model_spec.key == "ebm":
                ebm_model = pipeline.named_steps["model"]
                try:
                    metrics["ebm_term_count"] = int(len(ebm_model.term_names_))
                except AttributeError:
                    pass

            # RuleFit-specific complexity metric
            if model_spec.key == "rulefit":
                rulefit_model = pipeline.named_steps["model"]
                try:
                    if isinstance(rulefit_model, OneVsRestClassifier):
                        # Average nonzero rules across OvR estimators
                        nz_list = [
                            est.nonzero_rules_
                            for est in rulefit_model.estimators_
                            if hasattr(est, "nonzero_rules_")
                        ]
                        if nz_list:
                            metrics["rulefit_nonzero_rules"] = float(np.mean(nz_list))
                    else:
                        metrics["rulefit_nonzero_rules"] = float(
                            rulefit_model.nonzero_rules_
                        )
                except AttributeError:
                    pass

            rows.append(metrics)
    return rows


def _fit_pipeline(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    *,
    use_balanced_sample_weight: bool = False,
) -> Pipeline:
    with _suppressed_warnings():
        if use_balanced_sample_weight:
            weights = compute_sample_weight("balanced", y)
            try:
                pipeline.fit(X, y, model__sample_weight=weights)
                setattr(pipeline.named_steps["model"], "_fysvm_weighted_fit", True)
                return pipeline
            except TypeError as exc:
                raise RuntimeError(
                    "Balanced EBM fitting requires sample_weight support; "
                    "refusing an unweighted fallback."
                ) from exc
        pipeline.fit(X, y)
    return pipeline


def _select_params(
    model_spec: ModelSpec,
    dataset: PreparedDataset,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    inner_splits: int,
    random_state: int,
) -> tuple[dict[str, Any], float, float]:
    grid = model_spec.param_grid(dataset)
    if len(grid) == 1:
        return grid[0], float("nan"), float("nan")

    class_counts = np.unique(y_train, return_counts=True)[1]
    splits = min(inner_splits, int(np.min(class_counts)))
    if splits < 2:
        return grid[0], float("nan"), float("nan")

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    scored: list[tuple[float, float, int, dict[str, Any]]] = []
    for param_index, params in enumerate(grid):
        balanced_scores: list[float] = []
        f1_scores: list[float] = []
        for fold_i, (inner_train, inner_valid) in enumerate(
            cv.split(X_train, y_train), start=1
        ):
            pipeline = _build_pipeline(
                model_spec,
                dataset,
                random_state + fold_i,
                params,
            )
            _fit_pipeline(
                pipeline,
                X_train[inner_train],
                y_train[inner_train],
                use_balanced_sample_weight=model_spec.key == "ebm",
            )
            y_pred = pipeline.predict(X_train[inner_valid])
            y_valid = y_train[inner_valid]
            balanced_scores.append(
                float(balanced_accuracy_score(y_valid, y_pred))
            )
            f1_scores.append(
                float(f1_score(y_valid, y_pred, average="macro", zero_division=0))
            )
        scored.append(
            (
                float(np.mean(balanced_scores)),
                float(np.mean(f1_scores)),
                -param_index,
                params,
            )
        )

    best_balanced, best_f1, _, best_params = max(scored, key=lambda item: item[:3])
    return best_params, best_balanced, best_f1


def _build_pipeline(
    model_spec: ModelSpec,
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    steps.append(("model", model_spec.builder(dataset, random_state, params)))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Aggregation and statistics
# ---------------------------------------------------------------------------

def _summarize_metrics(
    rows: list[dict[str, Any]],
    rule_sets: dict[tuple[str, str], list[set[str]]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model_key"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (dataset, model_key), fold_rows in sorted(grouped.items()):
        first = fold_rows[0]
        numeric_keys = sorted(
            key
            for key, value in first.items()
            if isinstance(value, int | float | np.integer | np.floating)
            and key not in {"fold", "n_train", "n_test", "n_features", "n_classes"}
        )
        summary: dict[str, Any] = {
            "dataset": dataset,
            "dataset_name": first["dataset_name"],
            "task": first["task"],
            "model_key": model_key,
            "model": first["model"],
            "n_samples": int(sum(row["n_test"] for row in fold_rows)),
            "n_features": int(first["n_features"]),
            "n_classes": int(first["n_classes"]),
            "n_folds": int(len(fold_rows)),
        }
        if (dataset, model_key) in rule_sets:
            summary["support_rule_jaccard"] = _mean_pairwise_jaccard(
                rule_sets[(dataset, model_key)]
            )
        selected = Counter(row["selected_params"] for row in fold_rows)
        summary["selected_params_mode"] = selected.most_common(1)[0][0]
        summary["selected_params_unique"] = int(len(selected))
        for key in numeric_keys:
            values = np.asarray(
                [row[key] for row in fold_rows if key in row], dtype=float
            )
            if values.size == 0:
                continue
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
        summaries.append(summary)
    return summaries


def _statistical_tests(
    summaries: list[dict[str, Any]],
    models: list[ModelSpec],
) -> list[dict[str, Any]]:
    by_model = {
        model.key: {
            row["dataset"]: row
            for row in summaries
            if row["model_key"] == model.key
        }
        for model in models
    }
    datasets = sorted(
        set.intersection(*(set(rows) for rows in by_model.values()))
    )
    metrics = ["balanced_accuracy_mean", "f1_macro_mean"]
    tests: list[dict[str, Any]] = []

    fuzzy_rows = by_model["fuzzy_rule_svm"]
    pairwise_rows: list[dict[str, Any]] = []
    for model in models:
        if model.key == "fuzzy_rule_svm":
            continue
        baseline_rows = by_model[model.key]
        for metric in metrics:
            fuzzy_values = np.asarray(
                [fuzzy_rows[dataset][metric] for dataset in datasets], dtype=float
            )
            baseline_values = np.asarray(
                [baseline_rows[dataset][metric] for dataset in datasets], dtype=float
            )
            diffs = fuzzy_values - baseline_values
            p_value = _wilcoxon_pvalue(diffs)
            pairwise_rows.append(
                {
                    "comparison": f"FuzzyRuleSVM vs {model.name}",
                    "metric": metric.removesuffix("_mean"),
                    "n_datasets": int(len(datasets)),
                    "fuzzy_mean": float(np.mean(fuzzy_values)),
                    "baseline_mean": float(np.mean(baseline_values)),
                    "mean_delta": float(np.mean(diffs)),
                    "median_delta": float(np.median(diffs)),
                    "wins": int(np.sum(diffs > 0)),
                    "ties": int(np.sum(np.isclose(diffs, 0.0))),
                    "losses": int(np.sum(diffs < 0)),
                    "wilcoxon_pvalue": p_value,
                }
            )

    # Apply Holm–Bonferroni correction within each metric group.
    for metric in metrics:
        metric_key = metric.removesuffix("_mean")
        group = [r for r in pairwise_rows if r["metric"] == metric_key]
        p_values = [r["wilcoxon_pvalue"] for r in group]
        holm_p_values = _holm_bonferroni(p_values)
        for row, holm_p in zip(group, holm_p_values):
            row["wilcoxon_pvalue_holm"] = holm_p

    tests.extend(pairwise_rows)

    for metric in metrics:
        values = [
            np.asarray(
                [by_model[model.key][dataset][metric] for dataset in datasets],
                dtype=float,
            )
            for model in models
        ]
        tests.append(
            {
                "comparison": "Friedman omnibus",
                "metric": metric.removesuffix("_mean"),
                "n_datasets": int(len(datasets)),
                "models": ", ".join(model.name for model in models),
                "friedman_statistic": _friedman_statistic(values),
                "friedman_pvalue": _friedman_pvalue(values),
            }
        )
    return tests


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm–Bonferroni step-down correction to a list of p-values.

    Returns corrected p-values in the same order as the input.

    Reference: Holm (1979), "A Simple Sequentially Rejective Multiple Test
    Procedure", Scandinavian Journal of Statistics 6(2): 65–70.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    corrected = [0.0] * n
    for rank, idx in enumerate(order, start=1):
        corrected[idx] = min(1.0, p_values[idx] * (n - rank + 1))
    running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        running_max = max(running_max, corrected[idx])
        corrected[idx] = running_max
    return corrected


def _wilcoxon_pvalue(diffs: np.ndarray) -> float:
    if np.all(np.isclose(diffs, 0.0)):
        return 1.0
    try:
        return float(wilcoxon(diffs, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def _friedman_statistic(values: list[np.ndarray]) -> float:
    try:
        return float(friedmanchisquare(*values).statistic)
    except ValueError:
        return float("nan")


def _friedman_pvalue(values: list[np.ndarray]) -> float:
    try:
        return float(friedmanchisquare(*values).pvalue)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _build_markdown_report(
    summaries: list[dict[str, Any]],
    tests: list[dict[str, Any]],
    dataset_slugs: list[str],
    models: list[ModelSpec],
    output_dir: Path,
    *,
    outer_splits: int,
    inner_splits: int,
    random_state: int,
    max_samples: int | None,
) -> str:
    by_model = {
        model.key: [row for row in summaries if row["model_key"] == model.key]
        for model in models
    }
    by_dataset = _summary_by_dataset_and_model(summaries)

    lines = [
        "# Modern Interpretable Baselines Comparison",
        "",
        "## Scope",
        "",
        "Comparison of FuzzyRuleSVM against modern interpretable tabular baselines:",
        "",
        "- **EBM** (Explainable Boosting Machine; Nori et al. 2019): a glass-box GAM trained",
        "  via boosting over pairwise interactions. Tuned over interactions ∈ {0, 5}.",
        "- **RuleFit** (Friedman & Popescu 2008): sparse linear model over tree-derived rules.",
        "  Tuned over max_rules ∈ {100, 200}.",
        "- **FuzzyRuleSVM** (this work): regularised max-margin model over fuzzy linguistic",
        "  rule activations. Tuned over C ∈ {0.3, 1.0, 3.0} × penalty ∈ {L1, L2}.",
        "",
        "Note: EBM receives balanced sample weights on every fit. RuleFit is wrapped in",
        "OneVsRestClassifier for multiclass datasets.",
        "",
        "## Protocol",
        "",
        f"- Datasets: {len(dataset_slugs)} prepared datasets from `datasets/prepared`.",
        f"- Outer evaluation: stratified {outer_splits}-fold CV.",
        f"- Inner selection: stratified {inner_splits}-fold CV on each outer training fold.",
        "- Selection metric: mean inner balanced accuracy, macro F1 as tie-breaker.",
        "- Missing values: median imputation fitted inside each fold.",
        "- No StandardScaler applied (all models are scale-invariant or use own normalisation).",
        f"- Random state: `{random_state}`.",
        f"- Sample cap: `{max_samples}`." if max_samples is not None else "- Sample cap: none.",
        "",
        "## Overall Performance",
        "",
    ]

    overall_rows = []
    for model in models:
        rows = by_model[model.key]
        overall_rows.append(
            [
                model.name,
                _fmt(np.mean([row["balanced_accuracy_mean"] for row in rows])),
                _fmt(np.mean([row["f1_macro_mean"] for row in rows])),
                _fmt(np.mean([row["fit_seconds_mean"] for row in rows])),
            ]
        )
    lines.extend(
        _markdown_table(
            ["Model", "Mean Balanced Acc.", "Mean Macro F1", "Mean Fit Time (s)"],
            overall_rows,
        )
    )

    lines.extend([
        "",
        "## Paired Tests (FuzzyRuleSVM vs Baselines)",
        "",
        "Wilcoxon signed-rank tests over 20 per-dataset means. Positive Δ favours FuzzyRuleSVM.",
        "",
    ])
    paired_rows = [
        row for row in tests
        if row["comparison"].startswith("FuzzyRuleSVM")
        and row["metric"] in {"balanced_accuracy", "f1_macro"}
    ]
    lines.extend(
        _markdown_table(
            ["Comparison", "Metric", "FRS Mean", "Base Mean", "Δ", "W/T/L", "p"],
            [
                [
                    row["comparison"],
                    row["metric"],
                    _fmt(row["fuzzy_mean"]),
                    _fmt(row["baseline_mean"]),
                    _fmt_signed(row["mean_delta"]),
                    f"{row['wins']}/{row['ties']}/{row['losses']}",
                    _fmt_p(row["wilcoxon_pvalue"]),
                ]
                for row in paired_rows
            ],
        )
    )

    friedman_rows = [row for row in tests if row["comparison"] == "Friedman omnibus"]
    lines.extend(["", "Friedman omnibus across all three models:", ""])
    lines.extend(
        _markdown_table(
            ["Metric", "Statistic", "p-value"],
            [
                [row["metric"], _fmt(row["friedman_statistic"]), _fmt_p(row["friedman_pvalue"])]
                for row in friedman_rows
            ],
        )
    )

    lines.extend([
        "",
        "## Per-Dataset Results (Balanced Accuracy)",
        "",
    ])
    dataset_rows = []
    for dataset_slug in sorted(by_dataset.keys()):
        rows = by_dataset[dataset_slug]
        frs_ba = rows.get("fuzzy_rule_svm", {}).get("balanced_accuracy_mean", float("nan"))
        ebm_ba = rows.get("ebm", {}).get("balanced_accuracy_mean", float("nan"))
        rulefit_ba = rows.get("rulefit", {}).get("balanced_accuracy_mean", float("nan"))
        dataset_name = next(
            (r["dataset_name"] for r in rows.values()), dataset_slug
        )
        dataset_rows.append([
            dataset_name,
            _fmt(frs_ba),
            _fmt(ebm_ba),
            _fmt(rulefit_ba),
            _fmt_signed(frs_ba - max(ebm_ba, rulefit_ba)),
        ])
    lines.extend(
        _markdown_table(
            ["Dataset", "FuzzyRuleSVM", "EBM", "RuleFit", "FRS - Best"],
            dataset_rows,
        )
    )

    lines.extend([
        "",
        "## Model Complexity",
        "",
        "FuzzyRuleSVM: mean active support rules across datasets.",
        "EBM: mean number of feature terms (main effects + interactions).",
        "RuleFit: mean nonzero rules across datasets.",
        "",
    ])
    frs_rules = np.mean([
        row.get("rule_support_rule_count_mean", float("nan"))
        for row in by_model["fuzzy_rule_svm"]
    ])
    ebm_terms = np.mean([
        row.get("ebm_term_count_mean", float("nan"))
        for row in by_model["ebm"]
        if not np.isnan(row.get("ebm_term_count_mean", float("nan")))
    ])
    rulefit_nz = np.mean([
        row.get("rulefit_nonzero_rules_mean", float("nan"))
        for row in by_model["rulefit"]
        if not np.isnan(row.get("rulefit_nonzero_rules_mean", float("nan")))
    ])
    lines.extend(
        _markdown_table(
            ["Model", "Mean Complexity"],
            [
                ["FuzzyRuleSVM (support rules)", _fmt(frs_rules)],
                ["EBM (feature terms)", _fmt(ebm_terms)],
                ["RuleFit (nonzero rules)", _fmt(rulefit_nz)],
            ],
        )
    )

    lines.extend(["", "## Conclusion", ""])
    frs_mean = np.mean([
        row["balanced_accuracy_mean"] for row in by_model["fuzzy_rule_svm"]
    ])
    ebm_mean = np.mean([
        row["balanced_accuracy_mean"] for row in by_model["ebm"]
    ])
    rulefit_mean = np.mean([
        row["balanced_accuracy_mean"] for row in by_model["rulefit"]
    ])
    lines.append(
        f"FuzzyRuleSVM mean balanced accuracy: {_fmt(frs_mean)}. "
        f"EBM: {_fmt(ebm_mean)}. RuleFit: {_fmt(rulefit_mean)}. "
        "These results position FuzzyRuleSVM within the modern interpretable "
        "tabular ML landscape. See statistical tests for pairwise comparisons."
    )
    lines.append("")

    return "\n".join(lines)


def _summary_by_dataset_and_model(
    summaries: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        by_dataset[row["dataset"]][row["model_key"]] = row
    return dict(by_dataset)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    rendered = ["| " + " | ".join(headers) + " |"]
    rendered.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        rendered.append("| " + " | ".join(str(item) for item in row) + " |")
    return rendered


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(number):
        return "nan"
    if abs(number) < 0.0005 and number != 0.0:
        return f"{number:.2e}"
    return f"{number:.3f}"


def _fmt_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def _fmt_p(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def _suppressed_warnings() -> Iterable[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


if __name__ == "__main__":
    main()
