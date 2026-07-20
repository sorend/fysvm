"""Run the comparison experiments recommended in docs/findings.md."""

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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

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
    uses_standard_scaler: bool = True


@dataclass(frozen=True)
class ComparisonResult:
    """Files and in-memory rows produced by a comparison run."""

    output_dir: Path
    report_path: Path
    fold_metrics: list[dict[str, Any]]
    summary_metrics: list[dict[str, Any]]
    statistical_tests: list[dict[str, Any]]


def run_recommended_comparison(
    *,
    dataset_slugs: Iterable[str] | None = None,
    data_dir: str | Path = "datasets/prepared",
    output_dir: str | Path = "runs/recommended-comparison",
    report_path: str | Path = "docs/comparison.md",
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
    if max_samples is not None and max_samples < 2:
        raise ValueError("max_samples must be None or at least 2.")

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
        },
    )

    fold_metrics: list[dict[str, Any]] = []
    rule_sets: dict[tuple[str, str], list[set[str]]] = defaultdict(list)

    for dataset_index, slug in enumerate(selected):
        dataset = load_dataset(slug, data_dir)
        dataset_seed = _stable_dataset_seed(random_state, slug)
        if max_samples is not None and dataset.X.shape[0] > max_samples:
            dataset = _stratified_subset(dataset, max_samples, dataset_seed)
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
    parser.add_argument("--output-dir", default="runs/recommended-comparison")
    parser.add_argument("--report", default="docs/comparison.md")
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

    result = run_recommended_comparison(
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
            "linear_svm",
            "Linear SVM",
            _build_linear_svm,
            _regularized_grid,
        ),
        ModelSpec(
            "rbf_svm",
            "RBF SVM",
            _build_rbf_svm,
            _rbf_svm_grid,
        ),
        ModelSpec(
            "logistic_l2",
            "Logistic Regression L2",
            _build_logistic_l2,
            _regularized_grid,
        ),
        ModelSpec(
            "logistic_l1",
            "Logistic Regression L1",
            _build_logistic_l1,
            _regularized_grid,
        ),
        ModelSpec(
            "logistic_elasticnet",
            "Elastic-Net Logistic Regression",
            _build_logistic_elasticnet,
            _elastic_net_logistic_grid,
        ),
        ModelSpec(
            "random_forest",
            "Random Forest",
            _build_random_forest,
            _random_forest_grid,
            uses_standard_scaler=False,
        ),
        ModelSpec(
            "hist_gradient_boosting",
            "Histogram Gradient Boosting",
            _build_hist_gradient_boosting,
            _hist_gradient_boosting_grid,
            uses_standard_scaler=False,
        ),
    ]


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


def _regularized_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    del dataset
    return [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]


def _rbf_svm_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    del dataset
    return [
        {"C": c, "gamma": gamma}
        for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
        for gamma in ("scale", "auto", 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
    ]


def _elastic_net_logistic_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    del dataset
    return [
        {"C": c, "l1_ratio": l1_ratio}
        for c in (0.01, 0.1, 1.0, 10.0)
        for l1_ratio in (0.25, 0.5, 0.75)
    ]


def _random_forest_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    del dataset
    return [
        {"n_estimators": n_estimators, "max_features": max_features, "min_samples_leaf": leaf}
        for n_estimators in (200,)
        for max_features in ("sqrt", 0.5)
        for leaf in (1, 3)
    ]


def _hist_gradient_boosting_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    del dataset
    return [
        {"learning_rate": learning_rate, "max_leaf_nodes": max_leaf_nodes, "l2_regularization": l2}
        for learning_rate in (0.03, 0.1)
        for max_leaf_nodes in (15, 31)
        for l2 in (0.0, 0.1)
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


def _build_linear_svm(
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


def _build_rbf_svm(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Any:
    del dataset
    return SVC(
        **params,
        kernel="rbf",
        class_weight="balanced",
        random_state=random_state,
        decision_function_shape="ovr",
    )


def _build_logistic_l2(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Any:
    del dataset
    return LogisticRegression(
        **params,
        l1_ratio=0.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
        max_iter=5000,
    )


def _build_logistic_l1(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Any:
    del dataset
    return LogisticRegression(
        **params,
        l1_ratio=1.0,
        solver="saga",
        class_weight="balanced",
        random_state=random_state,
        max_iter=5000,
        tol=1e-3,
    )


def _build_logistic_elasticnet(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Any:
    del dataset
    return LogisticRegression(
        **params,
        penalty="elasticnet",
        solver="saga",
        class_weight="balanced",
        random_state=random_state,
        max_iter=5000,
        tol=1e-3,
    )


def _build_random_forest(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Any:
    del dataset
    return RandomForestClassifier(
        **params,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=1,
    )


def _build_hist_gradient_boosting(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Any:
    del dataset
    return HistGradientBoostingClassifier(
        **params,
        random_state=random_state,
        early_stopping=True,
    )


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
        raise ValueError(f"Dataset {dataset.spec.slug} has a class with fewer than two samples.")

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

            _, fit_seconds, fit_peak_memory_mb = timed_peak_memory(
                lambda: _fit_with_warnings(pipeline, X_train, y_train)
            )
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
                **_predictive_metrics(pipeline, X_test, y_test, y_pred),
            }

            if model_spec.has_rule_metrics:
                X_test_prepared = pipeline.named_steps["imputer"].transform(X_test)
                model = pipeline.named_steps["model"]
                metrics.update(_fuzzy_rule_metrics(model, X_test_prepared, y_test))
                rule_sets[(dataset.spec.slug, model_spec.key)].append(_support_rule_set(model))

            if model_spec.key in ("logistic_l1", "logistic_elasticnet", "linear_svm"):
                model = pipeline.named_steps["model"]
                metrics.update(_sparse_classifier_complexity(model))

            rows.append(metrics)
    return rows


def _fit_with_warnings(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    with _suppressed_warnings():
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
        for fold, (inner_train, inner_valid) in enumerate(cv.split(X_train, y_train), start=1):
            pipeline = _build_pipeline(
                model_spec,
                dataset,
                random_state + fold,
                params,
            )
            with _suppressed_warnings():
                pipeline.fit(X_train[inner_train], y_train[inner_train])
            y_pred = pipeline.predict(X_train[inner_valid])
            y_valid = y_train[inner_valid]
            balanced_scores.append(float(balanced_accuracy_score(y_valid, y_pred)))
            f1_scores.append(float(f1_score(y_valid, y_pred, average="macro", zero_division=0)))
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
    if model_spec.uses_standard_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model_spec.builder(dataset, random_state, params)))
    return Pipeline(steps)


def _sparse_classifier_complexity(model: Any) -> dict[str, float]:
    """Count nonzero coefficients in a sparse linear classifier (Logistic L1, LinearSVC).

    Handles multiclass models (LogisticRegression with multiple coefficient rows)
    by counting features active in *any* class.  Returns a dict with key
    ``linear_nonzero_coefs``.
    """
    clf = model
    if hasattr(clf, "coef_"):
        coef = clf.coef_
        if coef.ndim == 2:
            n = int(np.any(coef != 0, axis=0).sum())
        else:
            n = int((coef != 0).sum())
        return {"linear_nonzero_coefs": float(n)}
    return {}


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
            summary["support_rule_jaccard"] = _mean_pairwise_jaccard(rule_sets[(dataset, model_key)])
        selected = Counter(row["selected_params"] for row in fold_rows)
        summary["selected_params_mode"] = selected.most_common(1)[0][0]
        summary["selected_params_unique"] = int(len(selected))
        for key in numeric_keys:
            values = np.asarray([row[key] for row in fold_rows if key in row], dtype=float)
            if values.size == 0:
                continue
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            summary[f"{key}_median"] = float(np.median(values))
            summary[f"{key}_q25"] = float(np.percentile(values, 25))
            summary[f"{key}_q75"] = float(np.percentile(values, 75))
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
    datasets = sorted(set.intersection(*(set(rows) for rows in by_model.values())))
    metrics = ["balanced_accuracy_mean", "f1_macro_mean", "accuracy_mean"]
    tests: list[dict[str, Any]] = []

    fuzzy_rows = by_model["fuzzy_rule_svm"]
    # Collect pairwise test rows first so we can apply Holm correction per metric.
    pairwise_rows: list[dict[str, Any]] = []
    for model in models:
        if model.key == "fuzzy_rule_svm":
            continue
        baseline_rows = by_model[model.key]
        for metric in metrics:
            fuzzy_values = np.asarray([fuzzy_rows[dataset][metric] for dataset in datasets], dtype=float)
            baseline_values = np.asarray([baseline_rows[dataset][metric] for dataset in datasets], dtype=float)
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
    # The family consists of all pairwise Wilcoxon tests for a given metric;
    # corrections are applied separately per metric because balanced accuracy,
    # macro F1, and accuracy are distinct evaluation criteria.
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
            np.asarray([by_model[model.key][dataset][metric] for dataset in datasets], dtype=float)
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


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm–Bonferroni step-down correction to a list of p-values.

    Returns corrected p-values in the same order as the input.  Each corrected
    p-value is min(1.0, (n - rank + 1) * raw_p) where rank is the position in
    ascending raw-p order (1-based) and n is the number of tests.

    Reference: Holm (1979), "A Simple Sequentially Rejective Multiple Test
    Procedure", Scandinavian Journal of Statistics 6(2): 65–70.
    """
    n = len(p_values)
    if n == 0:
        return []
    # Sort indices by ascending p-value
    order = sorted(range(n), key=lambda i: p_values[i])
    corrected = [0.0] * n
    for rank, idx in enumerate(order, start=1):
        corrected[idx] = min(1.0, p_values[idx] * (n - rank + 1))
    # Enforce monotonicity: corrected p-values must be non-decreasing when
    # sorted by raw p-value (to preserve FWER control).
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
        "# Recommended Model Comparison",
        "",
        "## Scope",
        "",
        "This report implements the feasible recommendations from `docs/findings.md`:",
        "",
        "- Tuned `FuzzyRuleSVM`.",
        "- Tuned standardized linear SVM.",
        "- Tuned standardized RBF SVM.",
        "- Tuned standardized logistic regression with L2 penalty.",
        "- Tuned standardized logistic regression with L1 sparse penalty.",
        "- Nested model selection inside each training fold.",
        "- Paired statistical tests over datasets.",
        "- Predictive and `FuzzyRuleSVM` interpretability metrics.",
        "",
        "FURIA, FARC-HD, and IVTURS were not run because this Python repository does not contain local implementations or configured wrappers for Weka, KEEL, or R fuzzy-rule packages. They remain required external baselines for a paper-quality fuzzy-rule comparison.",
        "",
        "## Protocol",
        "",
        f"- Datasets: {len(dataset_slugs)} prepared datasets from `datasets/prepared`.",
        f"- Outer evaluation: stratified {outer_splits}-fold CV, reduced only if a class has fewer samples than folds.",
        f"- Inner selection: stratified {inner_splits}-fold CV on each outer training fold.",
        "- Selection metric: mean inner balanced accuracy, with macro F1 as a tie-breaker.",
        "- Missing values: median imputation fitted inside each fold.",
        "- Regular baselines: median imputation plus standard scaling inside each fold.",
        "- Fuzzy baseline: median imputation only; fuzzy partitions are fitted on each training fold.",
        f"- Random state: `{random_state}`.",
        f"- Sample cap: `{max_samples}`." if max_samples is not None else "- Sample cap: none.",
        f"- Artifacts: `{output_dir}`.",
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
                _fmt(np.mean([row["accuracy_mean"] for row in rows])),
                _fmt(np.mean([row["balanced_accuracy_mean"] for row in rows])),
                _fmt(np.mean([row["f1_macro_mean"] for row in rows])),
            ]
        )
    lines.extend(_markdown_table(
        ["Model", "Mean Accuracy", "Mean Balanced Accuracy", "Mean Macro F1"],
        overall_rows,
    ))

    lines.extend([
        "",
        "## Paired Tests",
        "",
        "Wilcoxon tests compare per-dataset mean scores. Positive deltas favor `FuzzyRuleSVM`.",
        "",
    ])
    paired_rows = [
        row for row in tests
        if row["comparison"].startswith("FuzzyRuleSVM") and row["metric"] in {"balanced_accuracy", "f1_macro"}
    ]
    lines.extend(_markdown_table(
        ["Comparison", "Metric", "Mean Delta", "Median Delta", "Wins/Ties/Losses", "p-value"],
        [
            [
                row["comparison"],
                row["metric"],
                _fmt_signed(row["mean_delta"]),
                _fmt_signed(row["median_delta"]),
                f"{row['wins']}/{row['ties']}/{row['losses']}",
                _fmt_p(row["wilcoxon_pvalue"]),
            ]
            for row in paired_rows
        ],
    ))

    friedman_rows = [row for row in tests if row["comparison"] == "Friedman omnibus"]
    lines.extend([
        "",
        "Omnibus tests across all five compared models:",
        "",
    ])
    lines.extend(_markdown_table(
        ["Metric", "Statistic", "p-value"],
        [
            [row["metric"], _fmt(row["friedman_statistic"]), _fmt_p(row["friedman_pvalue"])]
            for row in friedman_rows
        ],
    ))

    lines.extend([
        "",
        "## Dataset Results",
        "",
        "Balanced accuracy by dataset. `Delta` is `FuzzyRuleSVM` minus the best non-fuzzy baseline.",
        "",
    ])
    dataset_rows = []
    for dataset in sorted(by_dataset):
        rows = by_dataset[dataset]
        fuzzy = rows["fuzzy_rule_svm"]["balanced_accuracy_mean"]
        baseline_scores = {
            key: row["balanced_accuracy_mean"]
            for key, row in rows.items()
            if key != "fuzzy_rule_svm"
        }
        best_key, best_score = max(baseline_scores.items(), key=lambda item: item[1])
        dataset_rows.append(
            [
                rows["fuzzy_rule_svm"]["dataset_name"],
                _fmt(fuzzy),
                _fmt(rows["linear_svm"]["balanced_accuracy_mean"]),
                _fmt(rows["rbf_svm"]["balanced_accuracy_mean"]),
                _fmt(rows["logistic_l2"]["balanced_accuracy_mean"]),
                _fmt(rows["logistic_l1"]["balanced_accuracy_mean"]),
                models_by_key(models)[best_key].name,
                _fmt_signed(fuzzy - best_score),
            ]
        )
    lines.extend(_markdown_table(
        [
            "Dataset",
            "Fuzzy",
            "Linear SVM",
            "RBF SVM",
            "Logistic L2",
            "Logistic L1",
            "Best Baseline",
            "Delta",
        ],
        dataset_rows,
    ))

    lines.extend([
        "",
        "## FuzzyRuleSVM Interpretability",
        "",
    ])
    fuzzy_rows = by_model["fuzzy_rule_svm"]
    interpretability_keys = [
        ("rule_support_rule_count_mean", "Support Rules"),
        ("rule_mean_rules_for_90pct_contribution_mean", "Rules for 90% Contribution"),
        ("rule_top5_abs_contribution_share_mean", "Top-5 Contribution Share"),
        ("support_rule_jaccard", "Support Rule Jaccard"),
        ("rule_explanation_fidelity_max_abs_error_mean", "Max Fidelity Error"),
    ]
    lines.extend(_markdown_table(
        ["Metric", "Mean Across Datasets"],
        [
            [label, _fmt(np.mean([row[key] for row in fuzzy_rows if key in row]))]
            for key, label in interpretability_keys
        ],
    ))

    lines.extend([
        "",
        "Datasets with the largest positive and negative balanced-accuracy deltas against the best non-fuzzy baseline:",
        "",
    ])
    deltas = []
    for dataset, rows in by_dataset.items():
        fuzzy = rows["fuzzy_rule_svm"]["balanced_accuracy_mean"]
        best_baseline = max(
            row["balanced_accuracy_mean"]
            for key, row in rows.items()
            if key != "fuzzy_rule_svm"
        )
        deltas.append((fuzzy - best_baseline, rows["fuzzy_rule_svm"]["dataset_name"]))
    strongest = sorted(deltas, reverse=True)[:5]
    weakest = sorted(deltas)[:5]
    lines.extend(_markdown_table(
        ["Largest Positive Delta", "Delta", "Largest Negative Delta", "Delta"],
        [
            [pos_name, _fmt_signed(pos_delta), neg_name, _fmt_signed(neg_delta)]
            for (pos_delta, pos_name), (neg_delta, neg_name) in zip(strongest, weakest, strict=True)
        ],
    ))

    lines.extend([
        "",
        "## Conclusion",
        "",
        _conclusion_text(by_model, tests),
        "",
    ])
    return "\n".join(lines)


def _summary_by_dataset_and_model(
    summaries: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        by_dataset[row["dataset"]][row["model_key"]] = row
    return dict(by_dataset)


def models_by_key(models: list[ModelSpec]) -> dict[str, ModelSpec]:
    return {model.key: model for model in models}


def _conclusion_text(
    by_model: dict[str, list[dict[str, Any]]],
    tests: list[dict[str, Any]],
) -> str:
    fuzzy_balanced = np.mean([row["balanced_accuracy_mean"] for row in by_model["fuzzy_rule_svm"]])
    baseline_balanced = {
        key: np.mean([row["balanced_accuracy_mean"] for row in rows])
        for key, rows in by_model.items()
        if key != "fuzzy_rule_svm"
    }
    best_key, best_score = max(baseline_balanced.items(), key=lambda item: item[1])
    model_labels = {
        "linear_svm": "linear SVM",
        "rbf_svm": "RBF SVM",
        "logistic_l2": "L2 logistic regression",
        "logistic_l1": "L1 logistic regression",
    }
    fuzzy_vs_best = fuzzy_balanced - best_score
    fuzzy_vs_linear = next(
        row for row in tests
        if row["comparison"] == "FuzzyRuleSVM vs Linear SVM"
        and row["metric"] == "balanced_accuracy"
    )
    return (
        f"`FuzzyRuleSVM` remains competitive but not predictively dominant. "
        f"Its mean balanced accuracy is {_fmt(fuzzy_balanced)}, versus {_fmt(best_score)} "
        f"for the best average non-fuzzy baseline ({model_labels[best_key]}), a delta of "
        f"{_fmt_signed(fuzzy_vs_best)}. Against linear SVM, the balanced-accuracy delta is "
        f"{_fmt_signed(fuzzy_vs_linear['mean_delta'])} over datasets "
        f"with Wilcoxon p={_fmt_p(fuzzy_vs_linear['wilcoxon_pvalue'])}. "
        "The strongest defensible claim is therefore intrinsic exact rule-based explanation "
        "with competitive accuracy, not consistent predictive superiority."
    )


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
        warnings.simplefilter("ignore", ConvergenceWarning)
        yield


if __name__ == "__main__":
    main()
