"""Evaluation harness for fuzzy rule-space classifiers."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from fysvm.datasets import DATASET_SPECS, PreparedDataset, load_dataset
from fysvm.rule_svm import FuzzyRuleSVM


_DATASET_SEED_INDEX = {spec.slug: i for i, spec in enumerate(DATASET_SPECS)}


def _stable_dataset_seed(random_state: int, slug: str, *, stride: int = 1000) -> int:
    """Return an order-independent seed for a prepared dataset slug."""

    return random_state + _DATASET_SEED_INDEX.get(slug, 0) * stride


EstimatorFactory = Callable[[PreparedDataset, int], Any]


@dataclass(frozen=True)
class EvaluationResult:
    """Paths and in-memory metrics produced by an evaluation run."""

    output_dir: Path
    fold_metrics: list[dict[str, Any]]
    summary_metrics: list[dict[str, Any]]


def evaluate_classifier(
    estimator_factory: EstimatorFactory | None = None,
    *,
    dataset_slugs: Iterable[str] | None = None,
    data_dir: str | Path = "datasets/prepared",
    output_dir: str | Path = "runs/evaluation",
    n_splits: int = 5,
    random_state: int = 0,
    max_samples: int | None = None,
) -> EvaluationResult:
    """Evaluate a classifier across prepared datasets.

    The harness reports standard predictive metrics and model-native fuzzy rule
    metrics. For multiclass datasets, the binary fuzzy classifier is evaluated
    through one-vs-rest decomposition and the rule metrics are averaged across
    the fitted binary estimators.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if max_samples is not None and max_samples < 2:
        raise ValueError("max_samples must be None or at least 2.")

    factory = estimator_factory or default_fuzzy_rule_svm_factory
    selected = list(dataset_slugs) if dataset_slugs is not None else [
        spec.slug for spec in DATASET_SPECS
    ]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fold_metrics: list[dict[str, Any]] = []
    summary_metrics: list[dict[str, Any]] = []

    for slug in selected:
        dataset = load_dataset(slug, data_dir)
        if max_samples is not None and dataset.X.shape[0] > max_samples:
            dataset = _stratified_subset(dataset, max_samples, random_state)

        dataset_fold_metrics, rule_sets = _evaluate_dataset(
            dataset,
            factory,
            n_splits=n_splits,
            random_state=random_state,
        )
        fold_metrics.extend(dataset_fold_metrics)
        summary_metrics.append(_summarize_dataset(dataset, dataset_fold_metrics, rule_sets))

    _write_csv(output_path / "fold_metrics.csv", fold_metrics)
    _write_csv(output_path / "metrics.csv", summary_metrics)
    (output_path / "metrics.json").write_text(
        json.dumps(summary_metrics, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return EvaluationResult(output_path, fold_metrics, summary_metrics)


def default_fuzzy_rule_svm_factory(dataset: PreparedDataset, random_state: int) -> Any:
    """Build the default classifier used by the CLI harness."""

    n_features = dataset.X.shape[1]
    max_rule_length = 2 if n_features <= 32 else 1
    max_rules = min(256, max(24, 3 * n_features))
    base = FuzzyRuleSVM(
        C=1.0,
        penalty="l1",
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


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for the evaluation harness."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Prepared dataset slugs to evaluate. Defaults to all datasets.",
    )
    parser.add_argument("--data-dir", default="datasets/prepared")
    parser.add_argument("--output-dir", default="runs/evaluation")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional stratified cap for quick smoke runs.",
    )
    args = parser.parse_args(argv)

    result = evaluate_classifier(
        dataset_slugs=args.datasets or None,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_splits=args.splits,
        random_state=args.random_state,
        max_samples=args.max_samples,
    )
    print(result.output_dir)


def _evaluate_dataset(
    dataset: PreparedDataset,
    estimator_factory: EstimatorFactory,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], list[set[str]]]:
    y = np.asarray(dataset.y)
    class_counts = np.unique(y, return_counts=True)[1]
    splits = min(n_splits, int(np.min(class_counts)))
    if splits < 2:
        raise ValueError(f"Dataset {dataset.spec.slug} has a class with fewer than two samples.")

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    fold_metrics: list[dict[str, Any]] = []
    rule_sets: list[set[str]] = []

    for fold, (train_index, test_index) in enumerate(cv.split(dataset.X, y), start=1):
        X_train, X_test = dataset.X[train_index], dataset.X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        estimator = estimator_factory(dataset, random_state + fold)
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )

        fit_start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        predict_seconds = time.perf_counter() - predict_start

        metrics = {
            "dataset": dataset.spec.slug,
            "dataset_name": dataset.spec.name,
            "fold": fold,
            "n_train": int(len(train_index)),
            "n_test": int(len(test_index)),
            "n_features": int(dataset.X.shape[1]),
            "n_classes": int(len(np.unique(y))),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            **_predictive_metrics(pipeline, X_test, y_test, y_pred),
        }

        X_test_prepared = pipeline.named_steps["imputer"].transform(X_test)
        model = pipeline.named_steps["model"]
        metrics.update(_fuzzy_rule_metrics(model, X_test_prepared, y_test))
        fold_metrics.append(metrics)
        rule_sets.append(_support_rule_set(model))

    return fold_metrics, rule_sets


def _predictive_metrics(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    scores = _decision_scores(pipeline, X_test)
    if scores is not None:
        metrics.update(_score_metrics(y_test, scores, pipeline.classes_))
    return metrics


def _score_metrics(
    y_test: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    try:
        if scores.ndim == 1:
            positive_label = classes[1]
            y_binary = (y_test == positive_label).astype(int)
            metrics["roc_auc"] = float(roc_auc_score(y_binary, scores))
            metrics["mean_abs_margin"] = float(np.mean(np.abs(scores)))
            metrics["median_abs_margin"] = float(np.median(np.abs(scores)))
        else:
            indicator = np.column_stack([(y_test == label).astype(int) for label in classes])
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(indicator, scores, average="macro", multi_class="ovr")
            )
            sorted_scores = np.sort(scores, axis=1)
            metrics["mean_margin_gap"] = float(np.mean(sorted_scores[:, -1] - sorted_scores[:, -2]))
            metrics["median_margin_gap"] = float(
                np.median(sorted_scores[:, -1] - sorted_scores[:, -2])
            )
    except ValueError:
        return metrics
    return metrics


def _decision_scores(pipeline: Pipeline, X: np.ndarray) -> np.ndarray | None:
    try:
        scores = pipeline.decision_function(X)
    except AttributeError:
        return None
    return np.asarray(scores)


def _fuzzy_rule_metrics(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    estimators = _extract_binary_estimators(model)
    if not estimators:
        return {}

    labels = _extract_estimator_labels(model, y_test)
    per_estimator = []
    for index, estimator in enumerate(estimators):
        if not hasattr(estimator, "rules_"):
            continue
        y_for_estimator = labels[index] if labels is not None else y_test
        per_estimator.append(_single_fuzzy_estimator_metrics(estimator, X_test, y_for_estimator))

    if not per_estimator:
        return {}
    return _mean_metric_dicts(per_estimator, prefix="rule_")


def _single_fuzzy_estimator_metrics(
    estimator: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    Z = estimator.transform(X_test)
    coef = estimator.coef_.reshape(-1)
    contributions = Z * coef
    margins = estimator.decision_function(X_test)
    abs_contributions = np.abs(contributions)
    abs_mass = np.sum(abs_contributions, axis=1)
    active_mask = np.abs(coef) > 1e-10
    lengths = np.asarray([rule.length for rule in estimator.rules_], dtype=float)

    support_rule_count = int(np.sum(active_mask))
    contributing_counts = np.sum(abs_contributions > 1e-8, axis=1)
    firing_counts = np.sum(Z > 1e-6, axis=1)
    fidelity_errors = np.abs(margins - (estimator.intercept_ + np.sum(contributions, axis=1)))
    predicted_sign = np.where(margins >= 0.0, 1.0, -1.0)
    aligned_contributions = contributions * predicted_sign[:, np.newaxis]
    supporting = np.sum(np.maximum(aligned_contributions, 0.0), axis=1)
    opposing = np.sum(np.maximum(-aligned_contributions, 0.0), axis=1)
    total_contrast = supporting + opposing

    sorted_abs = np.sort(abs_contributions, axis=1)[:, ::-1]
    top1_share = _topk_share(sorted_abs, abs_mass, 1)
    top5_share = _topk_share(sorted_abs, abs_mass, 5)
    explanation_90 = _rules_to_cover_mass(sorted_abs, abs_mass, coverage=0.90)
    violations = estimator.fuzzy_violations(X_test, y_test)

    return {
        "candidate_rules": float(getattr(estimator, "n_candidate_rules_", len(estimator.rules_))),
        "n_rules": float(len(estimator.rules_)),
        "retained_rule_ratio": float(
            len(estimator.rules_) / max(1, getattr(estimator, "n_candidate_rules_", len(estimator.rules_)))
        ),
        "screened_features": float(getattr(estimator, "n_screened_features_", estimator.n_features_in_)),
        "screened_feature_ratio": float(
            getattr(estimator, "n_screened_features_", estimator.n_features_in_)
            / max(1, estimator.n_features_in_)
        ),
        "support_rule_count": float(support_rule_count),
        "support_rule_ratio": float(support_rule_count / max(1, len(estimator.rules_))),
        "avg_rule_length": float(np.mean(lengths)),
        "avg_support_rule_length": float(np.mean(lengths[active_mask])) if support_rule_count else 0.0,
        "weighted_rule_length": _safe_weighted_mean(lengths, np.abs(coef)),
        "coef_l1": float(np.sum(np.abs(coef))),
        "coef_l2": float(np.linalg.norm(coef)),
        "length_weighted_complexity": float(np.sum(lengths * np.abs(coef))),
        "mean_rule_activation": float(np.mean(Z)),
        "mean_fired_rules_per_sample": float(np.mean(firing_counts)),
        "mean_contributing_rules_per_sample": float(np.mean(contributing_counts)),
        "mean_rules_for_90pct_contribution": float(np.mean(explanation_90)),
        "top1_abs_contribution_share": float(np.mean(top1_share)),
        "top5_abs_contribution_share": float(np.mean(top5_share)),
        "explanation_fidelity_max_abs_error": float(np.max(fidelity_errors)),
        "explanation_fidelity_mean_abs_error": float(np.mean(fidelity_errors)),
        "mean_abs_margin": float(np.mean(np.abs(margins))),
        "mean_supporting_contribution": float(np.mean(supporting)),
        "mean_opposing_contribution": float(np.mean(opposing)),
        "mean_contrastive_gap": float(np.mean(supporting - opposing)),
        "mean_contrastive_ratio": float(
            np.mean(np.divide(supporting, total_contrast, out=np.zeros_like(supporting), where=total_contrast > 0))
        ),
        "mean_slack": float(np.mean([item["slack"] for item in violations])),
        "cleanly_classified_membership": float(
            np.mean([item["memberships"]["cleanly_classified"] for item in violations])
        ),
        "borderline_membership": float(
            np.mean([item["memberships"]["borderline"] for item in violations])
        ),
        "strong_violation_membership": float(
            np.mean([item["memberships"]["strong_violation"] for item in violations])
        ),
    }


def _support_rule_set(model: Any) -> set[str]:
    estimators = _extract_binary_estimators(model)
    labels = getattr(model, "classes_", None)
    rules: set[str] = set()
    for index, estimator in enumerate(estimators):
        if not hasattr(estimator, "support_rules"):
            continue
        label_prefix = f"ovr={labels[index]}|" if labels is not None and len(estimators) > 1 else ""
        for item in estimator.support_rules():
            rules.add(f"{label_prefix}{item['rule']}")
    return rules


def _extract_binary_estimators(model: Any) -> list[Any]:
    if isinstance(model, OneVsRestClassifier):
        return list(model.estimators_)
    return [model]


def _extract_estimator_labels(model: Any, y_test: np.ndarray) -> list[np.ndarray] | None:
    if isinstance(model, OneVsRestClassifier):
        return [(y_test == label).astype(int) for label in model.classes_]
    return None


def _topk_share(sorted_abs: np.ndarray, abs_mass: np.ndarray, k: int) -> np.ndarray:
    topk = np.sum(sorted_abs[:, : min(k, sorted_abs.shape[1])], axis=1)
    return np.divide(topk, abs_mass, out=np.zeros_like(abs_mass), where=abs_mass > 0)


def _rules_to_cover_mass(
    sorted_abs: np.ndarray,
    abs_mass: np.ndarray,
    *,
    coverage: float,
) -> np.ndarray:
    cumulative = np.cumsum(sorted_abs, axis=1)
    threshold = coverage * abs_mass
    counts = np.zeros(sorted_abs.shape[0], dtype=float)
    for index, total in enumerate(abs_mass):
        if total <= 0:
            counts[index] = 0.0
        else:
            counts[index] = float(np.searchsorted(cumulative[index], threshold[index], side="left") + 1)
    return counts


def _safe_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    return float(np.sum(values * weights) / total)


def _mean_metric_dicts(metrics: list[dict[str, float]], *, prefix: str = "") -> dict[str, float]:
    keys = sorted({key for item in metrics for key in item})
    return {
        f"{prefix}{key}": float(np.mean([item[key] for item in metrics if key in item]))
        for key in keys
    }


def _summarize_dataset(
    dataset: PreparedDataset,
    fold_metrics: list[dict[str, Any]],
    rule_sets: list[set[str]],
) -> dict[str, Any]:
    numeric_keys = sorted(
        key
        for key, value in fold_metrics[0].items()
        if isinstance(value, int | float | np.integer | np.floating)
        and key not in {"fold", "n_train", "n_test", "n_features", "n_classes"}
    )
    summary: dict[str, Any] = {
        "dataset": dataset.spec.slug,
        "dataset_name": dataset.spec.name,
        "task": dataset.spec.task,
        "n_samples": int(dataset.X.shape[0]),
        "n_features": int(dataset.X.shape[1]),
        "n_classes": int(len(np.unique(dataset.y))),
        "n_folds": int(len(fold_metrics)),
        "support_rule_jaccard": _mean_pairwise_jaccard(rule_sets),
    }
    for key in numeric_keys:
        values = np.asarray([metrics[key] for metrics in fold_metrics if key in metrics], dtype=float)
        if values.size == 0:
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_q25"] = float(np.percentile(values, 25))
        summary[f"{key}_q75"] = float(np.percentile(values, 75))
    return summary


def _mean_pairwise_jaccard(rule_sets: list[set[str]]) -> float:
    scores: list[float] = []
    for left, right in combinations(rule_sets, 2):
        union = left | right
        if not union:
            scores.append(1.0)
        else:
            scores.append(len(left & right) / len(union))
    return float(np.mean(scores)) if scores else 1.0


def _stratified_subset(
    dataset: PreparedDataset,
    max_samples: int,
    random_state: int,
) -> PreparedDataset:
    rng = np.random.default_rng(random_state)
    indices: list[int] = []
    y = np.asarray(dataset.y)
    labels, counts = np.unique(y, return_counts=True)
    proportions = counts / np.sum(counts)
    allocations = np.maximum(2, np.floor(proportions * max_samples).astype(int))
    while np.sum(allocations) > max_samples:
        largest = int(np.argmax(allocations))
        if allocations[largest] > 2:
            allocations[largest] -= 1
        else:
            break
    for label, allocation in zip(labels, allocations, strict=True):
        label_indices = np.flatnonzero(y == label)
        take = min(int(allocation), len(label_indices))
        indices.extend(rng.choice(label_indices, size=take, replace=False).tolist())
    selected = np.asarray(sorted(indices), dtype=int)
    return PreparedDataset(
        dataset.spec,
        dataset.X[selected],
        dataset.y[selected],
        dataset.feature_names,
        dataset.target_names,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
