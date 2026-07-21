"""Compare FuzzyRuleSVM against FURIA, FARC-HD, and IVTURS fuzzy rule baselines.

Uses the same nested-CV protocol as compare_recommendations.py.

For speed:
  - Use --fast to cap evaluations/pop for FARC-HD/IVTURS (smoke test).
  - Use --max-samples N to limit per-dataset sample count.
  - Use dataset slugs as positional arguments to run a subset.

Typical usage:
  # Smoke test on 3 datasets
  uv run python scripts/compare_fuzzy_baselines.py iris breast_cancer_diagnostic pima_diabetes --fast

  # Full run (slow, ~4-8 hours)
  uv run python scripts/compare_fuzzy_baselines.py
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

from fysvm.baselines import FARCHDClassifier, FURIAClassifier, IVTURSClassifier
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
from fysvm.rule_svm import FuzzyRuleSVM
from fysvm.run_metadata import write_run_metadata


ModelBuilder = Callable[[PreparedDataset, int, dict[str, Any]], Any]
ParamGridBuilder = Callable[[PreparedDataset], list[dict[str, Any]]]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    builder: ModelBuilder
    param_grid: ParamGridBuilder
    has_rule_metrics: bool = False


@dataclass(frozen=True)
class ComparisonResult:
    output_dir: Path
    report_path: Path
    fold_metrics: list[dict[str, Any]]
    summary_metrics: list[dict[str, Any]]
    statistical_tests: list[dict[str, Any]]


def run_fuzzy_comparison(
    *,
    dataset_slugs: Iterable[str] | None = None,
    data_dir: str | Path = "datasets/prepared",
    output_dir: str | Path = "runs/fuzzy-baselines-comparison",
    report_path: str | Path = "docs/fuzzy_baselines_comparison.md",
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 0,
    max_samples: int | None = None,
    fast: bool = False,
) -> ComparisonResult:
    """Run nested-CV comparing FuzzyRuleSVM vs FURIA, FARC-HD, IVTURS."""

    selected = list(dataset_slugs) if dataset_slugs is not None else [
        spec.slug for spec in DATASET_SPECS
    ]
    models = _model_specs(fast=fast)
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
            "fast": fast,
        },
    )

    fold_metrics: list[dict[str, Any]] = []
    excluded_runs: list[dict[str, Any]] = []
    rule_sets: dict[tuple[str, str], list[set[str]]] = defaultdict(list)

    for dataset_index, slug in enumerate(selected):
        print(f"\n[{dataset_index + 1}/{len(selected)}] {slug}", flush=True)
        dataset = load_dataset(slug, data_dir)
        dataset_seed = _stable_dataset_seed(random_state, slug)
        if max_samples is not None and dataset.X.shape[0] > max_samples:
            dataset = _stratified_subset(dataset, max_samples, dataset_seed)
        try:
            rows = _evaluate_dataset(
                dataset,
                models,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                random_state=dataset_seed,
                rule_sets=rule_sets,
            )
            fold_metrics.extend(rows)
        except Exception as exc:  # noqa: BLE001 - preserve failure artifact and continue.
            excluded_runs.append(
                {
                    "dataset": slug,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                    "stage": "dataset_evaluation",
                }
            )
            _write_csv(output_path / "excluded_runs.csv", excluded_runs)
            print(f"  excluded {slug}: {type(exc).__name__}: {exc}", flush=True)
        # Checkpoint after each dataset
        _write_csv(output_path / "fold_metrics.csv", fold_metrics)

    summary_metrics = _summarize_metrics(fold_metrics, rule_sets)
    statistical_tests = _statistical_tests(summary_metrics, models)

    _write_csv(output_path / "fold_metrics.csv", fold_metrics)
    _write_csv(output_path / "metrics.csv", summary_metrics)
    _write_csv(output_path / "statistical_tests.csv", statistical_tests)
    _write_csv(output_path / "excluded_runs.csv", excluded_runs)
    (output_path / "metrics.json").write_text(
        json.dumps(
            {
                "fold_metrics": fold_metrics,
                "summary_metrics": summary_metrics,
                "statistical_tests": statistical_tests,
                "excluded_runs": excluded_runs,
            },
            indent=2,
            default=_json_default,
        ) + "\n",
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
        fast=fast,
    )
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(report, encoding="utf-8")

    return ComparisonResult(output_path, rp, fold_metrics, summary_metrics, statistical_tests)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*")
    parser.add_argument("--data-dir", default="datasets/prepared")
    parser.add_argument("--output-dir", default="runs/fuzzy-baselines-comparison")
    parser.add_argument("--report", default="docs/fuzzy_baselines_comparison.md")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced FARC-HD/IVTURS settings for a quick smoke test.",
    )
    args = parser.parse_args(argv)

    result = run_fuzzy_comparison(
        dataset_slugs=args.datasets or None,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        random_state=args.random_state,
        max_samples=args.max_samples,
        fast=args.fast,
    )
    print(result.report_path)
    print(result.output_dir)


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------

def _model_specs(fast: bool = False) -> list[ModelSpec]:
    return [
        ModelSpec(
            "fuzzy_rule_svm",
            "FuzzyRuleSVM",
            _build_fuzzy_rule_svm,
            _fuzzy_rule_svm_grid,
            has_rule_metrics=True,
        ),
        ModelSpec(
            "furia",
            "FURIA",
            _build_furia,
            _furia_grid,
        ),
        ModelSpec(
            "farchd",
            "FARC-HD",
            lambda ds, rs, p: _build_farchd(ds, rs, p, fast=fast),
            lambda ds: _farchd_grid(ds, fast=fast),
        ),
        ModelSpec(
            "ivturs",
            "IVTURS",
            lambda ds, rs, p: _build_ivturs(ds, rs, p, fast=fast),
            lambda ds: _farchd_grid(ds, fast=fast),
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


def _furia_grid(dataset: PreparedDataset) -> list[dict[str, Any]]:
    del dataset
    return [
        {"min_no": min_no}
        for min_no in (1.0, 2.0, 3.0)
    ]


def _farchd_grid(dataset: PreparedDataset, fast: bool = False) -> list[dict[str, Any]]:
    n_features = dataset.X.shape[1]
    # parkinsons_disease_classification (754 features) is prohibitively slow
    # even at minimal evaluations; mark for minimal effort only.
    if n_features > 500:
        return [{"max_depth": 1, "min_support": 0.10, "evaluations": 5, "pop": 5}]
    evaluations = 10 if fast else 20
    pop = 10 if fast else 15
    # Limit depth for high-dimensional datasets to avoid combinatorial explosion
    max_depth = 1 if n_features > 32 else 2
    # Single configuration: no inner CV for FARC-HD/IVTURS to keep runtime feasible.
    # The original KEEL settings used 20000 evaluations; we note this reduction.
    return [{"max_depth": max_depth, "min_support": 0.05, "evaluations": evaluations, "pop": pop}]


def _build_fuzzy_rule_svm(
    dataset: PreparedDataset, random_state: int, params: dict[str, Any]
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


def _build_furia(
    dataset: PreparedDataset, random_state: int, params: dict[str, Any]
) -> Any:
    del dataset
    return FURIAClassifier(**params, random_state=random_state)


def _build_farchd(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
    fast: bool = False,
) -> Any:
    del dataset
    return FARCHDClassifier(**params, random_state=random_state)


def _build_ivturs(
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
    fast: bool = False,
) -> Any:
    del dataset
    return IVTURSClassifier(**params, random_state=random_state)


# ---------------------------------------------------------------------------
# FURIA complexity helper
# ---------------------------------------------------------------------------

def _furia_complexity_metrics(model: Any) -> dict[str, float]:
    """Extract rule count and antecedent statistics from a fitted FURIA model.

    ``model`` should be a :class:`FURIAClassifier` (binary or multiclass via
    the wrapper's internal single-model fit).  Returns a dict with keys
    ``furia_n_rules``, ``furia_total_antecedents``, and
    ``furia_avg_antecedents``.
    """
    clf = model if hasattr(model, "ruleset_") else getattr(model, "_clf", None)
    if clf is None or not hasattr(clf, "ruleset_"):
        return {}
    ruleset = clf.ruleset_
    n_rules = len(ruleset)
    if n_rules == 0:
        return {"furia_n_rules": 0.0, "furia_total_antecedents": 0.0, "furia_avg_antecedents": 0.0}
    total_ant = sum(len(r.antecedents) for r in ruleset)
    return {
        "furia_n_rules": float(n_rules),
        "furia_total_antecedents": float(total_ant),
        "furia_avg_antecedents": float(total_ant / n_rules),
    }


# ---------------------------------------------------------------------------
# Evaluation loop (same pattern as compare_recommendations.py)
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
        raise ValueError(f"Dataset {dataset.spec.slug} has a class with fewer than two samples.")

    rows: list[dict[str, Any]] = []
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)

    for fold, (train_index, test_index) in enumerate(cv.split(dataset.X, y), start=1):
        print(f"  fold {fold}/{splits}", flush=True)
        X_train, X_test = dataset.X[train_index], dataset.X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for model_index, model_spec in enumerate(models):
            model_seed = random_state + fold * 100 + model_index
            print(f"    {model_spec.name}", end=" ", flush=True)
            t_start = time.perf_counter()

            selected_params, inner_score, inner_f1 = _select_params(
                model_spec, dataset, X_train, y_train,
                inner_splits=inner_splits, random_state=model_seed,
            )
            pipeline = _build_pipeline(model_spec, dataset, model_seed, selected_params)

            fit_start = time.perf_counter()
            with _suppressed_warnings():
                pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            y_pred = pipeline.predict(X_test)
            predict_seconds = time.perf_counter() - predict_start

            elapsed = time.perf_counter() - t_start
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"→ balanced_acc={bal_acc:.3f} ({elapsed:.1f}s)", flush=True)

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
                **_predictive_metrics(pipeline, X_test, y_test, y_pred),
            }

            if model_spec.has_rule_metrics:
                X_test_prepared = pipeline.named_steps["imputer"].transform(X_test)
                model = pipeline.named_steps["model"]
                metrics.update(_fuzzy_rule_metrics(model, X_test_prepared, y_test))
                rule_sets[(dataset.spec.slug, model_spec.key)].append(_support_rule_set(model))

            if model_spec.key == "furia":
                furia_model = pipeline.named_steps["model"]
                metrics.update(_furia_complexity_metrics(furia_model))

            rows.append(metrics)
    return rows


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
    # Strip internal markers
    effective_grid = [{k: v for k, v in p.items() if not k.startswith("_")} for p in grid]
    if len(effective_grid) == 1:
        return effective_grid[0], float("nan"), float("nan")

    class_counts = np.unique(y_train, return_counts=True)[1]
    splits = min(inner_splits, int(np.min(class_counts)))
    if splits < 2:
        return effective_grid[0], float("nan"), float("nan")

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    scored: list[tuple[float, float, int, dict[str, Any]]] = []
    for param_index, params in enumerate(effective_grid):
        balanced_scores: list[float] = []
        f1_scores: list[float] = []
        for fold, (inner_train, inner_valid) in enumerate(cv.split(X_train, y_train), start=1):
            pipeline = _build_pipeline(model_spec, dataset, random_state + fold, params)
            with _suppressed_warnings():
                pipeline.fit(X_train[inner_train], y_train[inner_train])
            y_pred = pipeline.predict(X_train[inner_valid])
            y_valid = y_train[inner_valid]
            balanced_scores.append(float(balanced_accuracy_score(y_valid, y_pred)))
            f1_scores.append(float(f1_score(y_valid, y_pred, average="macro", zero_division=0)))
        scored.append((
            float(np.mean(balanced_scores)),
            float(np.mean(f1_scores)),
            -param_index,
            params,
        ))

    best_balanced, best_f1, _, best_params = max(scored, key=lambda item: item[:3])
    return best_params, best_balanced, best_f1


def _build_pipeline(
    model_spec: ModelSpec,
    dataset: PreparedDataset,
    random_state: int,
    params: dict[str, Any],
) -> Pipeline:
    # All fuzzy models use median imputation; no StandardScaler (fuzzy partitions are adaptive)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model_spec.builder(dataset, random_state, params)),
    ])


# ---------------------------------------------------------------------------
# Summarization and statistical tests
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
            key for key, value in first.items()
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
        model.key: {row["dataset"]: row for row in summaries if row["model_key"] == model.key}
        for model in models
    }
    datasets = sorted(set.intersection(*(set(rows) for rows in by_model.values())))
    metrics = ["balanced_accuracy_mean", "f1_macro_mean", "accuracy_mean"]
    tests: list[dict[str, Any]] = []

    fuzzy_rows = by_model["fuzzy_rule_svm"]
    pairwise_rows: list[dict[str, Any]] = []
    for model in models:
        if model.key == "fuzzy_rule_svm":
            continue
        baseline_rows = by_model[model.key]
        for metric in metrics:
            fuzzy_values = np.asarray([fuzzy_rows[d][metric] for d in datasets], dtype=float)
            baseline_values = np.asarray([baseline_rows[d][metric] for d in datasets], dtype=float)
            diffs = fuzzy_values - baseline_values
            pairwise_rows.append({
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
                "wilcoxon_pvalue": _wilcoxon_pvalue(diffs),
            })

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
            np.asarray([by_model[m.key][d][metric] for d in datasets], dtype=float)
            for m in models
        ]
        tests.append({
            "comparison": "Friedman omnibus",
            "metric": metric.removesuffix("_mean"),
            "n_datasets": int(len(datasets)),
            "models": ", ".join(m.name for m in models),
            "friedman_statistic": _friedman_statistic(values),
            "friedman_pvalue": _friedman_pvalue(values),
        })
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
    fast: bool,
) -> str:
    by_model: dict[str, list[dict[str, Any]]] = {
        model.key: [row for row in summaries if row["model_key"] == model.key]
        for model in models
    }
    by_dataset = _summary_by_dataset_and_model(summaries)

    lines = [
        "# Fuzzy Rule Baselines Comparison",
        "",
        "## Scope",
        "",
        "Direct comparison of FuzzyRuleSVM against three established fuzzy rule-based classifiers:",
        "",
        "- **FURIA** (Hühn & Hüllermeier 2009): Fuzzy Unordered Rule Induction Algorithm.",
        "  Python implementation: `asieriko/ex_fuzzy_farchd` `furia_classifier.py`.",
        "- **FARC-HD** (Alcalá-Fdez et al. 2011): Fuzzy Association Rule-Based Classification",
        "  for High-Dimensional problems with genetic rule selection and lateral tuning.",
        "  Python implementation: `asieriko/ex_fuzzy_farchd` FARCHD with ex-fuzzy T1 partitions.",
        "- **IVTURS** (Sanz et al. 2013): FARC-HD extended with interval-valued type-2 fuzzy sets.",
        "  Python approximation: FARCHD framework with ex-fuzzy IT2 partitions.",
        "  Reference Java implementation: JoseanSanz/IVTURSfast.",
        "",
        "## Protocol",
        "",
        f"- Datasets: {len(dataset_slugs)} datasets from `datasets/prepared`.",
        f"- Outer evaluation: stratified {outer_splits}-fold CV.",
        f"- Inner selection: stratified {inner_splits}-fold CV on each outer training fold.",
        "- Selection metric: mean inner balanced accuracy, macro F1 as tie-breaker.",
        "- Missing values: median imputation inside each fold.",
        "- Fuzzy partitions: fitted from training data inside each fold.",
        f"- Fast mode: {'yes (reduced FARC-HD/IVTURS evaluations)' if fast else 'no (full settings)'}.",
        f"- Sample cap: {max_samples if max_samples is not None else 'none'}.",
        f"- Random state: {random_state}.",
        f"- Artifacts: `{output_dir}`.",
        "",
        "## Overall Performance",
        "",
    ]

    overall_rows = []
    for model in models:
        rows = by_model[model.key]
        if not rows:
            continue
        overall_rows.append([
            model.name,
            _fmt(np.mean([row["accuracy_mean"] for row in rows])),
            _fmt(np.mean([row["balanced_accuracy_mean"] for row in rows])),
            _fmt(np.mean([row["f1_macro_mean"] for row in rows])),
            _fmt(np.mean([row.get("fit_seconds_mean", float("nan")) for row in rows])),
        ])
    lines.extend(_markdown_table(
        ["Model", "Mean Accuracy", "Mean Balanced Accuracy", "Mean Macro F1", "Mean Fit Time (s)"],
        overall_rows,
    ))

    lines.extend([
        "",
        "## Paired Tests (FuzzyRuleSVM vs baselines)",
        "",
        "Wilcoxon signed-rank tests over per-dataset mean scores. "
        "Positive delta favors FuzzyRuleSVM.",
        "",
    ])
    paired_rows = [
        row for row in tests
        if row["comparison"].startswith("FuzzyRuleSVM") and row["metric"] in {"balanced_accuracy", "f1_macro"}
    ]
    lines.extend(_markdown_table(
        ["Comparison", "Metric", "Fuzzy Mean", "Baseline Mean", "Mean Delta", "W/T/L", "p-value"],
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
    ))

    friedman_rows = [row for row in tests if row["comparison"] == "Friedman omnibus"]
    lines.extend([
        "",
        "Omnibus Friedman test across all four models:",
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
        "## Dataset Results (Balanced Accuracy)",
        "",
    ])
    dataset_rows = []
    for dataset in sorted(by_dataset):
        rows = by_dataset[dataset]
        fuzzy = rows.get("fuzzy_rule_svm", {}).get("balanced_accuracy_mean", float("nan"))
        furia = rows.get("furia", {}).get("balanced_accuracy_mean", float("nan"))
        farchd = rows.get("farchd", {}).get("balanced_accuracy_mean", float("nan"))
        ivturs = rows.get("ivturs", {}).get("balanced_accuracy_mean", float("nan"))
        best_baseline = max(v for v in [furia, farchd, ivturs] if not np.isnan(v)) if any(
            not np.isnan(v) for v in [furia, farchd, ivturs]
        ) else float("nan")
        dataset_rows.append([
            rows.get("fuzzy_rule_svm", rows.get("furia", {})).get("dataset_name", dataset),
            _fmt(fuzzy),
            _fmt(furia),
            _fmt(farchd),
            _fmt(ivturs),
            _fmt_signed(fuzzy - best_baseline) if not np.isnan(fuzzy + best_baseline) else "nan",
        ])
    lines.extend(_markdown_table(
        ["Dataset", "FuzzyRuleSVM", "FURIA", "FARC-HD", "IVTURS", "Delta vs Best Fuzzy Baseline"],
        dataset_rows,
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


def _conclusion_text(
    by_model: dict[str, list[dict[str, Any]]],
    tests: list[dict[str, Any]],
) -> str:
    fuzzy_ba = np.mean([r["balanced_accuracy_mean"] for r in by_model.get("fuzzy_rule_svm", [])])
    baselines_ba = {
        key: np.mean([r["balanced_accuracy_mean"] for r in rows])
        for key, rows in by_model.items()
        if key != "fuzzy_rule_svm" and rows
    }
    if not baselines_ba:
        return "No baseline data available yet."
    best_key, best_ba = max(baselines_ba.items(), key=lambda x: x[1])
    key_names = {"furia": "FURIA", "farchd": "FARC-HD", "ivturs": "IVTURS"}
    delta = fuzzy_ba - best_ba
    furia_test = next(
        (r for r in tests if r["comparison"] == "FuzzyRuleSVM vs FURIA" and r["metric"] == "balanced_accuracy"),
        None,
    )
    furia_str = (
        f"Against FURIA, delta={_fmt_signed(furia_test['mean_delta'])}, p={_fmt_p(furia_test['wilcoxon_pvalue'])}."
        if furia_test else ""
    )
    return (
        f"FuzzyRuleSVM mean balanced accuracy: {_fmt(fuzzy_ba)}. "
        f"Best fuzzy baseline: {key_names.get(best_key, best_key)} ({_fmt(best_ba)}), "
        f"delta {_fmt_signed(delta)}. "
        f"{furia_str} "
        "FuzzyRuleSVM's primary differentiator remains intrinsic, exact, additive rule-contribution "
        "explanations, not predictive dominance over established fuzzy rule-based classifiers."
    )


# ---------------------------------------------------------------------------
# Utilities
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
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore")
        yield


if __name__ == "__main__":
    main()
