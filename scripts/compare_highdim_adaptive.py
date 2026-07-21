"""Evaluate adaptive high-dimensional FuzzyRuleSVM against fallback models."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from fysvm.datasets import DATASET_SPECS, PreparedDataset, load_dataset
from fysvm.evaluation import _json_default, _stable_dataset_seed, _stratified_subset
from fysvm.membership import MembershipSVM, membership_nonzero_coef_count
from fysvm.profiling import timed_peak_memory
from fysvm.rule_svm import FuzzyRuleSVM
from fysvm.run_metadata import write_run_metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="Dataset slugs. Defaults to datasets with d > 100.")
    parser.add_argument("--data-dir", default="datasets/prepared")
    parser.add_argument("--output-dir", default="runs/highdim-adaptive")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args(argv)

    slugs = args.datasets or _highdim_slugs(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir, config={**vars(args), "datasets": slugs})

    rows: list[dict[str, Any]] = []
    for slug in slugs:
        dataset = load_dataset(slug, args.data_dir)
        seed = _stable_dataset_seed(args.random_state, slug)
        if args.max_samples is not None and dataset.X.shape[0] > args.max_samples:
            dataset = _stratified_subset(dataset, args.max_samples, seed)
        rows.extend(_evaluate_dataset(dataset, args.outer_splits, seed))

    summaries = _summarize(rows)
    _write_csv(output_dir / "fold_metrics.csv", rows)
    _write_csv(output_dir / "metrics.csv", summaries)
    (output_dir / "metrics.json").write_text(
        json.dumps({"fold_metrics": rows, "summary_metrics": summaries}, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_report(summaries), encoding="utf-8")
    print(output_dir)


def _evaluate_dataset(dataset: PreparedDataset, outer_splits: int, seed: int) -> list[dict[str, Any]]:
    model_builders = {
        "adaptive_fuzzy_rule_svm": lambda: FuzzyRuleSVM(
            C=1.0,
            penalty="l1",
            max_rule_length=2,
            max_rules=512,
            min_rule_coverage=0.01,
            rule_length_penalty=0.35,
            feature_screening="anova" if dataset.X.shape[1] > 100 else "none",
            screen_top_k=32 if dataset.X.shape[1] > 100 else None,
            feature_names=dataset.feature_names,
            class_weight="balanced",
            random_state=seed,
            max_iter=20000,
        ),
        "column_generation_fuzzy_rule_svm": lambda: FuzzyRuleSVM(
            C=1.0,
            penalty="l1",
            max_rule_length=2,
            max_rules=512,
            min_rule_coverage=0.01,
            rule_length_penalty=0.35,
            rule_generation="column_generation",
            feature_names=dataset.feature_names,
            class_weight="balanced",
            random_state=seed,
            max_iter=20000,
        ),
        "membership_svm": lambda: MembershipSVM(class_weight="balanced", random_state=seed, max_iter=20000),
        "ebm": lambda: _build_ebm(dataset, seed),
    }
    y = np.asarray(dataset.y)
    splits = min(outer_splits, int(np.min(np.unique(y, return_counts=True)[1])))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(cv.split(dataset.X, y), start=1):
        for model_key, builder in model_builders.items():
            estimator = builder()
            if dataset.spec.task == "multiclass" and model_key != "ebm":
                estimator = OneVsRestClassifier(estimator)
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                _, fit_seconds, fit_peak_memory_mb = timed_peak_memory(lambda: pipe.fit(dataset.X[train], y[train]))
            y_pred, predict_seconds, predict_peak_memory_mb = timed_peak_memory(lambda: pipe.predict(dataset.X[test]))
            row = {
                "dataset": dataset.spec.slug,
                "model_key": model_key,
                "model": model_key.replace("_", " "),
                "fold": fold,
                "n_features": dataset.X.shape[1],
                "balanced_accuracy": float(balanced_accuracy_score(y[test], y_pred)),
                "f1_macro": float(f1_score(y[test], y_pred, average="macro", zero_division=0)),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "fit_peak_memory_mb": fit_peak_memory_mb,
                "predict_peak_memory_mb": predict_peak_memory_mb,
            }
            model = pipe.named_steps["model"]
            if model_key in ("adaptive_fuzzy_rule_svm", "column_generation_fuzzy_rule_svm"):
                base = model.estimators_[0] if hasattr(model, "estimators_") else model
                row["candidate_rules"] = float(getattr(base, "n_candidate_rules_", 0))
                row["retained_rules"] = float(getattr(base, "n_rules_", 0))
                row["retention_ratio"] = row["retained_rules"] / max(1.0, row["candidate_rules"])
                row["screened_features"] = float(getattr(base, "n_screened_features_", dataset.X.shape[1]))
            if model_key == "membership_svm":
                row["membership_nonzero_coefs"] = membership_nonzero_coef_count(model)
            rows.append(row)
    return rows


def _highdim_slugs(data_dir: str) -> list[str]:
    slugs: list[str] = []
    for spec in DATASET_SPECS:
        try:
            dataset = load_dataset(spec.slug, data_dir)
        except FileNotFoundError:
            continue
        if dataset.X.shape[1] > 100:
            slugs.append(spec.slug)
    return slugs


def _build_ebm(dataset: PreparedDataset, seed: int) -> Any:
    from interpret.glassbox import ExplainableBoostingClassifier

    return ExplainableBoostingClassifier(
        interactions=0 if dataset.X.shape[1] > 100 else 5,
        max_rounds=200,
        outer_bags=4,
        inner_bags=0,
        n_jobs=1,
        random_state=seed,
    )


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["model_key"]), []).append(row)
    summaries = []
    for (dataset, model_key), group in sorted(groups.items()):
        score = float(np.mean([row["balanced_accuracy"] for row in group]))
        summaries.append({
            "dataset": dataset,
            "model_key": model_key,
            "model": group[0]["model"],
            "n_features": group[0]["n_features"],
            "balanced_accuracy_mean": score,
            "f1_macro_mean": float(np.mean([row["f1_macro"] for row in group])),
            "fit_seconds_mean": float(np.mean([row["fit_seconds"] for row in group])),
        })
    best_by_dataset: dict[str, str] = {}
    for dataset in {row["dataset"] for row in summaries}:
        best = max(
            (row for row in summaries if row["dataset"] == dataset),
            key=lambda item: item["balanced_accuracy_mean"],
        )
        best_by_dataset[dataset] = best["model"]
    for row in summaries:
        # This script compares outer-test summaries retrospectively. Deployable
        # model selection must use diagnostics computed inside the training fold.
        row["retrospective_best_model"] = best_by_dataset[row["dataset"]]
    return summaries


def _report(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Adaptive High-Dimensional Comparison",
        "",
        "Retrospective best models are selected from outer-test summaries "
        "and are not deployable policy recommendations.",
        "",
        "| Dataset | Model | Balanced Accuracy | Retrospective best |",
        "|---|---|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['balanced_accuracy_mean']:.3f} | "
            f"{row['retrospective_best_model']} |"
        )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
