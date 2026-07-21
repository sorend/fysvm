"""Compare rule-space models on controlled synthetic regimes."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fysvm.evaluation import _json_default
from fysvm.membership import MembershipSVM
from fysvm.profiling import timed_peak_memory
from fysvm.rule_svm import FuzzyRuleSVM
from fysvm.run_metadata import write_run_metadata
from fysvm.synthetic import (
    SyntheticDataset,
    make_additive_main_effects,
    make_pairwise_fuzzy_interaction,
    make_sparse_fuzzy_rule_ground_truth,
    make_xor_interaction,
)


Generator = Callable[..., SyntheticDataset]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/synthetic-regimes")
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--n-noise", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir, config=vars(args))

    rows = run_experiment(
        n_samples=args.n_samples,
        repeats=args.repeats,
        outer_splits=args.outer_splits,
        n_noise=args.n_noise,
        random_state=args.random_state,
    )
    _write_csv(output_dir / "fold_metrics.csv", rows)
    summaries = _summarize(rows)
    _write_csv(output_dir / "metrics.csv", summaries)
    (output_dir / "metrics.json").write_text(
        json.dumps({"fold_metrics": rows, "summary_metrics": summaries}, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_report(summaries), encoding="utf-8")
    print(output_dir)


def run_experiment(*, n_samples: int, repeats: int, outer_splits: int, n_noise: int, random_state: int) -> list[dict[str, Any]]:
    generators: list[tuple[str, Generator]] = [
        ("additive", make_additive_main_effects),
        ("pairwise", make_pairwise_fuzzy_interaction),
        ("xor", make_xor_interaction),
        ("sparse_rule", make_sparse_fuzzy_rule_ground_truth),
    ]
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for regime_key, generator in generators:
            data = generator(n_samples=n_samples, n_noise=n_noise, random_state=random_state + repeat)
            rows.extend(_evaluate_dataset(data, regime_key, repeat, outer_splits, random_state + repeat))
    return rows


def _evaluate_dataset(data: SyntheticDataset, regime_key: str, repeat: int, outer_splits: int, seed: int) -> list[dict[str, Any]]:
    specs = {
        "frs_length1": lambda: FuzzyRuleSVM(max_rule_length=1, max_rules=256, class_weight="balanced", random_state=seed),
        "frs_length2": lambda: FuzzyRuleSVM(max_rule_length=2, max_rules=512, class_weight="balanced", random_state=seed),
        "membership_svm": lambda: MembershipSVM(class_weight="balanced", random_state=seed),
        "rbf_svm": lambda: SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced", random_state=seed),
    }
    cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(cv.split(data.X, data.y), start=1):
        for model_key, builder in specs.items():
            steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
            if model_key == "rbf_svm":
                steps.append(("scaler", StandardScaler()))
            estimator = builder()
            if len(np.unique(data.y)) > 2 and model_key != "rbf_svm":
                estimator = OneVsRestClassifier(estimator)
            steps.append(("model", estimator))
            pipe = Pipeline(steps)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                _, fit_seconds, fit_peak_memory_mb = timed_peak_memory(lambda: pipe.fit(data.X[train], data.y[train]))
            y_pred, predict_seconds, predict_peak_memory_mb = timed_peak_memory(lambda: pipe.predict(data.X[test]))
            rows.append({
                "regime": regime_key,
                "repeat": repeat,
                "fold": fold,
                "model_key": model_key,
                "model": model_key.replace("_", " "),
                "n_features": data.X.shape[1],
                "balanced_accuracy": float(balanced_accuracy_score(data.y[test], y_pred)),
                "f1_macro": float(f1_score(data.y[test], y_pred, average="macro", zero_division=0)),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "fit_peak_memory_mb": fit_peak_memory_mb,
                "predict_peak_memory_mb": predict_peak_memory_mb,
            })
    return rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["regime"], row["model_key"]), []).append(row)
    summaries = []
    for (regime, model_key), group in sorted(groups.items()):
        summaries.append({
            "regime": regime,
            "model_key": model_key,
            "model": group[0]["model"],
            "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in group])),
            "balanced_accuracy_std": float(np.std([r["balanced_accuracy"] for r in group], ddof=1)) if len(group) > 1 else 0.0,
            "f1_macro_mean": float(np.mean([r["f1_macro"] for r in group])),
        })
    return summaries


def _report(summaries: list[dict[str, Any]]) -> str:
    lines = ["# Synthetic Regime Experiments", "", "| Regime | Model | Balanced Accuracy | Macro F1 |", "|---|---|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['regime']} | {row['model']} | {row['balanced_accuracy_mean']:.3f} | {row['f1_macro_mean']:.3f} |")
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
