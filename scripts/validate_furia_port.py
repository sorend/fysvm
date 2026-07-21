"""Run a small sanity check for the local FURIA Python port."""

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
from sklearn.pipeline import Pipeline

from fysvm.baselines import FURIAClassifier
from fysvm.datasets import PreparedDataset, load_dataset
from fysvm.evaluation import _json_default, _stable_dataset_seed, _stratified_subset
from fysvm.run_metadata import write_run_metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=["iris", "pima_diabetes", "statlog_heart"])
    parser.add_argument("--data-dir", default="datasets/prepared")
    parser.add_argument("--output-dir", default="runs/furia-validation")
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir, config=vars(args))
    rows: list[dict[str, Any]] = []
    for slug in args.datasets:
        dataset = load_dataset(slug, args.data_dir)
        seed = _stable_dataset_seed(args.random_state, slug)
        if args.max_samples is not None and dataset.X.shape[0] > args.max_samples:
            dataset = _stratified_subset(dataset, args.max_samples, seed)
        rows.extend(_evaluate(dataset, args.outer_splits, seed))

    _write_csv(output_dir / "metrics.csv", rows)
    (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2, default=_json_default) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(_report(rows), encoding="utf-8")
    print(output_dir)


def _evaluate(dataset: PreparedDataset, outer_splits: int, seed: int) -> list[dict[str, Any]]:
    y = np.asarray(dataset.y)
    splits = min(outer_splits, int(np.min(np.unique(y, return_counts=True)[1])))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (train, test) in enumerate(cv.split(dataset.X, y), start=1):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", FURIAClassifier(random_state=seed + fold)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipe.fit(dataset.X[train], y[train])
        y_pred = pipe.predict(dataset.X[test])
        rows.append({
            "dataset": dataset.spec.slug,
            "fold": fold,
            "balanced_accuracy": float(balanced_accuracy_score(y[test], y_pred)),
            "f1_macro": float(f1_score(y[test], y_pred, average="macro", zero_division=0)),
            "status": "completed",
        })
    return rows


def _report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# FURIA Port Sanity Check",
        "",
        "This artifact verifies that the local Python FURIA wrapper completes shared-dataset folds under the repository preprocessing protocol. It is not a claim that the port exactly reproduces every result from the original Java/Weka implementation.",
        "",
        "| Dataset | Mean Balanced Accuracy | Mean Macro F1 |",
        "|---|---:|---:|",
    ]
    for dataset in sorted({row["dataset"] for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset]
        lines.append(f"| {dataset} | {np.mean([r['balanced_accuracy'] for r in group]):.3f} | {np.mean([r['f1_macro'] for r in group]):.3f} |")
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
