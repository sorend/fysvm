"""Build a simple regime map from synthetic and real-data comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fysvm.run_metadata import write_run_metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-dir", default="runs/regime-analysis")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir, config=vars(args))
    rows = _synthetic_rows(runs_dir) + _real_rows(runs_dir)
    _write_csv(output_dir / "metrics.csv", rows)
    (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


def _synthetic_rows(runs_dir: Path) -> list[dict[str, Any]]:
    path = runs_dir / "synthetic-regimes" / "metrics.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for regime, group in df.groupby("regime"):
        scores = {row.model_key: row.balanced_accuracy_mean for row in group.itertuples()}
        rows.append({
            "source": "synthetic",
            "dataset_or_regime": regime,
            "interaction_proxy": 1.0 if regime in {"pairwise", "xor", "sparse_rule"} else 0.0,
            "fuzzy_minus_membership": float(scores.get("frs_length2", np.nan) - scores.get("membership_svm", np.nan)),
            "fuzzy_minus_rbf": float(scores.get("frs_length2", np.nan) - scores.get("rbf_svm", np.nan)),
        })
    return rows


def _real_rows(runs_dir: Path) -> list[dict[str, Any]]:
    path = runs_dir / "ablation-membership" / "metrics.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for dataset, group in df.groupby("dataset"):
        scores = {row.model_key: row.balanced_accuracy_mean for row in group.itertuples()}
        n_features = int(group.iloc[0].get("n_features", 0))
        rows.append({
            "source": "real",
            "dataset_or_regime": dataset,
            "interaction_proxy": float(n_features <= 32),
            "fuzzy_minus_membership": float(scores.get("fuzzy_rule_svm", np.nan) - scores.get("membership_svm", np.nan)),
            "fuzzy_minus_logistic_l1": float(scores.get("fuzzy_rule_svm", np.nan) - scores.get("membership_logistic_l1", np.nan)),
            "n_features": n_features,
        })
    return rows


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
