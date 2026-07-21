"""GA-budget sensitivity and high-budget fuzzy baseline comparison.

Runs FARC-HD and IVTURS at three GA budget levels (20, 200, 1000 evaluations)
on a representative 5-dataset subset plus a high-budget (1000-evaluation)
full comparison against FuzzyRuleSVM and FURIA.

Design decisions (documented):
- Budget levels: 20 (current paper baseline), 200 (10x), 1000 (50x)
  Canonical KEEL budget is 20,000; 1000 is 5% of canonical, representing
  a meaningful step while remaining computationally feasible (~3-4h total).
  2000 evaluations was tested but exceeded 7 min/fold on pima_diabetes,
  making a full 5-dataset 5-fold run impractical in wall-clock time.
- Representative subset: iris, pima_diabetes, heart_cleveland, wine,
  breast_cancer_diagnostic -- spans sizes (150-569 samples), feature
  dimensions (4-30), task types (binary + multiclass), and medical / general
  domains.
- Population size: 30 at all budget levels (canonical = 50; 30 is a
  reasonable intermediate value that fits in a reasonable wall-clock budget).
- The high-budget comparison uses 1000 evaluations and includes FuzzyRuleSVM
  and FURIA for a fair, same-protocol head-to-head on the subset.

Typical usage:
  # Budget sensitivity only (faster, ~1 h)
  uv run python scripts/compare_fuzzy_highbudget.py --mode sensitivity

  # High-budget full comparison only (~3 h)
  uv run python scripts/compare_fuzzy_highbudget.py --mode comparison

  # Both (default, ~4 h)
  uv run python scripts/compare_fuzzy_highbudget.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from fysvm.baselines import FARCHDClassifier, FURIAClassifier, IVTURSClassifier
from fysvm.datasets import DATASET_SPECS, PreparedDataset, load_dataset
from fysvm.rule_svm import FuzzyRuleSVM

# ---------------------------------------------------------------------------
# Representative subset
# ---------------------------------------------------------------------------

SUBSET_SLUGS = [
    "iris",                    # 150×4, multiclass
    "pima_diabetes",           # 768×8, binary, medical
    "heart_cleveland",         # 303×13, binary, medical
    "wine",                    # 178×13, multiclass
    "breast_cancer_diagnostic", # 569×30, binary, medical
]

BUDGET_LEVELS = [20, 200, 1000]   # GA evaluation counts for sensitivity sweep
# Decision: 1000 evaluations is 5% of canonical KEEL budget (20,000).
# At 2000 the runtime on pima_diabetes exceeds 7 min/fold, making a 5-dataset
# 5-fold run infeasible in wall-clock time. 1000 evaluations (≈3-4 h total)
# represents a meaningful improvement over the 20-evaluation baseline while
# remaining computationally tractable.


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunResult:
    sensitivity_rows: list[dict[str, Any]]
    comparison_rows: list[dict[str, Any]]
    output_dir: Path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["sensitivity", "comparison", "both"],
        default="both",
        help="Which experiment to run (default: both).",
    )
    parser.add_argument("--data-dir", default="datasets/prepared")
    parser.add_argument("--output-dir", default="runs/fuzzy-highbudget")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args(argv)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sensitivity_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    if args.mode in ("sensitivity", "both"):
        print("\n=== GA Budget Sensitivity Sweep ===")
        sensitivity_rows = run_budget_sensitivity(
            data_dir=args.data_dir,
            outer_splits=args.outer_splits,
            random_state=args.random_state,
        )
        _write_csv(output_path / "sensitivity_rows.csv", sensitivity_rows)
        _write_json(output_path / "sensitivity_rows.json", sensitivity_rows)
        print(f"Saved sensitivity results to {output_path}/sensitivity_rows.*")

    if args.mode in ("comparison", "both"):
        print("\n=== High-Budget Full Comparison (2000 evaluations) ===")
        comparison_rows = run_high_budget_comparison(
            data_dir=args.data_dir,
            outer_splits=args.outer_splits,
            random_state=args.random_state,
        )
        _write_csv(output_path / "comparison_rows.csv", comparison_rows)
        _write_json(output_path / "comparison_rows.json", comparison_rows)
        print(f"Saved comparison results to {output_path}/comparison_rows.*")

    # Build combined markdown report
    report = _build_report(sensitivity_rows, comparison_rows)
    report_path = Path("docs/fuzzy_highbudget_comparison.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {report_path}")

    result = RunResult(sensitivity_rows, comparison_rows, output_path)
    return result


# ---------------------------------------------------------------------------
# Budget sensitivity sweep
# ---------------------------------------------------------------------------

def run_budget_sensitivity(
    *,
    data_dir: str | Path,
    outer_splits: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Run FARC-HD and IVTURS at three GA budget levels on the representative subset."""
    rows: list[dict[str, Any]] = []
    data_path = Path(data_dir)

    for ds_idx, slug in enumerate(SUBSET_SLUGS):
        print(f"\n  Dataset {ds_idx+1}/{len(SUBSET_SLUGS)}: {slug}")
        dataset = load_dataset(slug, data_path)
        n_features = dataset.X.shape[1]
        max_depth = 2 if n_features <= 32 else 1

        for budget in BUDGET_LEVELS:
            for model_name, ModelClass in [("FARC-HD", FARCHDClassifier), ("IVTURS", IVTURSClassifier)]:
                print(f"    {model_name}, evals={budget}", end=" ... ", flush=True)
                fold_rows = _evaluate_with_budget(
                    dataset, ModelClass, budget,
                    max_depth=max_depth,
                    outer_splits=outer_splits,
                    random_state=random_state + ds_idx * 1000,
                )
                for row in fold_rows:
                    row["budget"] = budget
                    row["model"] = model_name
                    rows.append(row)
                mean_ba = np.mean([r["balanced_accuracy"] for r in fold_rows])
                print(f"mean_bal_acc={mean_ba:.3f}")

    return rows


def _evaluate_with_budget(
    dataset: PreparedDataset,
    ModelClass,
    budget: int,
    *,
    max_depth: int,
    outer_splits: int,
    random_state: int,
) -> list[dict[str, Any]]:
    y = np.asarray(dataset.y)
    class_counts = np.unique(y, return_counts=True)[1]
    splits = min(outer_splits, int(np.min(class_counts)))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(dataset.X, y), start=1):
        X_train, X_test = dataset.X[train_idx], dataset.X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ModelClass(
                evaluations=budget,
                pop=30,
                max_depth=max_depth,
                min_support=0.05,
                random_state=random_state + fold,
            )),
        ])
        t = time.perf_counter()
        with _suppressed_warnings():
            pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t

        y_pred = pipeline.predict(X_test)
        rows.append({
            "dataset": dataset.spec.slug,
            "dataset_name": dataset.spec.name,
            "fold": fold,
            "n_samples": int(dataset.X.shape[0]),
            "n_features": int(dataset.X.shape[1]),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "fit_seconds": fit_seconds,
        })
    return rows


# ---------------------------------------------------------------------------
# High-budget full comparison
# ---------------------------------------------------------------------------

def run_high_budget_comparison(
    *,
    data_dir: str | Path,
    outer_splits: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Full comparison: FuzzyRuleSVM vs FURIA vs FARC-HD vs IVTURS at 2000 evals."""
    rows: list[dict[str, Any]] = []
    data_path = Path(data_dir)

    for ds_idx, slug in enumerate(SUBSET_SLUGS):
        print(f"\n  Dataset {ds_idx+1}/{len(SUBSET_SLUGS)}: {slug}")
        dataset = load_dataset(slug, data_path)
        n_features = dataset.X.shape[1]
        max_depth = 2 if n_features <= 32 else 1

        # --- FuzzyRuleSVM ---
        print("    FuzzyRuleSVM", end=" ... ", flush=True)
        max_rules = min(256, max(24, 3 * n_features))
        frs_rows = _evaluate_fuzzy_rule_svm(
            dataset, outer_splits=outer_splits,
            random_state=random_state + ds_idx * 1000,
            max_rules=max_rules, max_depth=max_depth,
        )
        for row in frs_rows:
            row["model"] = "FuzzyRuleSVM"
        rows.extend(frs_rows)
        print(f"mean_bal_acc={np.mean([r['balanced_accuracy'] for r in frs_rows]):.3f}")

        # --- FURIA ---
        print("    FURIA", end=" ... ", flush=True)
        furia_rows = _evaluate_furia(
            dataset, outer_splits=outer_splits,
            random_state=random_state + ds_idx * 1000,
        )
        for row in furia_rows:
            row["model"] = "FURIA"
        rows.extend(furia_rows)
        print(f"mean_bal_acc={np.mean([r['balanced_accuracy'] for r in furia_rows]):.3f}")

        # --- FARC-HD at 1000 evals ---
        print("    FARC-HD (1000 evals)", end=" ... ", flush=True)
        farchd_rows = _evaluate_with_budget(
            dataset, FARCHDClassifier, 1000,
            max_depth=max_depth,
            outer_splits=outer_splits,
            random_state=random_state + ds_idx * 1000,
        )
        for row in farchd_rows:
            row["model"] = "FARC-HD (1000 evals)"
            row["budget"] = 1000
        rows.extend(farchd_rows)
        print(f"mean_bal_acc={np.mean([r['balanced_accuracy'] for r in farchd_rows]):.3f}")

        # --- IVTURS at 1000 evals ---
        print("    IVTURS (1000 evals)", end=" ... ", flush=True)
        ivturs_rows = _evaluate_with_budget(
            dataset, IVTURSClassifier, 1000,
            max_depth=max_depth,
            outer_splits=outer_splits,
            random_state=random_state + ds_idx * 1000,
        )
        for row in ivturs_rows:
            row["model"] = "IVTURS (1000 evals)"
            row["budget"] = 1000
        rows.extend(ivturs_rows)
        print(f"mean_bal_acc={np.mean([r['balanced_accuracy'] for r in ivturs_rows]):.3f}")

    return rows


def _evaluate_fuzzy_rule_svm(
    dataset: PreparedDataset,
    *,
    outer_splits: int,
    random_state: int,
    max_rules: int,
    max_depth: int,
) -> list[dict[str, Any]]:
    y = np.asarray(dataset.y)
    class_counts = np.unique(y, return_counts=True)[1]
    splits = min(outer_splits, int(np.min(class_counts)))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(dataset.X, y), start=1):
        X_train, X_test = dataset.X[train_idx], dataset.X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Simple inner-CV parameter selection (C and penalty)
        best_ba = -1.0
        best_params = {"C": 1.0, "penalty": "l1"}
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state + fold)
        for C in (0.3, 1.0, 3.0):
            for penalty in ("l1", "l2"):
                scores = []
                for it, iv in inner_cv.split(X_train, y_train):
                    base = FuzzyRuleSVM(
                        C=C, penalty=penalty,
                        max_rule_length=max_depth, max_rules=max_rules,
                        min_rule_coverage=0.01, rule_length_penalty=0.35,
                        feature_names=dataset.feature_names,
                        class_weight="balanced",
                        random_state=random_state + fold,
                        max_iter=20000,
                    )
                    model = OneVsRestClassifier(base) if dataset.spec.task == "multiclass" else base
                    pipeline = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", model),
                    ])
                    with _suppressed_warnings():
                        pipeline.fit(X_train[it], y_train[it])
                    scores.append(balanced_accuracy_score(y_train[iv], pipeline.predict(X_train[iv])))
                if np.mean(scores) > best_ba:
                    best_ba = float(np.mean(scores))
                    best_params = {"C": C, "penalty": penalty}

        base = FuzzyRuleSVM(
            **best_params,
            max_rule_length=max_depth, max_rules=max_rules,
            min_rule_coverage=0.01, rule_length_penalty=0.35,
            feature_names=dataset.feature_names,
            class_weight="balanced",
            random_state=random_state + fold,
            max_iter=20000,
        )
        model = OneVsRestClassifier(base) if dataset.spec.task == "multiclass" else base
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ])
        t = time.perf_counter()
        with _suppressed_warnings():
            pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t

        y_pred = pipeline.predict(X_test)
        rows.append({
            "dataset": dataset.spec.slug,
            "dataset_name": dataset.spec.name,
            "fold": fold,
            "n_samples": int(dataset.X.shape[0]),
            "n_features": int(dataset.X.shape[1]),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "fit_seconds": fit_seconds,
            "selected_params": json.dumps(best_params),
        })
    return rows


def _evaluate_furia(
    dataset: PreparedDataset,
    *,
    outer_splits: int,
    random_state: int,
) -> list[dict[str, Any]]:
    y = np.asarray(dataset.y)
    class_counts = np.unique(y, return_counts=True)[1]
    splits = min(outer_splits, int(np.min(class_counts)))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(dataset.X, y), start=1):
        X_train, X_test = dataset.X[train_idx], dataset.X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner-CV for min_no
        best_ba = -1.0
        best_min_no = 2.0
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state + fold)
        for min_no in (1.0, 2.0, 3.0):
            scores = []
            for it, iv in inner_cv.split(X_train, y_train):
                pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", FURIAClassifier(min_no=min_no, random_state=random_state + fold)),
                ])
                with _suppressed_warnings():
                    pipeline.fit(X_train[it], y_train[it])
                scores.append(balanced_accuracy_score(y_train[iv], pipeline.predict(X_train[iv])))
            if np.mean(scores) > best_ba:
                best_ba = float(np.mean(scores))
                best_min_no = min_no

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", FURIAClassifier(min_no=best_min_no, random_state=random_state + fold)),
        ])
        t = time.perf_counter()
        with _suppressed_warnings():
            pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t

        y_pred = pipeline.predict(X_test)
        rows.append({
            "dataset": dataset.spec.slug,
            "dataset_name": dataset.spec.name,
            "fold": fold,
            "n_samples": int(dataset.X.shape[0]),
            "n_features": int(dataset.X.shape[1]),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "fit_seconds": fit_seconds,
            "selected_params": json.dumps({"min_no": best_min_no}),
        })
    return rows


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _build_report(
    sensitivity_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# High-Budget FARC-HD/IVTURS Comparison",
        "",
        "## Design",
        "",
        "**Representative subset**: iris, pima_diabetes, heart_cleveland, wine, breast_cancer_diagnostic.",
        "",
        "**Budget levels** (sensitivity sweep): 20 (current paper baseline), 200 (10x), 1000 (50x).",
        "Canonical KEEL budget is 20,000 evaluations; 1000 represents 5% of canonical.",
        "",
        "**High-budget comparison**: FuzzyRuleSVM, FURIA, FARC-HD (1000 evals), IVTURS (1000 evals).",
        "",
    ]

    if sensitivity_rows:
        lines += _budget_sensitivity_section(sensitivity_rows)

    if comparison_rows:
        lines += _comparison_section(comparison_rows)

    return "\n".join(lines)


def _budget_sensitivity_section(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## GA Budget Sensitivity",
        "",
        "Mean balanced accuracy by dataset, model, and budget.",
        "",
    ]
    # Group by dataset × model × budget
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["budget"])].append(row["balanced_accuracy"])

    headers = ["Dataset", "Model"] + [f"evals={b}" for b in BUDGET_LEVELS]
    table_rows = []
    for slug in SUBSET_SLUGS:
        first = True
        for model in ("FARC-HD", "IVTURS"):
            r = [slug if first else "", model]
            first = False
            for budget in BUDGET_LEVELS:
                key = (slug, model, budget)
                vals = grouped.get(key, [])
                r.append(f"{np.mean(vals):.3f}" if vals else "-")
            table_rows.append(r)
    lines += _markdown_table(headers, table_rows)
    lines += [""]
    return lines


def _comparison_section(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## High-Budget Full Comparison (1000 evaluations)",
        "",
        "5-fold nested CV on the representative 5-dataset subset.",
        "",
    ]
    model_order = ["FuzzyRuleSVM", "FURIA", "FARC-HD (1000 evals)", "IVTURS (1000 evals)"]
    by_model_ds: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_model_ds[(row["dataset"], row["model"])].append(row["balanced_accuracy"])

    # Per-dataset table
    headers = ["Dataset", "FuzzyRuleSVM", "FURIA", "FARC-HD (1000)", "IVTURS (1000)"]
    table_rows = []
    for slug in SUBSET_SLUGS:
        r = [slug]
        for m in ["FuzzyRuleSVM", "FURIA", "FARC-HD (1000 evals)", "IVTURS (1000 evals)"]:
            vals = by_model_ds.get((slug, m), [])
            r.append(f"{np.mean(vals):.3f}" if vals else "-")
        table_rows.append(r)
    lines += _markdown_table(headers, table_rows)

    # Overall means
    lines += ["", "### Overall means", ""]
    for m in model_order:
        all_vals = []
        for slug in SUBSET_SLUGS:
            all_vals.extend(by_model_ds.get((slug, m), []))
        if all_vals:
            lines.append(f"- **{m}**: mean bal acc = {np.mean(all_vals):.3f}")
    lines += [""]

    # Wilcoxon test: FuzzyRuleSVM vs each baseline (per-dataset means)
    frs_per_ds = [
        np.mean(by_model_ds.get((slug, "FuzzyRuleSVM"), [float("nan")]))
        for slug in SUBSET_SLUGS
    ]
    lines += [    "### Wilcoxon signed-rank tests (FuzzyRuleSVM vs baselines, high-budget 1000 evals)", ""]
    for m in ["FURIA", "FARC-HD (1000 evals)", "IVTURS (1000 evals)"]:
        baseline_per_ds = [
            np.mean(by_model_ds.get((slug, m), [float("nan")]))
            for slug in SUBSET_SLUGS
        ]
        diffs = np.array(frs_per_ds) - np.array(baseline_per_ds)
        try:
            pval = wilcoxon(diffs, zero_method="wilcox").pvalue if not np.all(np.isclose(diffs, 0)) else 1.0
            pstr = f"p={pval:.3f}" if pval >= 0.001 else "p<0.001"
        except ValueError:
            pstr = "p=n/a (n<5)"
        wins = int(np.sum(diffs > 0))
        losses = int(np.sum(diffs < 0))
        lines.append(
            f"- FuzzyRuleSVM vs {m}: mean delta = {np.mean(diffs):+.3f}, "
            f"W/L = {wins}/{losses}, {pstr}"
        )
    lines += [""]
    return lines


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


@contextmanager
def _suppressed_warnings() -> Iterable[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore")
        yield


if __name__ == "__main__":
    main()
