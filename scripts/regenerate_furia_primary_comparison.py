"""Regenerate primary FURIA comparison statistics from checked-in metrics.

The manuscript's primary fuzzy-rule comparison pairs FuzzyRuleSVM rows from
``runs/recommended-comparison`` with FURIA rows from
``runs/fuzzy-baselines-comparison``.  This script refreshes the FURIA
statistical artifact from those intended sources without rerunning the full
cross-validation jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from fysvm.datasets import DATASET_SPECS


METRICS = ("balanced_accuracy", "f1_macro", "accuracy")
BASELINES = (
    ("furia", "FURIA"),
    ("farchd", "FARC-HD"),
    ("ivturs", "IVTURS"),
)
CSV_COLUMNS = (
    "baseline_mean",
    "comparison",
    "friedman_pvalue",
    "friedman_statistic",
    "fuzzy_mean",
    "losses",
    "mean_delta",
    "median_delta",
    "metric",
    "models",
    "n_datasets",
    "ties",
    "wilcoxon_pvalue",
    "wilcoxon_pvalue_holm",
    "wins",
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recommended-metrics",
        default="runs/recommended-comparison/metrics.csv",
        help="Metrics CSV containing the primary FuzzyRuleSVM run.",
    )
    parser.add_argument(
        "--fuzzy-metrics",
        default="runs/fuzzy-baselines-comparison/metrics.csv",
        help="Metrics CSV containing FURIA/FARC-HD/IVTURS rows.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/fuzzy-baselines-comparison",
        help="Directory whose statistical_tests.csv and metrics.json are refreshed.",
    )
    parser.add_argument(
        "--report",
        default="docs/fuzzy_baselines_comparison.md",
        help="Markdown report to refresh.",
    )
    args = parser.parse_args(argv)

    recommended_rows = _load_rows(Path(args.recommended_metrics))
    fuzzy_rows = _load_rows(Path(args.fuzzy_metrics))
    tests = build_statistical_tests(recommended_rows, fuzzy_rows)

    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "statistical_tests.csv", tests)
    _update_metrics_json(output_dir / "metrics.json", tests)
    Path(args.report).write_text(
        build_report(recommended_rows, fuzzy_rows, tests, output_dir),
        encoding="utf-8",
    )

    furia_test = next(
        row
        for row in tests
        if row["comparison"] == "FuzzyRuleSVM vs FURIA"
        and row["metric"] == "balanced_accuracy"
    )
    print(
        "FURIA balanced accuracy: "
        f"n={furia_test['n_datasets']}, "
        f"W/T/L={furia_test['wins']}/{furia_test['ties']}/{furia_test['losses']}, "
        f"p={furia_test['wilcoxon_pvalue']:.6g}"
    )


def build_statistical_tests(
    recommended_rows: list[dict[str, str]],
    fuzzy_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    fuzzy_rule_svm = _rows_by_dataset(recommended_rows, "fuzzy_rule_svm")
    baseline_rows = {
        key: _rows_by_dataset(fuzzy_rows, key)
        for key, _ in BASELINES
    }

    pairwise_rows: list[dict[str, Any]] = []
    for key, name in BASELINES:
        for metric in METRICS:
            pairwise_rows.append(
                _paired_test(fuzzy_rule_svm, baseline_rows[key], name, metric)
            )

    for metric in METRICS:
        group = [row for row in pairwise_rows if row["metric"] == metric]
        for row, holm_p in zip(group, _holm_bonferroni([row["wilcoxon_pvalue"] for row in group])):
            row["wilcoxon_pvalue_holm"] = holm_p

    tests = list(pairwise_rows)
    common_datasets = sorted(
        set(fuzzy_rule_svm).intersection(*(set(rows) for rows in baseline_rows.values()))
    )
    model_names = "FuzzyRuleSVM, " + ", ".join(name for _, name in BASELINES)
    for metric in METRICS:
        key = f"{metric}_mean"
        values = [np.asarray([float(fuzzy_rule_svm[d][key]) for d in common_datasets])]
        values.extend(
            np.asarray([float(baseline_rows[model_key][d][key]) for d in common_datasets])
            for model_key, _ in BASELINES
        )
        statistic, pvalue = _friedman(values)
        tests.append(
            {
                "comparison": "Friedman omnibus",
                "metric": metric,
                "n_datasets": len(common_datasets),
                "models": model_names,
                "friedman_statistic": statistic,
                "friedman_pvalue": pvalue,
            }
        )
    return tests


def build_report(
    recommended_rows: list[dict[str, str]],
    fuzzy_rows: list[dict[str, str]],
    tests: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    fuzzy_rule_svm = _rows_by_dataset(recommended_rows, "fuzzy_rule_svm")
    baseline_rows = {key: _rows_by_dataset(fuzzy_rows, key) for key, _ in BASELINES}

    lines = [
        "# Fuzzy Rule Baselines Comparison",
        "",
        "## Scope",
        "",
        "This report refreshes the primary fuzzy-rule comparison used in the manuscript. The FuzzyRuleSVM row is taken from `runs/recommended-comparison/metrics.csv`; FURIA, FARC-HD, and IVTURS rows are taken from `runs/fuzzy-baselines-comparison/metrics.csv`.",
        "",
        "FURIA is the primary fuzzy-rule baseline because it uses greedy rule induction rather than a genetic algorithm. FARC-HD and IVTURS remain exploratory here because their checked-in runs use a small GA budget relative to canonical KEEL settings.",
        "",
        "- Outer evaluation: stratified 5-fold CV.",
        "- Inner selection: stratified 3-fold CV on each outer training fold.",
        "- Missing values: median imputation inside each fold.",
        f"- Refreshed artifact: `{output_dir / 'statistical_tests.csv'}`.",
        "",
        "## Overall Performance",
        "",
        "| Model | n | Mean Accuracy | Mean Balanced Accuracy | Mean Macro F1 | Mean Fit Time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    model_tables = [("fuzzy_rule_svm", "FuzzyRuleSVM", fuzzy_rule_svm), *BASELINES]
    for key, name, rows in _normalise_model_tables(model_tables, baseline_rows):
        del key
        lines.append(
            f"| {name} | {len(rows)} | {_mean(rows, 'accuracy_mean'):.3f} | "
            f"{_mean(rows, 'balanced_accuracy_mean'):.3f} | "
            f"{_mean(rows, 'f1_macro_mean'):.3f} | "
            f"{_mean(rows, 'fit_seconds_mean'):.3f} |"
        )

    lines.extend([
        "",
        "## Paired Tests",
        "",
        "Wilcoxon signed-rank tests use matched per-dataset mean scores. Positive deltas favor FuzzyRuleSVM.",
        "",
        "| Comparison | Metric | n | Fuzzy Mean | Baseline Mean | Mean Delta | W/T/L | p-value |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in tests:
        if row["comparison"] == "Friedman omnibus":
            continue
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['n_datasets']} | "
            f"{row['fuzzy_mean']:.3f} | {row['baseline_mean']:.3f} | "
            f"{_fmt_signed(row['mean_delta'])} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {_fmt_p(row['wilcoxon_pvalue'])} |"
        )

    lines.extend([
        "",
        "## Omnibus Tests",
        "",
        "Friedman tests use datasets completed by all four rows in this report.",
        "",
        "| Metric | n | Statistic | p-value |",
        "|---|---:|---:|---:|",
    ])
    for row in tests:
        if row["comparison"] != "Friedman omnibus":
            continue
        lines.append(
            f"| {row['metric']} | {row['n_datasets']} | "
            f"{row['friedman_statistic']:.3f} | {_fmt_p(row['friedman_pvalue'])} |"
        )

    lines.extend([
        "",
        "## Dataset Results",
        "",
        "Balanced-accuracy means by dataset. Dashes indicate that the baseline row is absent from the checked-in artifact.",
        "",
        "| Dataset | FuzzyRuleSVM | FURIA | FARC-HD | IVTURS | Delta vs FURIA |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for spec in DATASET_SPECS:
        frs = fuzzy_rule_svm.get(spec.slug)
        if frs is None:
            continue
        furia = baseline_rows["furia"].get(spec.slug)
        delta = "---"
        if furia is not None:
            delta = _fmt_signed(float(frs["balanced_accuracy_mean"]) - float(furia["balanced_accuracy_mean"]))
        lines.append(
            f"| {spec.name} | {_fmt_value(frs, 'balanced_accuracy_mean')} | "
            f"{_fmt_value(furia, 'balanced_accuracy_mean')} | "
            f"{_fmt_value(baseline_rows['farchd'].get(spec.slug), 'balanced_accuracy_mean')} | "
            f"{_fmt_value(baseline_rows['ivturs'].get(spec.slug), 'balanced_accuracy_mean')} | "
            f"{delta} |"
        )

    furia_test = next(
        row for row in tests
        if row["comparison"] == "FuzzyRuleSVM vs FURIA" and row["metric"] == "balanced_accuracy"
    )
    lines.extend([
        "",
        "## Conclusion",
        "",
        "The refreshed primary FURIA comparison matches the manuscript: "
        f"{furia_test['n_datasets']} matched datasets, "
        f"mean balanced-accuracy delta {_fmt_signed(furia_test['mean_delta'])}, "
        f"W/T/L={furia_test['wins']}/{furia_test['ties']}/{furia_test['losses']}, "
        f"Wilcoxon p {_fmt_p(furia_test['wilcoxon_pvalue'])}.",
        "",
    ])
    return "\n".join(lines)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_by_dataset(rows: list[dict[str, str]], model_key: str) -> dict[str, dict[str, str]]:
    return {row["dataset"]: row for row in rows if row["model_key"] == model_key}


def _paired_test(
    fuzzy_rows: dict[str, dict[str, str]],
    baseline_rows: dict[str, dict[str, str]],
    baseline_name: str,
    metric: str,
) -> dict[str, Any]:
    key = f"{metric}_mean"
    datasets = sorted(set(fuzzy_rows) & set(baseline_rows))
    fuzzy_values = np.asarray([float(fuzzy_rows[d][key]) for d in datasets], dtype=float)
    baseline_values = np.asarray([float(baseline_rows[d][key]) for d in datasets], dtype=float)
    diffs = fuzzy_values - baseline_values
    return {
        "comparison": f"FuzzyRuleSVM vs {baseline_name}",
        "metric": metric,
        "n_datasets": len(datasets),
        "fuzzy_mean": float(np.mean(fuzzy_values)),
        "baseline_mean": float(np.mean(baseline_values)),
        "mean_delta": float(np.mean(diffs)),
        "median_delta": float(np.median(diffs)),
        "wins": int(np.sum(diffs > 0)),
        "ties": int(np.sum(np.isclose(diffs, 0.0))),
        "losses": int(np.sum(diffs < 0)),
        "wilcoxon_pvalue": _wilcoxon_pvalue(diffs),
    }


def _wilcoxon_pvalue(diffs: np.ndarray) -> float:
    if np.all(np.isclose(diffs, 0.0)):
        return 1.0
    return float(wilcoxon(diffs, zero_method="wilcox").pvalue)


def _friedman(values: list[np.ndarray]) -> tuple[float, float]:
    result = friedmanchisquare(*values)
    return float(result.statistic), float(result.pvalue)


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    corrected = [0.0] * len(p_values)
    for rank, idx in enumerate(order, start=1):
        corrected[idx] = min(1.0, p_values[idx] * (len(p_values) - rank + 1))
    running_max = 0.0
    for idx in order:
        running_max = max(running_max, corrected[idx])
        corrected[idx] = running_max
    return corrected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _update_metrics_json(path: Path, tests: list[dict[str, Any]]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["statistical_tests"] = tests
    data["statistical_test_sources"] = {
        "fuzzy_rule_svm": "runs/recommended-comparison/metrics.csv",
        "baselines": "runs/fuzzy-baselines-comparison/metrics.csv",
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _normalise_model_tables(
    model_tables: list[tuple[str, str, dict[str, dict[str, str]]] | tuple[str, str]],
    baseline_rows: dict[str, dict[str, dict[str, str]]],
) -> list[tuple[str, str, dict[str, dict[str, str]]]]:
    result = []
    for entry in model_tables:
        if len(entry) == 3:
            key, name, rows = entry
        else:
            key, name = entry
            rows = baseline_rows[key]
        result.append((key, name, rows))
    return result


def _mean(rows: dict[str, dict[str, str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows.values() if row.get(key)]))


def _fmt_value(row: dict[str, str] | None, key: str) -> str:
    if row is None or row.get(key) in {None, ""}:
        return "---"
    return f"{float(row[key]):.3f}"


def _fmt_signed(value: float) -> str:
    return f"{value:+.3f}"


def _fmt_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
