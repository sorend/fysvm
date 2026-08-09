"""Property Certification — Sound Interval-Arithmetic Property Certificates.

Evaluates all four certificate types across three biomedical benchmarks using
5-fold StratifiedKFold cross-validation.  Results saved to runs/prop_cert/results.json.

Usage:
    uv run python scripts/run_prop_cert.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

# Ensure src is on path when run directly (uv run already handles this via the
# editable install, but belt-and-suspenders for direct invocations).
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fysvm.certificates import (
    certificate_feature_exclusion,
    certificate_monotonicity,
    certificate_robustness,
    certificate_safe_region,
    check_validity_domain,
)
from fysvm.rule_svm import FuzzyRuleSVM
from fysvm.run_metadata import write_run_metadata

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS = ["pima_diabetes", "heart_cleveland", "mammographic_mass"]

DATASET_DIR = Path(__file__).parent.parent / "datasets" / "prepared"
OUTPUT_DIR = Path(__file__).parent.parent / "runs" / "prop_cert"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
RANDOM_STATE_BASE = 0

# FuzzyRuleSVM hyperparameters shared by primary and sensitivity fits.
CLF_PARAMS = dict(
    C=1.0,
    max_rules=256,
    min_rule_coverage=0.01,
    rule_length_penalty=0.35,
    class_weight="balanced",
    max_iter=20000,
    rule_generation="enumeration",
    softmin_temperature=0.1,
)

# The primary setting is one point in the 2x2 sensitivity design. It is fitted
# once and omitted from SENSITIVITY_CONFIGS to avoid duplicate work.
PRIMARY_CONFIG = {
    "name": "primary_l1_len2",
    "penalty": "l1",
    "max_rule_length": 2,
}
SENSITIVITY_CONFIGS = (
    {"name": "sensitivity_l1_len1", "penalty": "l1", "max_rule_length": 1},
    {"name": "sensitivity_l2_len1", "penalty": "l2", "max_rule_length": 1},
    {"name": "sensitivity_l2_len2", "penalty": "l2", "max_rule_length": 2},
)

# Fallback used only for datasets without a frozen tolerance design.
ROBUSTNESS_EPSILON_FRAC = 0.05

# Frozen annotations from docs/prop_cert_design.md Sections 4.5 and 6.3.
MONOTONE_FEATURE_ANNOTATIONS: dict[str, dict[str, list[str]]] = {
    "pima_diabetes": {
        "positive": ["glucose", "bmi", "age"],
        "negative": [],
    },
    "heart_cleveland": {
        "positive": ["age", "oldpeak"],
        "negative": ["max_heart_rate"],
    },
    "mammographic_mass": {
        "positive": ["age"],
        "negative": [],
    },
}

EXCLUSION_FEATURE_ANNOTATIONS: dict[str, list[str]] = {
    # Revised audit scope requested in peer review; this is not described as
    # preregistered because it differs from docs/prop_cert_design.md.
    "pima_diabetes": ["age"],
    "heart_cleveland": ["sex", "age"],
    "mammographic_mass": ["age"],
}

# Frozen engineering estimates from docs/prop_cert_design.md Section 5.3.
# Features not listed for these datasets have zero perturbation by design.
ROBUSTNESS_TOLERANCES: dict[str, dict[str, float]] = {
    "pima_diabetes": {
        "glucose": 5.0,
        "bmi": 0.5,
        "blood_pressure": 5.0,
        "age": 0.0,
    },
    "heart_cleveland": {
        "resting_blood_pressure": 5.0,
        "serum_cholesterol": 10.0,
        "age": 0.0,
    },
    "mammographic_mass": {
        "age": 0.0,
    },
}

# Pre-registered safe regions (from design doc Section 7.4)
SAFE_REGION_BOXES: dict[str, dict[str, tuple[float, float]]] = {
    "pima_diabetes": {
        "glucose": (60.0, 100.0),
        "bmi": (18.0, 25.0),
    },
    "heart_cleveland": {
        "age": (30.0, 45.0),
        "oldpeak": (0.0, 0.5),
    },
    "mammographic_mass": {
        "age": (20.0, 35.0),
    },
}

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a prepared dataset and return (X, y, feature_names)."""
    path = DATASET_DIR / f"{name}.npz"
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"]
    feature_names = list(data["feature_names"])
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Feature name resolution
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def _match_feature(annotation_name: str, feature_names: list[str]) -> int | None:
    """Find index in feature_names that matches annotation_name (case-insensitive substring).

    Requires at least 5 chars in both strings for substring match.
    Short strings require exact match to avoid false positives (e.g., "thal" ≠ "thalach").
    """
    query = _normalize(annotation_name)
    for i, fn in enumerate(feature_names):
        target = _normalize(fn)
        if len(query) >= 5 and len(target) >= 5:
            if query in target or target in query:
                return i
        else:
            if query == target:
                return i
    return None


def _resolve_feature_indices(
    clf: FuzzyRuleSVM,
    annotation_names: list[str],
    feature_names: list[str],
) -> list[tuple[str, int | None]]:
    """Map annotation feature names to internal model feature indices.

    Returns list of (annotation_name, internal_index or None).
    """
    result: list[tuple[str, int | None]] = []
    for ann_name in annotation_names:
        # First find the original dataset feature index
        orig_idx = _match_feature(ann_name, feature_names)
        if orig_idx is None:
            result.append((ann_name, None))
            continue
        # Map original index → internal model index via selected_feature_indices_
        # (clf.selected_feature_indices_ maps internal → original)
        internal_idx = None
        for internal_i, orig_i in enumerate(clf.selected_feature_indices_):
            if orig_i == orig_idx:
                internal_idx = internal_i
                break
        result.append((ann_name, internal_idx))
    return result


def _robustness_tolerances(
    clf: FuzzyRuleSVM,
    X_train: np.ndarray,
    dataset_name: str,
    feature_names: list[str],
) -> tuple[np.ndarray, dict]:
    """Return internal-feature tolerances and their provenance."""
    n_internal = len(clf.partitions_)
    eps_arr = np.zeros(n_internal)
    by_feature: dict[str, dict] = {}

    if dataset_name in ROBUSTNESS_TOLERANCES:
        configured = {
            _normalize(name): value
            for name, value in ROBUSTNESS_TOLERANCES[dataset_name].items()
        }
        for j, orig_j in enumerate(clf.selected_feature_indices_):
            feature_name = str(feature_names[orig_j])
            normalized_name = _normalize(feature_name)
            eps_arr[j] = configured.get(normalized_name, 0.0)
            by_feature[feature_name] = {
                "epsilon": float(eps_arr[j]),
                "source": (
                    "frozen_design_engineering_estimate"
                    if normalized_name in configured and eps_arr[j] > 0.0
                    else "frozen_design_no_perturbation"
                ),
            }
        metadata = {
            "scheme": "frozen_design_feature_specific",
            "source": "docs/prop_cert_design.md Section 5.3",
            "clinically_validated": False,
            "unspecified_feature_epsilon": 0.0,
            "by_feature": by_feature,
        }
        return eps_arr, metadata

    for j, orig_j in enumerate(clf.selected_feature_indices_):
        feature_name = str(feature_names[orig_j])
        feature_col = X_train[:, orig_j]
        q25 = float(np.percentile(feature_col, 25))
        q75 = float(np.percentile(feature_col, 75))
        feature_scale = q75 - q25
        if feature_scale <= 0.0:
            feature_scale = float(np.max(feature_col) - np.min(feature_col))
        eps_arr[j] = ROBUSTNESS_EPSILON_FRAC * feature_scale
        by_feature[feature_name] = {
            "epsilon": float(eps_arr[j]),
            "source": "fold_local_empirical_iqr_fraction",
        }
    metadata = {
        "scheme": "fold_local_empirical_iqr_fraction",
        "fraction": ROBUSTNESS_EPSILON_FRAC,
        "source": "data-scaled perturbation; not an engineering tolerance",
        "clinically_validated": False,
        "by_feature": by_feature,
    }
    return eps_arr, metadata


def _complete_safe_region_box(
    clf: FuzzyRuleSVM,
    X_train: np.ndarray,
    dataset_name: str,
    feature_names: list[str],
) -> tuple[dict[int, tuple[float, float]], dict, dict, list[str]]:
    """Complete a declared safe box with fold-local ranges by exact normalized name."""
    declared = SAFE_REGION_BOXES.get(dataset_name, {})
    declared_normalized = {_normalize(name): (name, bounds) for name, bounds in declared.items()}
    declared_box = {name: [float(lo), float(hi)] for name, (lo, hi) in declared.items()}
    completed_box: dict[str, dict] = {}
    safe_box: dict[int, tuple[float, float]] = {}
    matched_declared: set[str] = set()

    for internal_j, orig_j in enumerate(clf.selected_feature_indices_):
        feature_name = str(feature_names[orig_j])
        normalized_name = _normalize(feature_name)
        if normalized_name in declared_normalized:
            declared_name, bounds = declared_normalized[normalized_name]
            lo, hi = float(bounds[0]), float(bounds[1])
            source = "declared"
            matched_declared.add(declared_name)
        else:
            feature_col = X_train[:, orig_j]
            lo, hi = float(np.min(feature_col)), float(np.max(feature_col))
            source = "fold_local_train_min_max"
        safe_box[internal_j] = (lo, hi)
        completed_box[feature_name] = {
            "bounds": [lo, hi],
            "source": source,
            "internal_index": int(internal_j),
        }

    unmatched = sorted(set(declared) - matched_declared)
    return safe_box, declared_box, completed_box, unmatched


# ---------------------------------------------------------------------------
# Per-fold certificate runner
# ---------------------------------------------------------------------------

def run_certificates_for_fold(
    clf: FuzzyRuleSVM,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_name: str,
    fold: int,
    and_operator: str,
    feature_names: list[str],
    configuration: dict,
) -> dict:
    """Run all 4 certificate types for one fold. Returns a result dict."""
    valid_domain = check_validity_domain(clf)
    n_rules = int(len(clf.rules_))
    n_active_rules = int(len(clf.active_rule_indices_))
    rule_length_counts: dict[str, int] = {}
    active_rule_length_counts: dict[str, int] = {}
    for rule in clf.rules_:
        key = str(rule.length)
        rule_length_counts[key] = rule_length_counts.get(key, 0) + 1
    for rule_idx in clf.active_rule_indices_:
        key = str(clf.rules_[rule_idx].length)
        active_rule_length_counts[key] = active_rule_length_counts.get(key, 0) + 1

    fold_results: dict = {
        "fold": fold,
        "and_operator": and_operator,
        "configuration": dict(configuration),
        "validity_domain": valid_domain,
        "held_out_balanced_accuracy": float(
            balanced_accuracy_score(y_test, clf.predict(X_test))
        ),
        "n_rules": n_rules,
        "n_active_rules": n_active_rules,
        "active_rule_fraction": n_active_rules / max(n_rules, 1),
        "coefficient_l1_norm": float(np.linalg.norm(clf.coef_, ord=1)),
        "coefficient_l2_norm": float(np.linalg.norm(clf.coef_, ord=2)),
        "rule_length_counts": rule_length_counts,
        "active_rule_length_counts": active_rule_length_counts,
        "monotonicity": [],
        "robustness": {},
        "exclusion": None,
        "safe_region": None,
    }

    # ------------------------------------------------------------------
    # Certificate 1: Monotonicity
    # ------------------------------------------------------------------
    ann = MONOTONE_FEATURE_ANNOTATIONS.get(dataset_name, {"positive": [], "negative": []})
    for direction in ("positive", "negative"):
        for feat_name in ann.get(direction, []):
            resolved = _resolve_feature_indices(clf, [feat_name], feature_names)
            _, internal_idx = resolved[0]
            if internal_idx is None:
                fold_results["monotonicity"].append({
                    "feature_name": feat_name,
                    "direction": direction,
                    "status": "NO-MATCH",
                    "min_slack": None,
                    "internal_index": None,
                })
                continue
            res = certificate_monotonicity(clf, feature_index=internal_idx, direction=direction)
            fold_results["monotonicity"].append({
                "feature_name": feat_name,
                "direction": direction,
                "status": res.status,
                "min_slack": None if np.isnan(res.min_slack) else float(res.min_slack),
                "internal_index": int(internal_idx),
                "counterexample": res.counterexample,
            })

    # ------------------------------------------------------------------
    # Certificate 2: Robustness (all test samples)
    # ------------------------------------------------------------------
    eps_arr, tolerance_metadata = _robustness_tolerances(
        clf, X_train, dataset_name, feature_names
    )

    rob_counts: dict[str, int] = {"CERTIFIED": 0, "UNKNOWN": 0, "COUNTEREXAMPLE": 0}
    rob_min_slacks: list[float] = []

    for x_raw in X_test:
        # Project to internal features (consistent with clf.transform)
        x_internal = x_raw[clf.selected_feature_indices_]
        res = certificate_robustness(clf, x_internal, epsilon=eps_arr)
        rob_counts[res.status] = rob_counts.get(res.status, 0) + 1
        if not np.isnan(res.min_slack):
            rob_min_slacks.append(res.min_slack)

    n_test = len(X_test)
    fold_results["robustness"] = {
        "n_test": n_test,
        "certified": rob_counts.get("CERTIFIED", 0),
        "unknown": rob_counts.get("UNKNOWN", 0),
        "counterexample": rob_counts.get("COUNTEREXAMPLE", 0),
        "certified_rate": rob_counts.get("CERTIFIED", 0) / max(n_test, 1),
        "mean_min_slack": float(np.mean(rob_min_slacks)) if rob_min_slacks else float("nan"),
        "status_counts": rob_counts,
        "tolerances": tolerance_metadata,
    }

    # ------------------------------------------------------------------
    # Certificate 3: Feature exclusion
    # ------------------------------------------------------------------
    excl_names = EXCLUSION_FEATURE_ANNOTATIONS.get(dataset_name, [])
    excl_resolved = _resolve_feature_indices(clf, excl_names, feature_names)
    excl_internal_indices = [idx for _, idx in excl_resolved if idx is not None]
    excl_matched = [(name, idx) for name, idx in excl_resolved]
    excl_unmatched = [name for name, idx in excl_resolved if idx is None]

    if excl_internal_indices and not excl_unmatched:
        res = certificate_feature_exclusion(clf, feature_indices=excl_internal_indices)
        fold_results["exclusion"] = {
            "status": res.status,
            "min_slack": None if np.isnan(res.min_slack) else float(res.min_slack),
            "features_checked": [(n, i) for n, i in excl_matched],
            "unmatched_declared_features": [],
            "counterexample": res.counterexample,
        }
    else:
        fold_results["exclusion"] = {
            "status": "NO-MATCH",
            "features_checked": [(n, None) for n, _ in excl_resolved],
            "unmatched_declared_features": excl_unmatched,
        }

    # ------------------------------------------------------------------
    # Certificate 4: Pre-registered safe region, completed to all model features
    # ------------------------------------------------------------------
    safe_box, declared_box, completed_box, unmatched_declared = _complete_safe_region_box(
        clf, X_train, dataset_name, feature_names
    )
    if unmatched_declared:
        fold_results["safe_region"] = {
            "status": "NO-MATCH",
            "min_slack": None,
            "target_class": str(clf.classes_[0]),
            "declared_box": declared_box,
            "completed_box": completed_box,
            "unmatched_declared_features": unmatched_declared,
            "matching": "exact_normalized_feature_name",
        }
    else:
        res = certificate_safe_region(clf, box=safe_box, target_class_index=0)
        fold_results["safe_region"] = {
            "status": res.status,
            "lb_f": res.details.get("lb_f"),
            "ub_f": res.details.get("ub_f"),
            "min_slack": None if np.isnan(res.min_slack) else float(res.min_slack),
            "target_class": str(clf.classes_[0]),
            "n_box_features": len(safe_box),
            "declared_box": declared_box,
            "completed_box": completed_box,
            "unmatched_declared_features": [],
            "matching": "exact_normalized_feature_name",
        }

    mono_total = len(fold_results["monotonicity"])
    mono_certified = sum(
        item["status"] in ("CERTIFIED", "CERTIFIED-TRIVIAL")
        for item in fold_results["monotonicity"]
    )
    fold_results["certificate_rates"] = {
        "monotonicity": mono_certified / max(mono_total, 1),
        "robustness": fold_results["robustness"]["certified_rate"],
        "exclusion": float(fold_results["exclusion"]["status"] == "CERTIFIED"),
        "safe_region": float(fold_results["safe_region"]["status"] == "CERTIFIED"),
    }

    return fold_results


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def _status_counts(statuses: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts


def _build_summary(
    all_results: dict,
    configuration_name: str,
    *,
    include_configuration_in_key: bool = True,
) -> dict:
    summary: dict = {}
    for dataset_name, dataset_results in all_results.items():
        for and_op in ("min", "product"):
            op_results = [
                result
                for result in dataset_results
                if result["and_operator"] == and_op
                and result["configuration"]["name"] == configuration_name
            ]
            if not op_results:
                continue

            mono_all = [item for result in op_results for item in result["monotonicity"]]
            mono_statuses = [item["status"] for item in mono_all]
            mono_certified = sum(
                status in ("CERTIFIED", "CERTIFIED-TRIVIAL")
                for status in mono_statuses
            )

            rob_statuses = [
                status
                for result in op_results
                for status, count in result["robustness"]["status_counts"].items()
                for _ in range(count)
            ]
            rob_certified = sum(status == "CERTIFIED" for status in rob_statuses)

            exclusion_statuses = [result["exclusion"]["status"] for result in op_results]
            exclusion_certified = sum(status == "CERTIFIED" for status in exclusion_statuses)
            safe_statuses = [result["safe_region"]["status"] for result in op_results]
            safe_certified = sum(status == "CERTIFIED" for status in safe_statuses)

            key = (
                f"{configuration_name}_{dataset_name}_{and_op}"
                if include_configuration_in_key
                else f"{dataset_name}_{and_op}"
            )
            summary[key] = {
                "configuration": configuration_name,
                "dataset": dataset_name,
                "and_operator": and_op,
                "mean_held_out_balanced_accuracy": float(
                    np.mean([result["held_out_balanced_accuracy"] for result in op_results])
                ),
                "monotonicity": {
                    "n_total": len(mono_statuses),
                    "n_valid": sum(
                        status not in ("NO-MATCH", "UNKNOWN") for status in mono_statuses
                    ),
                    "n_certified": mono_certified,
                    "certified_rate": mono_certified / max(len(mono_statuses), 1),
                    "status_counts": _status_counts(mono_statuses),
                },
                "robustness": {
                    "n_total": len(rob_statuses),
                    "n_certified": rob_certified,
                    "mean_certified_rate": rob_certified / max(len(rob_statuses), 1),
                    "status_counts": _status_counts(rob_statuses),
                },
                "exclusion": {
                    "n_folds": len(exclusion_statuses),
                    "n_certified": exclusion_certified,
                    "certified_rate": exclusion_certified / max(len(exclusion_statuses), 1),
                    "status_counts": _status_counts(exclusion_statuses),
                },
                "safe_region": {
                    "n_folds": len(safe_statuses),
                    "n_certified": safe_certified,
                    "certified_rate": safe_certified / max(len(safe_statuses), 1),
                    "status_counts": _status_counts(safe_statuses),
                },
            }
    return summary


def run_evaluation(
    primary_config: dict = PRIMARY_CONFIG,
    sensitivity_configs: tuple[dict, ...] = SENSITIVITY_CONFIGS,
) -> dict:
    primary_results: dict[str, list[dict]] = {}
    sensitivity_results: dict[str, list[dict]] = {}
    configurations = (primary_config, *sensitivity_configs)

    for dataset_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        X_raw, y, feature_names = load_dataset(dataset_name)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE_BASE)
        dataset_primary_results: list[dict] = []
        dataset_sensitivity_results: list[dict] = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, y)):
            print(f"  Fold {fold_idx} ...", end=" ", flush=True)
            t0 = time.time()

            X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            X_train = imputer.fit_transform(X_train_raw)
            X_test = imputer.transform(X_test_raw)
            if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(X_test)):
                raise ValueError(
                    "Non-finite values remain after fold-local imputation: "
                    f"{dataset_name} fold {fold_idx}"
                )

            for configuration in configurations:
                target = (
                    dataset_primary_results
                    if configuration["name"] == primary_config["name"]
                    else dataset_sensitivity_results
                )
                clf_params = {
                    **CLF_PARAMS,
                    "penalty": configuration["penalty"],
                    "max_rule_length": configuration["max_rule_length"],
                }
                for and_operator in ("min", "product"):
                    clf = FuzzyRuleSVM(
                        **clf_params,
                        and_operator=and_operator,
                        random_state=RANDOM_STATE_BASE + fold_idx,
                        feature_names=feature_names,
                    )  # type: ignore[arg-type]
                    clf.fit(X_train, y_train)

                    target.append(
                        run_certificates_for_fold(
                            clf=clf,
                            X_train=X_train,
                            X_test=X_test,
                            y_test=y_test,
                            dataset_name=dataset_name,
                            fold=fold_idx,
                            and_operator=and_operator,
                            feature_names=feature_names,
                            configuration=configuration,
                        )
                    )

            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")

        primary_results[dataset_name] = dataset_primary_results
        sensitivity_results[dataset_name] = dataset_sensitivity_results

    primary_summary = _build_summary(
        primary_results,
        primary_config["name"],
        include_configuration_in_key=False,
    )
    sensitivity_summary: dict = {}
    for configuration in sensitivity_configs:
        sensitivity_summary.update(
            _build_summary(sensitivity_results, configuration["name"])
        )

    return {
        "metadata": {
            "preprocessing": "fold-local median imputation fitted on training data only",
            "n_folds": N_FOLDS,
            "operators": ["min", "product"],
            "classifier_shared_params": CLF_PARAMS,
            "primary_configuration": primary_config,
            "sensitivity_configurations": list(sensitivity_configs),
            "sensitivity_grid": {
                "penalty": ["l1", "l2"],
                "max_rule_length": [1, 2],
                "primary_grid_point_reused": True,
            },
        },
        "results": primary_results,
        "summary": primary_summary,
        "sensitivity": {
            "results": sensitivity_results,
            "summary": sensitivity_summary,
        },
    }


def _print_summary_section(title: str, summary: dict) -> None:
    print(f"\n--- {title} ---")
    header = f"{'Config+Dataset+Op':<60} {'BalAcc':>8} {'Mono%':>8} {'Rob%':>8} {'Excl%':>8} {'Safe%':>8}"
    print(header)
    print("-" * len(header))
    for s in summary.values():
        mono_pct = 100.0 * s["monotonicity"]["certified_rate"]
        rob_pct = 100.0 * s["robustness"]["mean_certified_rate"]
        excl_pct = 100.0 * s["exclusion"]["certified_rate"]
        safe_pct = 100.0 * s["safe_region"]["certified_rate"]
        accuracy = s["mean_held_out_balanced_accuracy"]
        label = f"{s['configuration']}:{s['dataset']}({s['and_operator']})"
        print(
            f"{label:<60} {accuracy:>8.3f} {mono_pct:>7.1f}% {rob_pct:>7.1f}% "
            f"{excl_pct:>7.1f}% {safe_pct:>7.1f}%"
        )

    print("Status counts (NO-MATCH and UNKNOWN remain in denominators):")
    for s in summary.values():
        label = f"{s['configuration']}:{s['dataset']}({s['and_operator']})"
        print(
            f"  {label}: mono={s['monotonicity']['status_counts']} "
            f"rob={s['robustness']['status_counts']} "
            f"excl={s['exclusion']['status_counts']} "
            f"safe={s['safe_region']['status_counts']}"
        )


def print_summary(output: dict) -> None:
    summary = output["summary"]
    print("\n" + "=" * 100)
    print("PROPERTY CERTIFICATION SUMMARY - CERTIFIED RATES PER CERTIFICATE TYPE")
    print("=" * 100)
    _print_summary_section("PRIMARY CONFIGURATION", summary)
    _print_summary_section("SENSITIVITY CONFIGURATIONS", output["sensitivity"]["summary"])

    print("\n--- Hypothesis evaluation vs. targets ---")
    for s in summary.values():
        if s["and_operator"] != "min":
            continue
        ds = s["dataset"]
        mono_r = s["monotonicity"]["certified_rate"]
        rob_r = s["robustness"]["mean_certified_rate"]
        excl_r = s["exclusion"]["certified_rate"]
        safe_n = s["safe_region"]["n_certified"]

        h11 = "SUPPORT" if mono_r >= 0.80 else ("INCONCLUSIVE" if mono_r >= 0.50 else "FAIL")
        h12 = "SUPPORT" if excl_r >= 0.60 else ("INCONCLUSIVE" if excl_r >= 0.30 else "FAIL")
        h13 = "SUPPORT" if rob_r >= 0.70 else ("INCONCLUSIVE" if rob_r >= 0.40 else "FAIL")
        h14_note = f"(H1.4 for pima: {safe_n}/5 folds CERTIFIED)" if ds == "pima_diabetes" else ""

        print(f"  {ds} | H1.1={h11}({mono_r:.0%}) H1.2={h12}({excl_r:.0%}) H1.3={h13}({rob_r:.0%}) {h14_note}")

    print("=" * 100)


def _json_safe(value):
    """Convert NumPy scalars and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    print("Property Certification — Sound Interval-Arithmetic Property Certificates")
    print(f"Datasets: {DATASETS}")
    print(f"Folds: {N_FOLDS}, Operators: min + product")

    output = run_evaluation()

    # Save JSON results
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(_json_safe(output), f, indent=2, allow_nan=False)
    write_run_metadata(
        OUTPUT_DIR,
        command=["uv", "run", "python", *sys.argv],
        config={
            "datasets": DATASETS,
            "n_folds": N_FOLDS,
            "primary_config": PRIMARY_CONFIG,
            "sensitivity_configs": SENSITIVITY_CONFIGS,
            "classifier": CLF_PARAMS,
            "exclusion_scope": EXCLUSION_FEATURE_ANNOTATIONS,
            "exclusion_scope_note": "Revised after peer review; not preregistered.",
        },
    )
    print(f"\nResults saved to {out_path}")

    print_summary(output)


if __name__ == "__main__":
    main()
