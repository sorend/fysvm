# ============================================================
# Specification Fidelity — Paper Section 4
# Implements: independent reference specification, SHA-256 hash,
# and metamorphic verification (MR1–MR6).
# ============================================================
"""Conformance checking: reference spec vs production FuzzyRuleSVM.

This module implements:

- ``ConformanceResult`` — structured result of a reference/production comparison.
- ``check_conformance(clf, X, dataset_name, tolerance)`` — compare reference
  implementations (tests/reference_implementations.py) against production for
  membership values, rule activations, and the decision function.
- ``MetamorphicResult`` — structured result of one metamorphic relation check.
- ``check_metamorphic_relations(clf, X, y, random_state)`` — run six
  metamorphic relation categories.
- ``run_conformance_suite(clf, datasets, output_dir)`` — batch run over
  multiple datasets and serialise results.

Result semantics
----------------
``status`` reports only measured agreement with the executable specification:
CERTIFIED if ``max_abs_error < tolerance``, otherwise COUNTEREXAMPLE.

``certificate_eligibility_status`` separately reports whether the selected
operator is covered by the property-certificate theory: min and product are
ELIGIBLE, softmin is INELIGIBLE, and unrecognised operators are UNKNOWN.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone


# ---------------------------------------------------------------------------
# Reference module loader
# ---------------------------------------------------------------------------

def _load_reference_module():
    """Load tests/reference_implementations.py, trying multiple paths."""
    if "reference_implementations" in sys.modules:
        return sys.modules["reference_implementations"]

    try:
        import reference_implementations  # works when tests/ is in sys.path
        return reference_implementations
    except ImportError:
        pass

    # Path-based fallback: locate relative to this file (src/fysvm/conformance.py)
    candidates = [
        Path(__file__).parent.parent.parent / "tests" / "reference_implementations.py",
        Path.cwd() / "tests" / "reference_implementations.py",
    ]
    for ref_path in candidates:
        if ref_path.exists():
            spec = importlib.util.spec_from_file_location(
                "reference_implementations", ref_path
            )
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            sys.modules["reference_implementations"] = mod
            return mod

    raise ImportError(
        "reference_implementations.py not found. "
        "Ensure tests/ is in sys.path or run from the project root."
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConformanceResult:
    """Structured result of a reference/production conformance check."""

    status: str
    """Measured conformance: 'CERTIFIED' or 'COUNTEREXAMPLE'."""

    certificate_eligibility_status: str
    """Property-certificate eligibility: 'ELIGIBLE', 'INELIGIBLE', or 'UNKNOWN'."""

    max_abs_error: float
    """Maximum absolute error across memberships, activations, and decision function."""

    mean_abs_error: float
    """Mean absolute error across the three comparison components."""

    n_samples: int
    """Number of samples evaluated."""

    dataset_name: str
    """Identifier of the dataset used."""

    and_operator: str
    """T-norm used by the classifier ('min', 'product', or 'softmin')."""

    details: dict[str, Any] = field(default_factory=dict)
    """Breakdown by component and optional counterexample witness."""


@dataclass
class MetamorphicResult:
    """Structured result of one metamorphic relation evaluation."""

    relation_name: str
    """Short name, e.g. 'MR1_row_permutation'."""

    passed: bool
    """True if no violations were found."""

    n_violations: int
    """Number of individual invariant violations."""

    max_violation: float
    """Largest observed violation magnitude (0.0 if passed)."""

    details: dict[str, Any] = field(default_factory=dict)
    """Auxiliary information: inputs, expected, actual, etc."""


# ---------------------------------------------------------------------------
# Conformance check
# ---------------------------------------------------------------------------

def check_conformance(
    clf,
    X: np.ndarray,
    dataset_name: str,
    tolerance: float = 1e-10,
) -> ConformanceResult:
    """Compare reference spec against production clf on dataset X.

    Checks three levels:
    1. Membership values  (n, d, 3) tensor
    2. Rule activations   (n, K) matrix
    3. Decision function  (n,) vector

    Parameters
    ----------
    clf : fitted SparseMaxMarginFuzzyRuleMachine
        Must be fitted.
    X : np.ndarray, shape (n, n_features_in)
        Input samples (full feature set; function projects internally).
    dataset_name : str
        Label for the result record.
    tolerance : float
        Absolute tolerance for measured CERTIFIED status.

    Returns
    -------
    ConformanceResult
    """
    ref = _load_reference_module()

    X = np.asarray(X, dtype=np.float64)

    # Project to the screened feature subset (same as production transform)
    X_model = X[:, clf.selected_feature_indices_]

    # ------------------------------------------------------------------
    # 1. Membership conformance
    # ------------------------------------------------------------------
    ref_memberships = ref.compute_membership_matrix_ref(X_model, clf.partitions_)
    prod_memberships = clf._concept_membership_tensor(X_model)

    membership_errors = np.abs(ref_memberships - prod_memberships)
    max_membership_error = float(np.max(membership_errors))
    mean_membership_error = float(np.mean(membership_errors))

    # ------------------------------------------------------------------
    # 2. Activation conformance
    # ------------------------------------------------------------------
    ref_Z = ref.compute_activation_matrix_ref(
        ref_memberships,
        clf.rules_,
        and_operator=clf.and_operator,
        temperature=clf.softmin_temperature,
    )
    prod_Z = clf.transform(X)

    activation_errors = np.abs(ref_Z - prod_Z)
    max_activation_error = float(np.max(activation_errors))
    mean_activation_error = float(np.mean(activation_errors))

    # ------------------------------------------------------------------
    # 3. Decision function conformance
    # ------------------------------------------------------------------
    ref_df = ref.decision_function_ref(ref_Z, clf.coef_, clf.intercept_)
    prod_df = clf.decision_function(X)

    df_errors = np.abs(ref_df - prod_df)
    max_df_error = float(np.max(df_errors))
    mean_df_error = float(np.mean(df_errors))

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    max_abs_error = max(max_membership_error, max_activation_error, max_df_error)
    mean_abs_error = (mean_membership_error + mean_activation_error + mean_df_error) / 3.0

    details: dict[str, Any] = {
        "max_membership_error": max_membership_error,
        "max_activation_error": max_activation_error,
        "max_decision_function_error": max_df_error,
        "mean_membership_error": mean_membership_error,
        "mean_activation_error": mean_activation_error,
        "mean_decision_function_error": mean_df_error,
        "tolerance": tolerance,
    }

    # ------------------------------------------------------------------
    # Measured conformance and independent property-certificate eligibility
    # ------------------------------------------------------------------
    certificate_eligibility_status = {
        "min": "ELIGIBLE",
        "product": "ELIGIBLE",
        "softmin": "INELIGIBLE",
    }.get(clf.and_operator, "UNKNOWN")

    if max_abs_error < tolerance:
        status = "CERTIFIED"
    else:
        component_errors = {
            "membership": membership_errors,
            "activation": activation_errors,
            "decision_function": df_errors,
        }
        component = max(
            component_errors,
            key=lambda name: float(np.max(component_errors[name])),
        )
        worst_index = np.unravel_index(
            int(np.argmax(component_errors[component])),
            component_errors[component].shape,
        )
        sample_index = int(worst_index[0])
        counterexample: dict[str, Any] = {
            "component": component,
            "sample_index": sample_index,
            "X_sample": X[sample_index].tolist(),
            "absolute_error": float(component_errors[component][worst_index]),
        }
        if component == "membership":
            _, feature_index, term_index = worst_index
            counterexample.update({
                "feature_index": int(feature_index),
                "term_index": int(term_index),
                "reference_value": float(ref_memberships[worst_index]),
                "production_value": float(prod_memberships[worst_index]),
            })
        elif component == "activation":
            _, rule_index = worst_index
            counterexample.update({
                "rule_index": int(rule_index),
                "reference_value": float(ref_Z[worst_index]),
                "production_value": float(prod_Z[worst_index]),
            })
        else:
            counterexample.update({
                "reference_value": float(ref_df[worst_index]),
                "production_value": float(prod_df[worst_index]),
            })
        details["counterexample"] = counterexample
        status = "COUNTEREXAMPLE"

    return ConformanceResult(
        status=status,
        certificate_eligibility_status=certificate_eligibility_status,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        n_samples=int(len(X)),
        dataset_name=dataset_name,
        and_operator=clf.and_operator,
        details=details,
    )


# ---------------------------------------------------------------------------
# Metamorphic relations
# ---------------------------------------------------------------------------

def check_metamorphic_relations(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> list[MetamorphicResult]:
    """Run six metamorphic relation checks on a fitted clf.

    Parameters
    ----------
    clf : fitted SparseMaxMarginFuzzyRuleMachine
    X : np.ndarray, shape (n, n_features_in)
    y : np.ndarray, shape (n,)
    random_state : int

    Returns
    -------
    list[MetamorphicResult]  — six results in MR1..MR6 order.
    """
    results: list[MetamorphicResult] = []
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # MR1 — Row-permutation invariance
    # predict(X[π]) == predict(X)[π]  and  transform(X[π]) == transform(X)[π]
    # ------------------------------------------------------------------
    perm = rng.permutation(len(X))
    pred_permuted = clf.predict(X[perm])
    pred_original_permuted = clf.predict(X)[perm]

    df_perm = clf.decision_function(X[perm])
    df_orig_perm = clf.decision_function(X)[perm]
    df_perm_errors = np.abs(df_perm - df_orig_perm)
    max_df_perm_error = float(np.max(df_perm_errors))

    label_mismatches = int(np.sum(pred_permuted != pred_original_permuted))
    mr1_passed = label_mismatches == 0 and max_df_perm_error <= 1e-12

    results.append(MetamorphicResult(
        relation_name="MR1_row_permutation",
        passed=mr1_passed,
        n_violations=label_mismatches,
        max_violation=max_df_perm_error,
        details={
            "n_samples": len(X),
            "n_label_mismatches": label_mismatches,
            "max_df_error": max_df_perm_error,
            "tolerance": 1e-12,
        },
    ))

    # ------------------------------------------------------------------
    # MR2 — Partition determinism
    # Fitting two clones on same data yields identical partition breakpoints.
    # ------------------------------------------------------------------
    clf2 = clone(clf)
    clf2.fit(X, y)

    n_partition_violations = 0
    max_partition_error = 0.0
    for p1, p2 in zip(clf.partitions_, clf2.partitions_):
        for attr in ("low", "medium", "high"):
            err = abs(getattr(p1, attr) - getattr(p2, attr))
            max_partition_error = max(max_partition_error, err)
            if err > 0.0:
                n_partition_violations += 1

    mr2_passed = n_partition_violations == 0

    results.append(MetamorphicResult(
        relation_name="MR2_partition_determinism",
        passed=mr2_passed,
        n_violations=n_partition_violations,
        max_violation=max_partition_error,
        details={
            "n_partitions_checked": len(clf.partitions_) * 3,
            "description": (
                "Fitting two clones with same settings on same data should "
                "produce identical partition breakpoints."
            ),
        },
    ))

    # ------------------------------------------------------------------
    # MR3 — Decision function monotonicity with activation
    # For each active rule k, contributions = coef_k * activation_k.
    # Since contribution is linear in activation, sorting by activation
    # should yield sorted contributions (ascending if coef>0, descending if coef<0).
    # ------------------------------------------------------------------
    Z = clf.transform(X)
    n_monotone_violations = 0
    max_monotone_error = 0.0
    tol_mono = 1e-12

    for k in clf.active_rule_indices_:
        activation_k = Z[:, k]
        contribution_k = clf.coef_[k] * activation_k
        sorted_idx = np.argsort(activation_k)
        sorted_contribs = contribution_k[sorted_idx]
        diffs = np.diff(sorted_contribs)
        if clf.coef_[k] > 0:
            violations = diffs[diffs < -tol_mono]
        else:
            violations = diffs[diffs > tol_mono]
        n_monotone_violations += len(violations)
        if len(violations) > 0:
            max_monotone_error = max(max_monotone_error, float(np.max(np.abs(violations))))

    mr3_passed = n_monotone_violations == 0

    results.append(MetamorphicResult(
        relation_name="MR3_contribution_monotonicity",
        passed=mr3_passed,
        n_violations=n_monotone_violations,
        max_violation=max_monotone_error,
        details={
            "n_active_rules_checked": int(len(clf.active_rule_indices_)),
            "tolerance": tol_mono,
            "description": (
                "contribution_k = coef_k * activation_k; sorting by activation "
                "should yield monotonically sorted contributions."
            ),
        },
    ))

    # ------------------------------------------------------------------
    # MR4 — Membership boundary conditions
    # At each partition anchor (q_low, q_mid, q_high) for each feature,
    # membership values must match the independent reference implementation.
    # This retains the standard one-hot patterns when anchors are distinct and
    # handles coincident anchors according to the specified step semantics.
    # Tests production clf._concept_membership_tensor.
    # ------------------------------------------------------------------
    n_boundary_violations = 0
    max_boundary_error = 0.0
    tol_boundary = 1e-12

    n_screened = len(clf.partitions_)
    ref = _load_reference_module()

    for j, partition in enumerate(clf.partitions_):
        # Build a test matrix: 3 rows, n_screened features
        X_test = np.zeros((3, n_screened), dtype=np.float64)
        X_test[0, j] = partition.low    # q_low for feature j
        X_test[1, j] = partition.medium  # q_mid
        X_test[2, j] = partition.high   # q_high

        expected_memberships = ref.compute_membership_matrix_ref(
            X_test, clf.partitions_
        )
        memberships = clf._concept_membership_tensor(X_test)  # (3, n_screened, 3)

        for row_idx in range(3):
            exp_vals = expected_memberships[row_idx, j, :].tolist()
            act_vals = memberships[row_idx, j, :].tolist()
            for term_idx, (exp_v, act_v) in enumerate(zip(exp_vals, act_vals)):
                err = abs(exp_v - act_v)
                max_boundary_error = max(max_boundary_error, err)
                if err > tol_boundary:
                    n_boundary_violations += 1

    mr4_passed = n_boundary_violations == 0

    results.append(MetamorphicResult(
        relation_name="MR4_membership_boundaries",
        passed=mr4_passed,
        n_violations=n_boundary_violations,
        max_violation=max_boundary_error,
        details={
            "n_checks": len(clf.partitions_) * 3 * 3,
            "tolerance": tol_boundary,
            "description": (
                "Memberships at each actual partition anchor match the independent "
                "reference; distinct anchors retain the standard low/medium/high "
                "one-hot patterns."
            ),
        },
    ))

    # ------------------------------------------------------------------
    # MR5 — Explanation additivity
    # sum(contribution) == net_rule_contribution  and
    # net_rule_contribution + bias == margin
    # ------------------------------------------------------------------
    explanations = clf.explain(X, top_n=clf.n_rules_, min_abs_contribution=0.0)
    n_additivity_violations = 0
    max_additivity_error = 0.0
    tol_additivity = 1e-10

    for expl in explanations:
        sum_contribs = sum(item["contribution"] for item in expl["top_rules"])
        net = expl["net_rule_contribution"]
        margin = expl["margin"]
        bias = expl["bias"]

        err1 = abs(sum_contribs - net)
        err2 = abs(net + bias - margin)
        max_err = max(err1, err2)
        max_additivity_error = max(max_additivity_error, max_err)
        if max_err > tol_additivity:
            n_additivity_violations += 1

    mr5_passed = n_additivity_violations == 0

    results.append(MetamorphicResult(
        relation_name="MR5_explanation_additivity",
        passed=mr5_passed,
        n_violations=n_additivity_violations,
        max_violation=max_additivity_error,
        details={
            "n_samples_checked": len(X),
            "tolerance": tol_additivity,
            "description": (
                "sum(top_rules contributions) == net_rule_contribution "
                "and net_rule_contribution + bias == margin."
            ),
        },
    ))

    # ------------------------------------------------------------------
    # MR6 — Unit invariance under feature scaling
    # Scaling feature j=0 by c>0 and refitting on the scaled data should
    # yield identical predictions, because membership values are invariant
    # to uniform scaling of feature values and partition anchors.
    # ------------------------------------------------------------------
    c = 2.0
    j_scale = 0  # scale the first screened feature

    X_scaled = X.copy()
    # Scale the raw feature column corresponding to screened index j_scale
    raw_feature_idx = int(clf.selected_feature_indices_[j_scale])
    X_scaled[:, raw_feature_idx] *= c

    clf_scaled = clone(clf)
    clf_scaled.fit(X_scaled, y)

    pred_orig = clf.predict(X)
    pred_scaled = clf_scaled.predict(X_scaled)
    n_unit_violations = int(np.sum(pred_orig != pred_scaled))
    max_unit_violation = 0.0 if n_unit_violations == 0 else float(np.max(
        np.abs(clf.decision_function(X) - clf_scaled.decision_function(X_scaled))
    ))
    mr6_passed = n_unit_violations == 0

    results.append(MetamorphicResult(
        relation_name="MR6_unit_invariance",
        passed=mr6_passed,
        n_violations=n_unit_violations,
        max_violation=max_unit_violation,
        details={
            "scale_factor": c,
            "scaled_raw_feature_index": raw_feature_idx,
            "n_samples_checked": len(X),
            "description": (
                "Scaling feature j by c>0 and refitting on scaled data "
                "should yield identical predictions (membership values are "
                "invariant to uniform coordinate scaling)."
            ),
        },
    ))

    return results


# ---------------------------------------------------------------------------
# Batch conformance suite
# ---------------------------------------------------------------------------

def run_conformance_suite(
    clf,
    datasets: list[tuple[str, np.ndarray, np.ndarray]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run conformance and metamorphic checks across multiple datasets.

    Parameters
    ----------
    clf : fitted SparseMaxMarginFuzzyRuleMachine
    datasets : list of (dataset_name, X, y) triples
    output_dir : path
        Directory to write JSON result files.

    Returns
    -------
    dict with keys 'conformance' and 'metamorphic'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_conformance: list[dict] = []
    all_metamorphic: list[dict] = []

    for dataset_name, X, y in datasets:
        conf_result = check_conformance(clf, X, dataset_name)
        mr_results = check_metamorphic_relations(clf, X, y)

        all_conformance.append(asdict(conf_result))
        for mr in mr_results:
            mr_dict = asdict(mr)
            mr_dict["dataset_name"] = dataset_name
            all_metamorphic.append(mr_dict)

    summary = {
        "conformance": all_conformance,
        "metamorphic": all_metamorphic,
    }

    (output_dir / "conformance_results.json").write_text(
        json.dumps(all_conformance, indent=2)
    )
    (output_dir / "metamorphic_results.json").write_text(
        json.dumps(all_metamorphic, indent=2)
    )

    return summary
