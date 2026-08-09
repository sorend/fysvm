"""Deterministic tests for sound property certificates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fysvm.certificates import (
    _interval_f,
    certificate_feature_exclusion,
    certificate_monotonicity,
    certificate_robustness,
    certificate_safe_region,
    check_validity_domain,
    find_feature_index,
)
from fysvm.rule_svm import FuzzyRule, FuzzyRuleSVM, RuleCondition


def _make_separable_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return two-feature data with a binary target determined by feature zero."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (n, 2))
    y = (X[:, 0] > 0.5).astype(int)
    return X, y


def _fit_clf(and_operator: Any = "min") -> FuzzyRuleSVM:
    X, y = _make_separable_data()
    return FuzzyRuleSVM(
        C=0.3,
        penalty="l1",
        and_operator=and_operator,
        max_rule_length=2,
        max_rules=64,
        min_rule_coverage=0.01,
        rule_length_penalty=0.25,
        random_state=0,
        max_iter=10000,
        feature_names=["glucose", "bmi"],
    ).fit(X, y)


def _set_rules(
    clf: FuzzyRuleSVM,
    conditions: list[tuple[tuple[int, str], ...]],
    coefficients: list[float],
    intercept: float = 0.0,
) -> None:
    """Replace learned rules while retaining a genuinely fitted model fixture."""
    clf.rules_ = [
        FuzzyRule(tuple(RuleCondition(feature, term) for feature, term in rule))
        for rule in conditions
    ]
    clf.coef_ = np.asarray(coefficients, dtype=np.float64)
    clf.intercept_ = float(intercept)
    clf.n_rules_ = len(clf.rules_)
    clf.active_rule_indices_ = np.flatnonzero(np.abs(clf.coef_) > 1e-12)


@pytest.mark.parametrize("and_operator", ["min", "product"])
def test_monotonicity_length_one_rules_use_exact_derivatives(and_operator):
    clf = _fit_clf(and_operator)
    _set_rules(
        clf,
        [((0, "low"),), ((0, "high"),)],
        [-1.0, 1.0],
    )

    positive = certificate_monotonicity(clf, 0, "positive")
    negative = certificate_monotonicity(clf, 0, "negative")

    assert positive.status == "CERTIFIED"
    assert positive.min_slack > 0.0
    assert negative.status == "UNKNOWN"
    assert negative.counterexample is None


@pytest.mark.parametrize("and_operator", ["min", "product"])
def test_multicondition_cancellation_is_not_falsely_certified(and_operator):
    """Point derivatives cancel, but independently gated rules need intervals."""
    clf = _fit_clf(and_operator)
    _set_rules(
        clf,
        [
            ((0, "low"), (1, "low")),
            ((0, "medium"), (1, "medium")),
            ((0, "high"), (1, "high")),
        ],
        [1.0, 1.0, 1.0],
    )

    result = certificate_monotonicity(clf, 0, "positive")

    # The old point-sum construction produced zero on both segments and CERTIFIED.
    assert result.status == "UNKNOWN"
    assert result.counterexample is None
    assert min(result.details["derivative_lower_by_segment"]) < 0.0
    assert max(result.details["derivative_upper_by_segment"]) > 0.0


def test_tied_partition_fails_closed_unless_feature_is_structurally_unused():
    clf = _fit_clf()
    partition_type = type(clf.partitions_[0])
    clf.partitions_[0] = partition_type(low=0.5, medium=0.5, high=0.8)
    _set_rules(clf, [((0, "low"),)], [1.0])

    used = certificate_monotonicity(clf, 0)
    assert used.status == "UNKNOWN"
    assert used.details["reason"] == "tied partition"

    clf.coef_[0] = 0.0
    unused = certificate_monotonicity(clf, 0)
    assert unused.status == "CERTIFIED-TRIVIAL"
    assert "structurally absent" in unused.details["reason"]


def test_exclusion_checks_exact_tiny_coefficients_and_supports_softmin():
    clf = _fit_clf("softmin")
    _set_rules(clf, [((1, "low"),)], [1e-15])
    assert clf.active_rule_indices_.size == 0

    result = certificate_feature_exclusion(clf, [1])

    assert result.status == "COUNTEREXAMPLE"
    assert result.counterexample is not None
    assert result.counterexample["violating_rules"][0]["beta"] == 1e-15
    assert result.details["property"] == "structural non-reference"

    clf.coef_[0] = 0.0
    assert certificate_feature_exclusion(clf, [1]).status == "CERTIFIED"


def test_safe_region_does_not_skip_tiny_nonzero_coefficient():
    clf = _fit_clf()
    _set_rules(clf, [((0, "low"),)], [1e-15], intercept=-0.5e-15)
    low_anchor = clf.partitions_[0].low

    result = certificate_safe_region(
        clf,
        {0: (low_anchor - 1.0, low_anchor - 1.0)},
        target_class_index=0,
    )

    assert result.status == "UNKNOWN"
    assert result.details["lb_f"] == pytest.approx(0.5e-15)
    assert result.details["ub_f"] == pytest.approx(0.5e-15)


def test_robustness_returns_unknown_on_both_sides_of_boundary():
    """An inconclusive interval is not a fragility counterexample for either class."""
    clf = _fit_clf()
    _set_rules(clf, [((0, "high"),)], [1.0], intercept=-0.25)
    partition = clf.partitions_[0]
    width = partition.high - partition.medium
    epsilon = 0.4 * width
    x_positive = np.array([partition.medium + 0.5 * width, 0.5])
    x_negative = np.array([partition.medium + 0.1 * width, 0.5])

    positive = certificate_robustness(clf, x_positive, epsilon, feature_indices=[0])
    negative = certificate_robustness(clf, x_negative, epsilon, feature_indices=[0])

    assert positive.details["f_x"] > 0.0
    assert negative.details["f_x"] < 0.0
    assert positive.status == negative.status == "UNKNOWN"
    assert positive.counterexample is negative.counterexample is None
    assert positive.details["lb_f"] < 0.0
    assert negative.details["ub_f"] > 0.0


def test_zero_perturbation_robustness_is_trivial():
    clf = _fit_clf()
    _set_rules(clf, [((0, "high"),)], [1.0], intercept=-2.0)
    result = certificate_robustness(clf, np.array([0.5, 0.5]), epsilon=0.0)

    assert result.status == "CERTIFIED-TRIVIAL"
    assert result.details["reason"] == "zero perturbation vector"

    _set_rules(clf, [((0, "high"),)], [1.0], intercept=0.0)
    boundary = np.array([clf.partitions_[0].medium, 0.5])
    boundary_result = certificate_robustness(clf, boundary, epsilon=0.0)
    assert boundary_result.status == "CERTIFIED-TRIVIAL"
    assert boundary_result.details["f_x"] == 0.0


@pytest.mark.parametrize("and_operator", ["min", "product"])
def test_decision_interval_contains_dense_grid(and_operator):
    clf = _fit_clf(and_operator)
    box = {0: (0.2, 0.8), 1: (0.1, 0.9)}
    lower, upper = _interval_f(clf, box)
    points = np.array(
        [(x0, x1) for x0 in np.linspace(0.2, 0.8, 21) for x1 in np.linspace(0.1, 0.9, 21)]
    )
    margins = clf.decision_function(points)

    assert np.all(margins >= lower)
    assert np.all(margins <= upper)


def test_monotonicity_rejects_invalid_public_inputs():
    clf = _fit_clf()

    invalid_feature_indices: list[Any] = [-1, 2, 0.5, True]
    for feature_index in invalid_feature_indices:
        with pytest.raises(ValueError, match="feature index"):
            certificate_monotonicity(clf, feature_index)
    with pytest.raises(ValueError, match="direction"):
        certificate_monotonicity(clf, 0, "up")


def test_robustness_rejects_invalid_public_inputs():
    clf = _fit_clf()
    valid_x = np.array([0.5, 0.5])

    invalid_x_values = [np.array([0.5]), np.array([[0.5, 0.5]]), np.array([np.nan, 0.5])]
    for x in invalid_x_values:
        with pytest.raises(ValueError, match="x must be finite"):
            certificate_robustness(clf, x, 0.1)

    invalid_epsilons = [np.array([0.1]), np.array([0.1, np.inf]), -0.1]
    for epsilon in invalid_epsilons:
        with pytest.raises(ValueError, match="epsilon"):
            certificate_robustness(clf, valid_x, epsilon)

    with pytest.raises(ValueError, match="feature index"):
        certificate_robustness(clf, valid_x, 0.1, feature_indices=[2])


def test_safe_region_and_exclusion_reject_invalid_public_inputs():
    clf = _fit_clf()

    invalid_boxes = [{2: (0.0, 1.0)}, {0: (np.nan, 1.0)}, {0: (1.0, 0.0)}]
    for box in invalid_boxes:
        with pytest.raises(ValueError):
            certificate_safe_region(clf, box)

    invalid_targets: list[Any] = [-1, 2, 0.5, True]
    for target in invalid_targets:
        with pytest.raises(ValueError, match="target_class_index"):
            certificate_safe_region(clf, {}, target)

    invalid_exclusion_indices: list[list[Any]] = [[-1], [2], [0.5], [True]]
    for indices in invalid_exclusion_indices:
        with pytest.raises(ValueError, match="feature index"):
            certificate_feature_exclusion(clf, indices)


def test_unsupported_operator_and_unfitted_models_remain_unknown():
    softmin = _fit_clf("softmin")
    X, _ = _make_separable_data()
    unfitted = FuzzyRuleSVM()

    assert check_validity_domain(softmin) is False
    assert certificate_monotonicity(softmin, 0).status == "UNKNOWN"
    assert certificate_robustness(softmin, X[0], 0.1).status == "UNKNOWN"
    assert certificate_safe_region(softmin, {}, 0).status == "UNKNOWN"
    assert certificate_monotonicity(unfitted, 0).status == "UNKNOWN"
    assert certificate_robustness(unfitted, X[0], 0.1).status == "UNKNOWN"
    assert certificate_feature_exclusion(unfitted, [0]).status == "UNKNOWN"
    assert certificate_safe_region(unfitted, {}, 0).status == "UNKNOWN"


def test_find_feature_index_matches_normalized_names():
    clf = _fit_clf()

    assert find_feature_index(clf, "Glucose") == 0
    assert find_feature_index(clf, "BMI") == 1
    assert find_feature_index(clf, "xyz_unknown") is None
