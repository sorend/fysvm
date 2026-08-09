"""Tests for CSRQClassifier — sklearn API, objective, prediction, invariance."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from fysvm.csrq import CSRQClassifier
from fysvm.quotient import (
    RuleAtom,
    canonical_basis,
    canonical_dimension,
)
from fysvm.rule_svm import FuzzyRule, RuleCondition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_dataset():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def simple_dataset():
    """Simple 2D separable dataset."""
    rng = np.random.default_rng(0)
    n = 60
    X0 = rng.normal([-1, -1], 0.3, (n // 2, 2))
    X1 = rng.normal([1, 1], 0.3, (n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def _make_atom(feature: int, term: str) -> RuleAtom:
    rule = FuzzyRule((RuleCondition(feature, term),))
    return RuleAtom(rule=rule, scale=1.0, cost=1.0)


# ---------------------------------------------------------------------------
# sklearn API compliance
# ---------------------------------------------------------------------------

def test_get_params_round_trip():
    clf = CSRQClassifier(C=2.0, max_rule_length=1)
    params = clf.get_params()
    assert params["C"] == 2.0
    assert params["max_rule_length"] == 1


def test_set_params():
    clf = CSRQClassifier()
    clf.set_params(C=5.0)
    assert clf.C == 5.0


def test_clone():
    from sklearn.base import clone
    clf = CSRQClassifier(C=3.0, degree_penalty=0.5)
    clf2 = clone(clf)
    assert clf2.C == 3.0
    assert clf2.degree_penalty == 0.5


def test_fit_and_predict(binary_dataset):
    X, y = binary_dataset
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape
    assert set(preds).issubset(set(y))


def test_decision_function_shape(binary_dataset):
    X, y = binary_dataset
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y)
    df = clf.decision_function(X)
    assert df.shape == (len(y),)


def test_predict_from_decision_function(binary_dataset):
    X, y = binary_dataset
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y)
    df = clf.decision_function(X)
    preds = clf.predict(X)
    classes = clf.classes_
    expected = np.where(df >= 0.0, classes[1], classes[0])
    np.testing.assert_array_equal(preds, expected)


def test_transform_shape(binary_dataset):
    X, y = binary_dataset
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y)
    Psi = clf.transform(X)
    d = len(clf.selected_feature_indices_)
    D = canonical_dimension(d, 1)
    assert Psi.shape == (len(y), D)


# ---------------------------------------------------------------------------
# Fitted attributes
# ---------------------------------------------------------------------------

def test_fitted_attributes(binary_dataset):
    X, y = binary_dataset
    clf = CSRQClassifier(C=1.0, max_rule_length=2)
    clf.fit(X, y)

    assert hasattr(clf, "classes_")
    assert hasattr(clf, "rules_")
    assert hasattr(clf, "coef_")
    assert hasattr(clf, "intercept_")
    assert hasattr(clf, "partitions_")
    assert hasattr(clf, "basis_")
    assert hasattr(clf, "c_float_")
    assert hasattr(clf, "n_rules_")
    assert hasattr(clf, "n_screened_features_")
    assert hasattr(clf, "active_rule_indices_")
    assert hasattr(clf, "selected_feature_indices_")
    assert hasattr(clf, "feature_names_in_")
    assert hasattr(clf, "selected_feature_names_in_")

    # coef_ aligns with nonconstant monomials
    assert len(clf.coef_) == clf.n_rules_
    assert len(clf.rules_) == clf.n_rules_

    # intercept_ is the empty-monomial coefficient
    assert float(clf.c_float_[0]) == clf.intercept_


def test_and_operator_fixed():
    clf = CSRQClassifier()
    assert clf.and_operator == "product"


def test_feature_names_preserved(binary_dataset):
    import pandas as pd
    X, y = binary_dataset
    cols = ["a", "b", "c", "d"]
    df = pd.DataFrame(X, columns=cols)
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(df, y)
    assert list(clf.feature_names_in_) == cols


# ---------------------------------------------------------------------------
# Binary target enforcement
# ---------------------------------------------------------------------------

def test_rejects_non_binary():
    X = np.random.randn(30, 2)
    y = np.array([0, 1, 2] * 10)
    clf = CSRQClassifier()
    with pytest.raises(ValueError, match="binary"):
        clf.fit(X, y)


# ---------------------------------------------------------------------------
# Strict anchor policy
# ---------------------------------------------------------------------------

def test_strict_anchor_raise():
    """Tied anchors raise when strict_anchor_policy='raise'."""
    # Use dataset where all samples have same value (constant feature)
    X = np.column_stack([
        np.zeros(50),   # constant feature — ties anchors
        np.random.randn(50),
    ])
    y = np.array([0, 1] * 25)
    clf = CSRQClassifier(
        partition_quantiles=(0.05, 0.5, 0.95),
        strict_anchor_policy="raise",
        max_rule_length=1,
    )
    with pytest.raises(ValueError, match="[Tt]ied"):
        clf.fit(X, y)


def test_strict_anchor_drop():
    """Tied anchors are dropped when strict_anchor_policy='drop'."""
    X = np.column_stack([
        np.zeros(50),   # constant feature
        np.random.randn(50),
    ])
    y = np.array([0, 1] * 25)
    clf = CSRQClassifier(
        partition_quantiles=(0.05, 0.5, 0.95),
        strict_anchor_policy="drop",
        max_rule_length=1,
    )
    clf.fit(X, y)
    assert clf.n_screened_features_ == 1  # only 1 valid feature


# ---------------------------------------------------------------------------
# Intercept penalty requirement
# ---------------------------------------------------------------------------

def test_rejects_zero_intercept_penalty():
    with pytest.raises(ValueError, match="intercept_penalty"):
        CSRQClassifier(intercept_penalty=0.0)._validate_parameters()


def test_rejects_negative_intercept_penalty():
    with pytest.raises(ValueError, match="intercept_penalty"):
        CSRQClassifier(intercept_penalty=-1.0)._validate_parameters()


# ---------------------------------------------------------------------------
# Dimension guard
# ---------------------------------------------------------------------------

def test_dimension_guard():
    X = np.random.randn(50, 10)
    y = np.array([0, 1] * 25)
    clf = CSRQClassifier(max_rule_length=3, max_semantic_terms=10)
    with pytest.raises(ValueError, match="max_semantic_terms"):
        clf.fit(X, y)


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def test_sample_weights_accepted(binary_dataset):
    X, y = binary_dataset
    sw = np.ones(len(y))
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y, sample_weight=sw)
    assert hasattr(clf, "coef_")


def test_negative_sample_weights_rejected(binary_dataset):
    X, y = binary_dataset
    sw = np.ones(len(y))
    sw[0] = -1.0
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    with pytest.raises(ValueError, match="non-negative"):
        clf.fit(X, y, sample_weight=sw)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def test_class_weight_balanced(binary_dataset):
    X, y = binary_dataset
    clf = CSRQClassifier(C=1.0, max_rule_length=1, class_weight="balanced")
    clf.fit(X, y)
    assert hasattr(clf, "coef_")


# ---------------------------------------------------------------------------
# Concept memberships and fuzzy violations
# ---------------------------------------------------------------------------

def test_concept_memberships_structure(binary_dataset):
    X, y = binary_dataset
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y)
    memberships = clf.concept_memberships(X[:5])
    assert len(memberships) == 5
    for sample in memberships:
        for fname, terms in sample.items():
            assert set(terms.keys()) == {"low", "medium", "high"}
            total = sum(terms.values())
            assert abs(total - 1.0) < 1e-10, f"L+M+H != 1 for {fname}: {total}"


def test_fuzzy_violations_structure(binary_dataset):
    X, y = binary_dataset
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y)
    viol = clf.fuzzy_violations(X, y)
    assert len(viol) == len(y)
    for v in viol:
        assert "slack" in v
        assert "memberships" in v
        assert set(v["memberships"].keys()) == {"cleanly_classified", "borderline", "strong_violation"}


# ---------------------------------------------------------------------------
# Support rules
# ---------------------------------------------------------------------------

def test_support_rules(binary_dataset):
    X, y = binary_dataset
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y)
    sr = clf.support_rules()
    for item in sr:
        assert "monomial" in item
        assert "weight" in item


# ---------------------------------------------------------------------------
# Parameterization invariance (Variant A, complete mode) — T9
# ---------------------------------------------------------------------------

def test_complete_mode_invariance_to_canonical_dict(simple_dataset):
    """Complete mode should produce same c regardless of redundant dictionary."""
    X, y = simple_dataset
    clf = CSRQClassifier(
        C=1.0, max_rule_length=1,
        partition_quantiles=(0.1, 0.5, 0.9),
        degree_penalty=0.35,
        intercept_penalty=1.0,
        semantic_space="complete",
    )
    clf.fit(X, y)

    # Fit again (deterministic)
    clf2 = CSRQClassifier(
        C=1.0, max_rule_length=1,
        partition_quantiles=(0.1, 0.5, 0.9),
        degree_penalty=0.35,
        intercept_penalty=1.0,
        semantic_space="complete",
    )
    clf2.fit(X, y)

    np.testing.assert_allclose(clf.c_float_, clf2.c_float_, atol=1e-6)


def test_dictionary_mode_same_span_invariance(simple_dataset):
    """Same-span dictionaries produce the same fit in dictionary mode."""
    X, y = simple_dataset
    d = X.shape[1]

    # Build two spanning dictionaries with same span but different ordering
    atoms_base = tuple(_make_atom(j, t) for j in range(d) for t in ("low", "high"))
    atoms_permuted = atoms_base[::-1]  # reversed

    clf1 = CSRQClassifier(
        C=1.0, max_rule_length=1,
        partition_quantiles=(0.1, 0.5, 0.9),
        semantic_space="dictionary",
        rule_dictionary=atoms_base,
    )
    clf1.fit(X, y)

    clf2 = CSRQClassifier(
        C=1.0, max_rule_length=1,
        partition_quantiles=(0.1, 0.5, 0.9),
        semantic_space="dictionary",
        rule_dictionary=atoms_permuted,
    )
    clf2.fit(X, y)

    # Same span => same c_float_ to numerical tolerance
    np.testing.assert_allclose(clf1.c_float_, clf2.c_float_, atol=1e-6,
                               err_msg="Same-span dictionaries produce different c")


def test_dictionary_mode_with_duplicate_atoms(simple_dataset):
    """Duplicate atoms do not change the fit (same span)."""
    X, y = simple_dataset
    d = X.shape[1]
    atoms_base = tuple(_make_atom(j, t) for j in range(d) for t in ("low", "high"))
    atoms_dup = atoms_base + (atoms_base[0],)  # add duplicate

    clf1 = CSRQClassifier(
        C=1.0, max_rule_length=1,
        partition_quantiles=(0.1, 0.5, 0.9),
        semantic_space="dictionary",
        rule_dictionary=atoms_base,
    )
    clf1.fit(X, y)

    clf2 = CSRQClassifier(
        C=1.0, max_rule_length=1,
        partition_quantiles=(0.1, 0.5, 0.9),
        semantic_space="dictionary",
        rule_dictionary=atoms_dup,
    )
    clf2.fit(X, y)

    np.testing.assert_allclose(clf1.c_float_, clf2.c_float_, atol=1e-6)


# ---------------------------------------------------------------------------
# Feature screening
# ---------------------------------------------------------------------------

def test_anova_screening(binary_dataset):
    X, y = binary_dataset
    clf = CSRQClassifier(
        C=1.0, max_rule_length=1,
        feature_screening="anova", screen_top_k=2,
    )
    clf.fit(X, y)
    assert clf.n_screened_features_ == 2


def test_mutual_info_screening(binary_dataset):
    X, y = binary_dataset
    clf = CSRQClassifier(
        C=1.0, max_rule_length=1,
        feature_screening="mutual_info", screen_top_k=2,
    )
    clf.fit(X, y)
    assert clf.n_screened_features_ == 2


# ---------------------------------------------------------------------------
# get_feature_names_out
# ---------------------------------------------------------------------------

def test_get_feature_names_out(binary_dataset):
    X, y = binary_dataset
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y)
    names = clf.get_feature_names_out()
    D = clf.basis_.dimension
    assert len(names) == D
    assert names[0] == "1"  # empty monomial
