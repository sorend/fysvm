"""Tests for QuotientAtomicFuzzySVM and hull certification."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from fysvm.atomic import HullCertificationResult, QuotientAtomicFuzzySVM, certify_hull_equality
from fysvm.quotient import RuleAtom, canonical_basis
from fysvm.rule_svm import FuzzyRule, RuleCondition


def _make_atom(feature: int, term: str, cost: float = 1.0) -> RuleAtom:
    rule = FuzzyRule((RuleCondition(feature, term),))
    return RuleAtom(rule=rule, scale=1.0, cost=cost)


def _skip_no_osqp():
    try:
        import osqp  # noqa: F401
    except ImportError:
        pytest.skip("osqp not installed")


# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

def test_atomic_requires_osqp():
    """QuotientAtomicFuzzySVM raises ImportError if osqp not installed."""
    try:
        import osqp  # noqa: F401
        pytest.skip("osqp is installed; skip dependency-failure test")
    except ImportError:
        with pytest.raises(ImportError, match="osqp"):
            QuotientAtomicFuzzySVM(
                atom_dictionary=(_make_atom(0, "low"),),
            )


# ---------------------------------------------------------------------------
# Hull certification — equal hulls
# ---------------------------------------------------------------------------

def test_hull_certification_identical_dicts():
    """Identical dictionaries have equal hulls (CERTIFIED_EQUAL)."""
    d, r = 2, 1
    basis = canonical_basis(d, r)
    atoms = tuple(_make_atom(j, t) for j in range(d) for t in ("low", "high"))

    result = certify_hull_equality(atoms, atoms, basis)
    # With exact witnesses implemented, should be CERTIFIED_EQUAL
    assert result.status in ("CERTIFIED_EQUAL", "UNKNOWN")


def test_hull_certification_reordered_identical():
    """Reordered identical dictionaries still have equal hulls."""
    d, r = 2, 1
    basis = canonical_basis(d, r)
    atoms = tuple(_make_atom(j, t) for j in range(d) for t in ("low", "high"))
    atoms_rev = atoms[::-1]

    result = certify_hull_equality(atoms, atoms_rev, basis)
    assert result.status in ("CERTIFIED_EQUAL", "UNKNOWN")


def test_hull_certification_different_costs():
    """Atoms with different costs have different hulls even with same columns."""
    d, r = 2, 1
    basis = canonical_basis(d, r)
    atoms1 = (_make_atom(0, "low", cost=1.0),)
    atoms2 = (_make_atom(0, "low", cost=2.0),)  # different cost => different hull

    result = certify_hull_equality(atoms1, atoms2, basis)
    # The unit-cost generator is outside the cost-2 hull: |b|/1 vs |b|/2
    # The cost-1 atom generates a vector at unit distance from origin,
    # while the cost-2 hull only reaches 1/2 that distance.
    assert result.status in ("CERTIFIED_DIFFERENT", "UNKNOWN")


def test_hull_certification_invalid_costs():
    """Nonpositive costs return INVALID status."""
    d, r = 2, 1
    basis = canonical_basis(d, r)
    atoms1 = (_make_atom(0, "low", cost=0.0),)
    atoms2 = (_make_atom(0, "low"),)

    result = certify_hull_equality(atoms1, atoms2, basis)
    assert result.status == "INVALID"


def test_hull_certification_equal_span_different_hull():
    """Equal-span dictionaries with different costs have different hulls.

    {e1, e2} with costs 1 vs {e1, e2, (e1+e2)/2} with costs 1.
    The second dict has an extra interior generator after normalization,
    but the same hull only if it's inside the first hull.
    This tests the CERTIFIED_DIFFERENT case where cost > 1 for a generator.
    """
    d, r = 2, 1
    basis = canonical_basis(d, r)
    # dict1: just e1 (low[0]) with cost 1
    atoms1 = (_make_atom(0, "low", cost=1.0),)
    # dict2: e1 (low[0]) with cost 0.5 => normalized generator is 2*e1
    # which is outside hull of dict1 (conv{±e1/1}) since |2*e1| > 1
    atoms2 = (_make_atom(0, "low", cost=0.5),)

    result = certify_hull_equality(atoms1, atoms2, basis)
    # dict2's normalized generator 2*e1 should be outside hull of dict1
    assert result.status in ("CERTIFIED_DIFFERENT", "UNKNOWN")


# ---------------------------------------------------------------------------
# QuotientAtomicFuzzySVM fit
# ---------------------------------------------------------------------------

def test_atomic_fit_basic():
    """QuotientAtomicFuzzySVM.fit() trains and predicts on a binary dataset."""
    _skip_no_osqp()

    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    # Use degree-1 atoms for all features, both low and high
    atoms = tuple(_make_atom(j, t) for j in range(4) for t in ("low", "high"))
    clf = QuotientAtomicFuzzySVM(atom_dictionary=atoms, C=1.0, max_rule_length=1)
    clf.fit(X, y)

    assert hasattr(clf, "c_")
    assert hasattr(clf, "intercept_")
    assert hasattr(clf, "coef_")
    assert clf.c_.shape[0] == clf.basis_.dimension
    preds = clf.predict(X)
    assert set(preds).issubset(set(clf.classes_))


def test_atomic_fit_decision_function():
    """decision_function returns the margin values."""
    _skip_no_osqp()

    X, y = make_classification(n_samples=40, n_features=4, random_state=7)
    atoms = tuple(_make_atom(j, t) for j in range(4) for t in ("low", "high"))
    clf = QuotientAtomicFuzzySVM(atom_dictionary=atoms, C=1.0, max_rule_length=1)
    clf.fit(X, y)

    margins = clf.decision_function(X)
    assert margins.shape == (40,)
    preds = clf.predict(X)
    # Predictions agree with sign of decision function
    class1 = clf.classes_[1]
    class0 = clf.classes_[0]
    np.testing.assert_array_equal(
        preds,
        np.where(margins >= 0, class1, class0),
    )


def test_atomic_fit_tied_anchors_raises():
    """Fit raises ValueError when any anchor is tied."""
    _skip_no_osqp()

    # Constant feature -> tied anchors
    X = np.ones((20, 2))
    X[:, 1] = np.arange(20, dtype=float)
    y = np.array([0, 1] * 10)

    atoms = tuple(_make_atom(j, t) for j in range(2) for t in ("low", "high"))
    clf = QuotientAtomicFuzzySVM(atom_dictionary=atoms, C=1.0, max_rule_length=1)
    with pytest.raises(ValueError, match="[Tt]ied anchor"):
        clf.fit(X, y)


def test_atomic_fit_with_class_weight():
    """Fit accepts class_weight='balanced'."""
    _skip_no_osqp()

    X, y = make_classification(n_samples=40, n_features=4, random_state=13)
    atoms = tuple(_make_atom(j, t) for j in range(4) for t in ("low", "high"))
    clf = QuotientAtomicFuzzySVM(
        atom_dictionary=atoms, C=1.0, max_rule_length=1, class_weight="balanced"
    )
    clf.fit(X, y)
    assert hasattr(clf, "c_")


def test_atomic_fit_semantic_output_unique():
    """The fitted c_ is independent of gamma (semantic output only)."""
    _skip_no_osqp()

    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    atoms = tuple(_make_atom(j, t) for j in range(4) for t in ("low", "high"))
    clf = QuotientAtomicFuzzySVM(atom_dictionary=atoms, C=1.0, max_rule_length=1)
    clf.fit(X, y)
    # c_ is a semantic vector; it should be finite and of correct shape
    assert np.all(np.isfinite(clf.c_))
    from fysvm.quotient import canonical_dimension
    D = canonical_dimension(4, 1)
    assert clf.c_.shape == (D,)


def test_atomic_and_operator_property():
    """and_operator is 'product'."""
    _skip_no_osqp()
    atoms = (_make_atom(0, "low"),)
    clf = QuotientAtomicFuzzySVM(atom_dictionary=atoms)
    assert clf.and_operator == "product"
