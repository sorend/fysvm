"""Deterministic evaluation smoke test for CSRQClassifier."""

from __future__ import annotations

import numpy as np
import pytest

from fysvm.csrq import CSRQClassifier
from fysvm.quotient import RuleAtom, canonical_dimension
from fysvm.rule_svm import FuzzyRule, RuleCondition
from sklearn.metrics import balanced_accuracy_score


def _make_atom(feature: int, term: str) -> RuleAtom:
    rule = FuzzyRule((RuleCondition(feature, term),))
    return RuleAtom(rule=rule, scale=1.0, cost=1.0)


def make_synthetic_dataset(n: int = 200, d: int = 4, seed: int = 42):
    """Generate a simple linearly separable synthetic dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    # Decision boundary: sum of first two features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Smoke test: complete mode on synthetic data
# ---------------------------------------------------------------------------

def test_smoke_complete_mode():
    X, y = make_synthetic_dataset()
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = balanced_accuracy_score(y, preds)
    # Should achieve at least 60% balanced accuracy on training data
    assert acc >= 0.60, f"Balanced accuracy too low: {acc:.3f}"


def test_smoke_degree_2():
    X, y = make_synthetic_dataset(n=150, d=3)
    clf = CSRQClassifier(C=1.0, max_rule_length=2)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = balanced_accuracy_score(y, preds)
    assert acc >= 0.60


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    """Two fits with same data and params produce identical output."""
    X, y = make_synthetic_dataset()
    clf1 = CSRQClassifier(C=1.0, max_rule_length=1, random_state=0)
    clf1.fit(X, y)

    clf2 = CSRQClassifier(C=1.0, max_rule_length=1, random_state=0)
    clf2.fit(X, y)

    np.testing.assert_array_equal(clf1.c_float_, clf2.c_float_)


# ---------------------------------------------------------------------------
# Dictionary mode smoke test
# ---------------------------------------------------------------------------

def test_smoke_dictionary_mode():
    X, y = make_synthetic_dataset(n=120, d=3)
    atoms = tuple(_make_atom(j, t) for j in range(3) for t in ("low", "high"))
    clf = CSRQClassifier(
        C=1.0, max_rule_length=1,
        semantic_space="dictionary",
        rule_dictionary=atoms,
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = balanced_accuracy_score(y, preds)
    assert acc >= 0.55


# ---------------------------------------------------------------------------
# Reproducibility of canonical fit across instantiations
# ---------------------------------------------------------------------------

def test_reproducible_canonical_coefficients():
    """Same dataset always gives the same canonical coefficients."""
    X, y = make_synthetic_dataset(seed=7)

    coefs_list = []
    for _ in range(3):
        clf = CSRQClassifier(C=1.0, max_rule_length=1)
        clf.fit(X, y)
        coefs_list.append(clf.c_float_.copy())

    for i in range(1, len(coefs_list)):
        np.testing.assert_allclose(
            coefs_list[0], coefs_list[i],
            atol=1e-5,
            err_msg=f"Run {i} produced different coefficients",
        )


# ---------------------------------------------------------------------------
# Balanced vs imbalanced class weights
# ---------------------------------------------------------------------------

def test_balanced_class_weight():
    rng = np.random.default_rng(0)
    n_pos, n_neg = 30, 120
    X = np.vstack([rng.normal([1, 1], 0.5, (n_pos, 2)),
                   rng.normal([-1, -1], 0.5, (n_neg, 2))])
    y = np.array([1] * n_pos + [0] * n_neg)

    clf_no_weight = CSRQClassifier(C=1.0, max_rule_length=1)
    clf_no_weight.fit(X, y)

    clf_balanced = CSRQClassifier(C=1.0, max_rule_length=1, class_weight="balanced")
    clf_balanced.fit(X, y)

    # Both should give predictions
    assert len(clf_balanced.predict(X)) == len(y)


# ---------------------------------------------------------------------------
# Explain API
# ---------------------------------------------------------------------------

def test_explain_returns_top_n():
    X, y = make_synthetic_dataset(n=80, d=3)
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y)
    explanations = clf.explain(X[:5], top_n=3)
    assert len(explanations) == 5
    for exp in explanations:
        assert "top_rules" in exp
        assert len(exp["top_rules"]) <= 3


# ---------------------------------------------------------------------------
# Semantic map method
# ---------------------------------------------------------------------------

def test_semantic_map_none_in_complete_mode():
    X, y = make_synthetic_dataset(n=80, d=3)
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y)
    assert clf.semantic_map() is None


def test_semantic_map_present_in_dictionary_mode():
    X, y = make_synthetic_dataset(n=80, d=2)
    atoms = tuple(_make_atom(j, t) for j in range(2) for t in ("low", "high"))
    clf = CSRQClassifier(
        C=1.0, max_rule_length=1,
        semantic_space="dictionary",
        rule_dictionary=atoms,
    )
    clf.fit(X, y)
    smap = clf.semantic_map()
    assert smap is not None


# ---------------------------------------------------------------------------
# Decode API
# ---------------------------------------------------------------------------

def test_decode_canonical_api():
    X, y = make_synthetic_dataset(n=80, d=2)
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y)
    result = clf.decode(method="canonical")
    assert result["method"] == "canonical"
    assert "monomials" in result


def test_decode_rref_api_dictionary_mode():
    X, y = make_synthetic_dataset(n=80, d=2)
    atoms = tuple(_make_atom(j, t) for j in range(2) for t in ("low", "high"))
    clf = CSRQClassifier(
        C=1.0, max_rule_length=1,
        semantic_space="dictionary",
        rule_dictionary=atoms,
    )
    clf.fit(X, y)
    result = clf.decode(method="rref")
    assert result["method"] == "rref"
    assert "status" in result
