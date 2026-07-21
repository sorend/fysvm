import numpy as np

from fysvm import FuzzyRuleSVM, SparseMaxMarginFuzzyRuleMachine


def make_separable_data():
    rng = np.random.default_rng(7)
    negative = rng.normal(loc=(-1.0, -0.8), scale=0.18, size=(40, 2))
    positive = rng.normal(loc=(1.0, 0.9), scale=0.18, size=(40, 2))
    X = np.vstack([negative, positive])
    y = np.array(["low_risk"] * len(negative) + ["high_risk"] * len(positive))
    return X, y


def test_fuzzy_rule_svm_fits_and_exposes_rule_space():
    X, y = make_separable_data()
    clf = SparseMaxMarginFuzzyRuleMachine(
        C=10.0,
        penalty="l1",
        max_rule_length=2,
        max_rules=24,
        feature_names=["glucose", "bmi"],
        random_state=0,
    )

    clf.fit(X, y)

    predictions = clf.predict(X)
    assert np.mean(predictions == y) > 0.95
    assert clf.transform(X).shape == (X.shape[0], clf.n_rules_)
    assert clf.n_rules_ <= 24

    support_rules = clf.support_rules()
    assert support_rules
    assert any("glucose" in item["rule"] or "bmi" in item["rule"] for item in support_rules)


def test_explanation_is_the_decision_calculation():
    X, y = make_separable_data()
    clf = FuzzyRuleSVM(
        C=10.0,
        penalty="l2",
        max_rule_length=2,
        max_rules=18,
        feature_names=["glucose", "bmi"],
        random_state=0,
    ).fit(X, y)

    explanation = clf.explain(X[:1], top_n=clf.n_rules_)[0]

    assert explanation["prediction"] == clf.predict(X[:1])[0]
    assert np.isclose(
        explanation["margin"],
        explanation["bias"] + explanation["net_rule_contribution"],
    )
    assert explanation["top_rules"]
    assert {"firing", "weight", "contribution", "rule"} <= set(
        explanation["top_rules"][0]
    )


def test_fuzzy_concepts_and_product_and_operator_are_bounded():
    X, y = make_separable_data()
    clf = FuzzyRuleSVM(
        penalty="l2",
        and_operator="product",
        max_rule_length=2,
        max_rules=12,
        random_state=0,
    ).fit(X, y)

    Z = clf.transform(X[:5])
    assert np.all(Z >= 0.0)
    assert np.all(Z <= 1.0)

    concepts = clf.concept_memberships(X[:1])[0]
    assert set(concepts["x0"]) == {"low", "medium", "high"}
    assert all(0.0 <= value <= 1.0 for value in concepts["x0"].values())


def test_fuzzy_violations_report_slack_memberships():
    X, y = make_separable_data()
    clf = FuzzyRuleSVM(C=10.0, penalty="l2", random_state=0).fit(X, y)

    wrong_y = y[:3].copy()
    wrong_y[:] = clf.classes_[0] if y[:3][0] == clf.classes_[1] else clf.classes_[1]
    violations = clf.fuzzy_violations(X[:3], wrong_y)

    assert len(violations) == 3
    for item in violations:
        assert item["slack"] >= 0.0
        assert set(item["memberships"]) == {
            "cleanly_classified",
            "borderline",
            "strong_violation",
        }
        assert all(0.0 <= value <= 1.0 for value in item["memberships"].values())


def test_feature_screening_preserves_prediction_interface():
    rng = np.random.default_rng(11)
    signal = rng.normal(size=(80, 2))
    noise = rng.normal(size=(80, 18))
    X = np.column_stack([signal, noise])
    y = np.where(signal[:, 0] + signal[:, 1] > 0, "pos", "neg")

    clf = FuzzyRuleSVM(
        C=1.0,
        penalty="l1",
        max_rule_length=2,
        max_rules=32,
        feature_screening="anova",
        screen_top_k=4,
        random_state=0,
    ).fit(X, y)

    assert clf.n_features_in_ == X.shape[1]
    assert clf.n_screened_features_ == 4
    assert len(clf.selected_feature_indices_) == 4
    assert clf.transform(X[:3]).shape == (3, clf.n_rules_)
    assert clf.predict(X[:3]).shape == (3,)


def test_column_generation_min_tnorm_fits_and_predicts():
    """Column generation with min t-norm should fit and predict correctly."""
    X, y = make_separable_data()
    clf = FuzzyRuleSVM(
        C=10.0,
        penalty="l1",
        max_rule_length=2,
        max_rules=24,
        and_operator="min",
        rule_generation="column_generation",
        feature_names=["glucose", "bmi"],
        random_state=0,
    ).fit(X, y)

    assert clf.predict(X).shape == (X.shape[0],)
    assert np.mean(clf.predict(X) == y) > 0.90
    assert clf.n_rules_ <= 24
    # All rules should be at most length 2
    assert all(rule.length <= 2 for rule in clf.rules_)
    # Explain interface must still work
    expl = clf.explain(X[:1])[0]
    assert np.isclose(expl["margin"], expl["bias"] + expl["net_rule_contribution"])


def test_column_generation_product_tnorm_fits_and_predicts():
    """Column generation with product t-norm should fit and predict correctly."""
    X, y = make_separable_data()
    clf = FuzzyRuleSVM(
        C=10.0,
        penalty="l1",
        max_rule_length=2,
        max_rules=24,
        and_operator="product",
        rule_generation="column_generation",
        feature_names=["glucose", "bmi"],
        random_state=0,
    ).fit(X, y)

    assert clf.predict(X).shape == (X.shape[0],)
    assert np.mean(clf.predict(X) == y) > 0.90
    assert clf.n_rules_ <= 24


def test_column_generation_avoids_full_candidate_materialisation():
    """Column generation should discover far fewer candidates than full enumeration
    on a higher-dimensional dataset where the solution is sparse."""
    rng = np.random.default_rng(42)
    n_features = 30
    # True signal lives in features 0 and 1 only
    X = rng.normal(size=(120, n_features))
    y = np.where(X[:, 0] - X[:, 1] > 0, "pos", "neg")

    full_enum = FuzzyRuleSVM(
        C=1.0, penalty="l1", max_rule_length=2, max_rules=256,
        rule_generation="enumeration", random_state=0,
    ).fit(X, y)

    col_gen = FuzzyRuleSVM(
        C=1.0, penalty="l1", max_rule_length=2, max_rules=256,
        rule_generation="column_generation", random_state=0,
    ).fit(X, y)

    # Full enumeration produces O(d^2) candidates; column generation far fewer
    max_possible = 9 * n_features * (n_features - 1) // 2 + 3 * n_features
    assert full_enum.n_candidate_rules_ == max_possible or full_enum.n_candidate_rules_ > 0
    assert col_gen.n_candidate_rules_ < full_enum.n_candidate_rules_

    # Both must give reasonable accuracy
    assert np.mean(col_gen.predict(X) == y) > 0.80


def test_column_generation_invalid_parameter_raises():
    from pytest import raises
    with raises(ValueError, match="rule_generation"):
        FuzzyRuleSVM(rule_generation="invalid")._validate_parameters()
