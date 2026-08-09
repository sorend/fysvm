"""Tests for bootstrap stability and finite C-grid near-optimal analysis."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fysvm.rule_svm import FuzzyRuleSVM
from fysvm.stability import (
    BootstrapStabilityResult,
    CertificateRetentionResult,
    FiniteCGridResult,
    bootstrap_prediction_stability,
    certificate_retention_bootstrap,
    rashomon_stability,
)


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


def _make_clearly_separable(n_per_class: int = 40, n_features: int = 2, seed: int = 0):
    """Linearly separable dataset: class 0 in [-1, -0.5], class 1 in [0.5, 1] for feature 0."""
    rng = np.random.default_rng(seed)
    X0 = rng.uniform(-1.0, -0.5, (n_per_class, n_features))
    X1 = rng.uniform(0.5, 1.0, (n_per_class, n_features))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def _base_clf_factory():
    """Factory that returns a fresh FuzzyRuleSVM with fast settings."""
    return FuzzyRuleSVM(
        C=1.0,
        penalty="l1",
        and_operator="min",
        max_rule_length=2,
        max_rules=64,
        min_rule_coverage=0.01,
        rule_length_penalty=0.35,
        class_weight="balanced",
        random_state=0,
        max_iter=5000,
    )


def _finite_c_grid_clf_factory(C: float):
    return FuzzyRuleSVM(
        C=C,
        penalty="l1",
        and_operator="min",
        max_rule_length=2,
        max_rules=64,
        min_rule_coverage=0.01,
        rule_length_penalty=0.35,
        class_weight="balanced",
        random_state=0,
        max_iter=5000,
    )


# ---------------------------------------------------------------------------
# Tests: bootstrap_prediction_stability
# ---------------------------------------------------------------------------


def test_bootstrap_stability_identical_data():
    """On clearly separable data any bootstrap resample gives identical predictions → agreement ≈ 1.0."""
    X, y = _make_clearly_separable(n_per_class=50, seed=1)
    # 80/20 split
    n_train = 80
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    result = bootstrap_prediction_stability(
        _base_clf_factory,
        X_train,
        y_train,
        X_test,
        n_bootstrap=20,
        random_state=42,
        dataset_name="separable_test",
    )

    # With clearly separable data all bootstrap models should agree with the reference
    assert result.mean_prediction_agreement >= 0.95, (
        f"Expected high agreement on separable data, got {result.mean_prediction_agreement:.4f}"
    )
    assert result.n_bootstrap == 20


def test_bootstrap_stability_agreement_is_bounded():
    """mean_prediction_agreement must be in [0, 1] for arbitrary data."""
    rng = np.random.default_rng(99)
    X_train = rng.standard_normal((60, 4))
    y_train = rng.integers(0, 2, 60)
    # Ensure both classes present
    y_train[:30] = 0
    y_train[30:] = 1
    X_test = rng.standard_normal((20, 4))

    result = bootstrap_prediction_stability(
        _base_clf_factory,
        X_train,
        y_train,
        X_test,
        n_bootstrap=10,
        random_state=7,
        dataset_name="bounded_test",
    )

    assert isinstance(result, BootstrapStabilityResult)
    assert 0.0 <= result.mean_prediction_agreement <= 1.0
    assert 0.0 <= result.std_prediction_agreement
    assert len(result.per_bootstrap_agreement) == result.n_bootstrap
    assert len(result.per_sample_agreement) == len(X_test)
    assert np.all(result.per_sample_agreement >= 0.0)
    assert np.all(result.per_sample_agreement <= 1.0)


def test_bootstrap_aggregation_is_over_models(monkeypatch):
    """Summary moments and quantiles use per-bootstrap, not per-sample, rates."""
    reference_predictions = np.array([0, 0, 1, 1])
    bootstrap_predictions = [
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 0]),
        np.array([1, 1, 1, 1]),
    ]

    class ReferenceClassifier:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return reference_predictions

    def fake_bootstrap_fit(*args, bootstrap_seed, **kwargs):
        return bootstrap_predictions[bootstrap_seed]

    monkeypatch.setattr(
        "fysvm.stability._fit_and_predict_bootstrap", fake_bootstrap_fit
    )
    result = bootstrap_prediction_stability(
        ReferenceClassifier,
        np.arange(8, dtype=float).reshape(4, 2),
        np.array([0, 0, 1, 1]),
        np.arange(8, dtype=float).reshape(4, 2),
        n_bootstrap=3,
        random_state=0,
        y_test=np.array([0, 1, 1, 0]),
    )

    expected_per_bootstrap = np.array([1.0, 0.5, 0.5])
    expected_per_sample = np.array([2 / 3, 1 / 3, 1.0, 2 / 3])
    np.testing.assert_allclose(result.per_bootstrap_agreement, expected_per_bootstrap)
    np.testing.assert_allclose(result.per_sample_agreement, expected_per_sample)
    assert result.mean_prediction_agreement == pytest.approx(
        np.mean(expected_per_bootstrap)
    )
    assert result.std_prediction_agreement == pytest.approx(
        np.std(expected_per_bootstrap)
    )
    expected_quantiles = np.quantile(expected_per_bootstrap, [0.05, 0.50, 0.95])
    assert result.q05_prediction_agreement == pytest.approx(expected_quantiles[0])
    assert result.q50_prediction_agreement == pytest.approx(expected_quantiles[1])
    assert result.q95_prediction_agreement == pytest.approx(expected_quantiles[2])
    assert result.reference_test_balanced_accuracy == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: finite C-grid near-optimal validation set
# ---------------------------------------------------------------------------


def test_best_c_is_in_near_optimal_set():
    """The best validation C must always be in the near-optimal set."""
    X, y = _make_clearly_separable(n_per_class=50, seed=2)
    n = len(X)
    X_train, y_train = X[: int(0.6 * n)], y[: int(0.6 * n)]
    X_val, y_val = X[int(0.6 * n) : int(0.8 * n)], y[int(0.6 * n) : int(0.8 * n)]
    X_test, y_test = X[int(0.8 * n) :], y[int(0.8 * n) :]

    small_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    result = rashomon_stability(
        _finite_c_grid_clf_factory,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        c_grid=small_grid,
        delta_bacc=0.02,
    )

    assert isinstance(result, FiniteCGridResult)

    # Best C must always be in the near-optimal set
    best_bacc = max(result.val_balanced_accuracies.values())
    best_c = max(result.val_balanced_accuracies, key=lambda c: result.val_balanced_accuracies[c])
    assert best_c in result.near_optimal_set, (
        f"Best C={best_c} (bacc={best_bacc:.4f}) not in set {result.near_optimal_set}"
    )

    # All members must satisfy the additive best-delta formula.
    for c in result.near_optimal_set:
        assert result.val_balanced_accuracies[c] >= best_bacc - result.delta_bacc - 1e-9, (
            f"C={c} in set but bacc={result.val_balanced_accuracies[c]:.4f} "
            f"< best - delta = {best_bacc - result.delta_bacc:.4f}"
        )


def test_near_optimal_size_increases_with_delta_bacc():
    """A larger additive tolerance must yield a set that is at least as large."""
    X, y = _make_clearly_separable(n_per_class=50, seed=3)
    n = len(X)
    X_train, y_train = X[: int(0.6 * n)], y[: int(0.6 * n)]
    X_val, y_val = X[int(0.6 * n) : int(0.8 * n)], y[int(0.6 * n) : int(0.8 * n)]
    X_test = X[int(0.8 * n) :]
    y_test = y[int(0.8 * n) :]

    small_grid = [0.01, 0.1, 1.0, 10.0, 100.0]

    result_small = rashomon_stability(
        _finite_c_grid_clf_factory,
        X_train, y_train, X_val, y_val, X_test,
        c_grid=small_grid, delta_bacc=0.01,
    )
    result_large = rashomon_stability(
        _finite_c_grid_clf_factory,
        X_train, y_train, X_val, y_val, X_test,
        c_grid=small_grid, delta_bacc=0.20,
    )

    assert result_large.near_optimal_size >= result_small.near_optimal_size, (
        f"Expected larger set for delta=0.20 ({result_large.near_optimal_size}) "
        f"vs delta=0.01 ({result_small.near_optimal_size})"
    )


def test_finite_c_grid_agreement_on_simple_separable():
    """On clearly separable data all C values should learn the same boundary → high agreement."""
    X, y = _make_clearly_separable(n_per_class=60, n_features=1, seed=4)
    # Shuffle to ensure stratification across splits
    rng = np.random.default_rng(4)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n = len(X)
    X_train, y_train = X[: int(0.6 * n)], y[: int(0.6 * n)]
    X_val, y_val = X[int(0.6 * n) : int(0.8 * n)], y[int(0.6 * n) : int(0.8 * n)]
    # Test samples clearly in each class
    X_test = np.array([[-0.75], [0.75]] * 5)

    small_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    result = rashomon_stability(
        _finite_c_grid_clf_factory,
        X_train, y_train, X_val, y_val, X_test,
        c_grid=small_grid, delta_bacc=0.05,
    )

    assert result.test_prediction_agreement >= 0.9, (
        f"Expected high agreement on separable data, got {result.test_prediction_agreement:.4f}"
    )


def test_finite_c_grid_uses_additive_best_minus_delta(monkeypatch):
    """Selection uses BAcc(C) >= best BAcc - delta and reports held-out BAcc."""
    validation_bacc = {1.0: 0.90, 2.0: 0.88, 3.0: 0.87}
    test_bacc = {1.0: 0.75, 2.0: 0.50, 3.0: 0.25}

    class GridClassifier:
        def __init__(self, c):
            self.c = c

        def fit(self, X, y):
            return self

        def predict(self, X):
            if len(X) == 3:
                return np.full(len(X), validation_bacc[self.c])
            return np.full(len(X), self.c)

    def fake_balanced_accuracy(y_true, y_pred):
        if len(y_true) == 3:
            return float(y_pred[0])
        return test_bacc[float(y_pred[0])]

    monkeypatch.setattr(
        "fysvm.stability.balanced_accuracy_score", fake_balanced_accuracy
    )
    result = rashomon_stability(
        GridClassifier,
        np.arange(8, dtype=float).reshape(4, 2),
        np.array([0, 0, 1, 1]),
        np.arange(6, dtype=float).reshape(3, 2),
        np.array([0, 1, 1]),
        np.arange(8, dtype=float).reshape(4, 2),
        c_grid=[1.0, 2.0, 3.0],
        delta_bacc=0.021,
        y_test=np.array([0, 0, 1, 1]),
    )

    assert result.near_optimal_set == [1.0, 2.0]
    assert result.near_optimal_size == 2
    assert result.val_balanced_accuracies[2.0] >= 0.90 - result.delta_bacc
    assert result.val_balanced_accuracies[3.0] < 0.90 - result.delta_bacc
    assert result.best_model_test_balanced_accuracy == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Tests: certificate_retention_bootstrap
# ---------------------------------------------------------------------------


def test_certificate_retention_trivially_certified():
    """When all bootstrap models are CERTIFIED, retention rate should be 1.0."""
    # Create data with clear positive monotone feature 0
    rng = np.random.default_rng(5)
    n = 60
    X0 = rng.uniform(-1.0, -0.4, (n // 2, 2))
    X1 = rng.uniform(0.4, 1.0, (n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    # feature 0 clearly separates: higher → class 1 → positive direction
    result = certificate_retention_bootstrap(
        _base_clf_factory,
        X,
        y,
        feature_index=0,
        direction="positive",
        n_bootstrap=15,
        random_state=42,
    )

    assert isinstance(result, CertificateRetentionResult)
    # With clearly separable data the reference model should be CERTIFIED
    # and most / all bootstrap models should match
    assert result.reference_status in (
        "CERTIFIED",
        "CERTIFIED-TRIVIAL",
        "COUNTEREXAMPLE",
        "UNKNOWN",
    )
    # If reference is CERTIFIED, retention should be high
    if result.reference_status == "CERTIFIED":
        assert result.certified_rate >= 0.7, (
            f"Low certified rate {result.certified_rate:.4f} despite CERTIFIED reference"
        )
    assert 0.0 <= result.reference_status_retention_rate <= 1.0
    assert 0.0 <= result.certified_rate <= 1.0
    assert sum(result.status_counts.values()) == result.n_bootstrap
    assert sum(result.status_fractions.values()) == pytest.approx(1.0)
    assert result.certificate_type == "monotonicity"
    assert result.n_bootstrap > 0


def test_certificate_status_aggregation_is_exact(monkeypatch):
    """Reference-status retention is exact while certified status is grouped."""
    statuses = ["CERTIFIED", "CERTIFIED-TRIVIAL", "COUNTEREXAMPLE", "UNKNOWN"]

    class ReferenceClassifier:
        def fit(self, X, y):
            return self

    def fake_bootstrap_certificate(*args, bootstrap_seed, **kwargs):
        return statuses[bootstrap_seed]

    monkeypatch.setattr(
        "fysvm.stability.certificate_monotonicity",
        lambda *args, **kwargs: SimpleNamespace(status="CERTIFIED"),
    )
    monkeypatch.setattr(
        "fysvm.stability._fit_and_certificate_bootstrap",
        fake_bootstrap_certificate,
    )
    result = certificate_retention_bootstrap(
        ReferenceClassifier,
        np.arange(16, dtype=float).reshape(8, 2),
        np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        feature_index=0,
        direction="positive",
        n_bootstrap=4,
        random_state=0,
    )

    assert result.status_counts == {
        "CERTIFIED": 1,
        "CERTIFIED-TRIVIAL": 1,
        "COUNTEREXAMPLE": 1,
        "UNKNOWN": 1,
    }
    assert result.status_fractions == {
        "CERTIFIED": 0.25,
        "CERTIFIED-TRIVIAL": 0.25,
        "COUNTEREXAMPLE": 0.25,
        "UNKNOWN": 0.25,
    }
    assert result.reference_status_retention_rate == pytest.approx(0.25)
    assert result.certified_rate == pytest.approx(0.50)
