# ============================================================
# Stability analyses — Paper Section 4
# Implements: bootstrap prediction stability, finite C-grid
# analysis, and certificate-status retention across bootstrap
# resamples.
# ============================================================
"""Bootstrap prediction stability, finite C-grid analysis, and certificate retention.

Functions
---------
bootstrap_prediction_stability
    Fit B bootstrap-resampled models; measure how often they agree with a reference model.
rashomon_stability
    Fit one model per C in a pre-registered grid; identify the near-optimal validation set
    via validation balanced accuracy and measure prediction agreement within that set.
certificate_retention_bootstrap
    Repeat the monotonicity certificate across bootstrap models; report status rates.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score

from fysvm.certificates import certificate_monotonicity


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BootstrapStabilityResult:
    """Results from a bootstrap prediction stability run."""

    n_bootstrap: int
    """Number of successful bootstrap iterations."""

    mean_prediction_agreement: float
    """Mean of the per-bootstrap prediction agreement rates."""

    std_prediction_agreement: float
    """Standard deviation of the per-bootstrap prediction agreement rates."""

    q05_prediction_agreement: float
    q50_prediction_agreement: float
    q95_prediction_agreement: float

    per_bootstrap_agreement: np.ndarray
    """Fraction of test predictions agreeing with the reference for each bootstrap model."""

    per_sample_agreement: np.ndarray
    """Agreement rate (fraction of bootstrap models agreeing with reference) per test sample."""

    reference_test_balanced_accuracy: float | None
    """Reference-model test balanced accuracy, when test labels are supplied."""

    dataset_name: str
    details: dict


@dataclass
class FiniteCGridResult:
    """Results from a finite C-grid near-optimal validation-set analysis."""

    c_grid: list[float]
    """Full C grid evaluated."""

    val_balanced_accuracies: dict[float, float]
    """Mapping C → balanced accuracy on the validation split."""

    near_optimal_set: list[float]
    """C values within ``delta_bacc`` of the best validation balanced accuracy."""

    delta_bacc: float
    """Additive balanced-accuracy tolerance used to define the near-optimal set."""

    test_prediction_agreement: float
    """Fraction of test samples on which all near-optimal models agree."""

    near_optimal_size: int
    best_model_test_balanced_accuracy: float | None
    """Held-out BAcc of the validation-selected best model, when test labels are supplied."""

    details: dict


@dataclass
class CertificateRetentionResult:
    """Results from running the monotonicity certificate on bootstrap models."""

    n_bootstrap: int
    certificate_type: str
    reference_status: str
    """Certificate status of the reference model (fit on full X_train)."""

    reference_status_retention_rate: float
    """Fraction of bootstrap models with exactly the reference model's status."""

    certified_rate: float
    """Fraction with status ``CERTIFIED`` or ``CERTIFIED-TRIVIAL``."""

    status_counts: dict[str, int]
    status_fractions: dict[str, float]

    details: dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median")


def _fit_and_predict_bootstrap(
    clf_factory: Callable,
    X_boot: np.ndarray,
    y_boot: np.ndarray,
    X_test: np.ndarray,
    bootstrap_seed: int,
) -> np.ndarray | None:
    """Fit one bootstrap model and return predictions on X_test.

    Returns None if the bootstrap resample is degenerate (all one class).
    """
    if len(np.unique(y_boot)) < 2:
        return None
    try:
        imputer = _make_imputer()
        X_boot_imp = imputer.fit_transform(X_boot)
        X_test_imp = imputer.transform(X_test)
        clf = clf_factory()
        # Override random_state for bootstrap diversity if supported
        if hasattr(clf, "random_state"):
            clf.random_state = bootstrap_seed
        clf.fit(X_boot_imp, y_boot)
        return clf.predict(X_test_imp)
    except Exception as exc:
        warnings.warn(f"Bootstrap iteration {bootstrap_seed} failed: {exc}", RuntimeWarning, stacklevel=2)
        return None


def _fit_and_certificate_bootstrap(
    clf_factory: Callable,
    X_boot: np.ndarray,
    y_boot: np.ndarray,
    feature_index: int,
    direction: str,
    bootstrap_seed: int,
) -> str | None:
    """Fit one bootstrap model and return its certificate status.

    Returns None on failure.
    """
    if len(np.unique(y_boot)) < 2:
        return None
    try:
        imputer = _make_imputer()
        X_boot_imp = imputer.fit_transform(X_boot)
        clf = clf_factory()
        if hasattr(clf, "random_state"):
            clf.random_state = bootstrap_seed
        clf.fit(X_boot_imp, y_boot)
        cert = certificate_monotonicity(clf, feature_index, direction)
        return cert.status
    except Exception as exc:
        warnings.warn(
            f"Certificate bootstrap iteration {bootstrap_seed} failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bootstrap_prediction_stability(
    clf_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_bootstrap: int = 200,
    random_state: int = 42,
    dataset_name: str = "",
    y_test: np.ndarray | None = None,
) -> BootstrapStabilityResult:
    """Empirical bootstrap prediction stability.

    Implements the H3.1 analysis (paper §6.1). Fits *n_bootstrap* models on
    bootstrap resamples of (X_train, y_train). For each bootstrap model, computes

        agreement_b = (1/|X_test|) Σ_{x} 1[ŷ_b(x) = ŷ_ref(x)]   [paper Eq. (8)]

    where ŷ_ref is the reference model fit on full X_train. Summary statistics
    are computed across the per-bootstrap values ``agreement_b``. The separate
    per-sample values average the same indicator over bootstrap models.

    NaN values are handled by fitting a fresh :class:`~sklearn.impute.SimpleImputer`
    on each bootstrap resample and applying it to X_test.

    Parameters
    ----------
    clf_factory:
        Callable with no required arguments that returns a fresh, unfitted
        FuzzyRuleSVM.  Called once per bootstrap iteration plus once for the
        reference model.
    X_train, y_train:
        Training data (may contain NaN).
    X_test:
        Test data for evaluating prediction stability (may contain NaN;
        imputed using the same imputer fitted on the training resample).
    n_bootstrap:
        Number of bootstrap resamples.
    random_state:
        Base seed.  Resample *b* uses seed ``random_state + b``.
    dataset_name:
        Stored in the result for identification.
    y_test:
        Optional test labels. When supplied, the reference model's held-out
        balanced accuracy is included in the result.

    Returns
    -------
    BootstrapStabilityResult
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)
    if y_test is not None:
        y_test = np.asarray(y_test)
        if len(y_test) != len(X_test):
            raise ValueError("y_test must have the same number of samples as X_test.")

    # --- Reference model ---
    imp_ref = _make_imputer()
    X_train_ref = imp_ref.fit_transform(X_train)
    X_test_ref = imp_ref.transform(X_test)
    clf_ref = clf_factory()
    clf_ref.fit(X_train_ref, y_train)
    pred_ref = clf_ref.predict(X_test_ref)

    n_train = len(X_train)
    n_test = len(X_test)

    # --- Bootstrap iterations ---
    def _one_bootstrap(b: int) -> np.ndarray | None:
        rng = np.random.default_rng(random_state + b)
        idx = rng.choice(n_train, n_train, replace=True)
        return _fit_and_predict_bootstrap(
            clf_factory, X_train[idx], y_train[idx], X_test, bootstrap_seed=b
        )

    raw_results = Parallel(n_jobs=1)(
        delayed(_one_bootstrap)(b) for b in range(n_bootstrap)
    )

    valid_preds = [r for r in raw_results if r is not None]
    n_valid = len(valid_preds)
    n_failed = n_bootstrap - n_valid

    if n_failed > 0:
        warnings.warn(
            f"{n_failed}/{n_bootstrap} bootstrap iterations failed for '{dataset_name}'.",
            RuntimeWarning,
            stacklevel=2,
        )

    if n_valid == 0:
        raise RuntimeError(
            f"All {n_bootstrap} bootstrap iterations failed for dataset '{dataset_name}'."
        )

    # --- Aggregate ---
    all_preds = np.stack(valid_preds, axis=0)  # (n_valid, n_test)
    agreement = all_preds == pred_ref[np.newaxis, :]
    per_bootstrap_agreement = np.mean(agreement, axis=1)
    per_sample_agreement = np.mean(agreement, axis=0)
    q05, q50, q95 = np.quantile(per_bootstrap_agreement, [0.05, 0.50, 0.95])
    reference_test_bacc = (
        float(balanced_accuracy_score(y_test, pred_ref)) if y_test is not None else None
    )

    return BootstrapStabilityResult(
        n_bootstrap=n_valid,
        mean_prediction_agreement=float(np.mean(per_bootstrap_agreement)),
        std_prediction_agreement=float(np.std(per_bootstrap_agreement)),
        q05_prediction_agreement=float(q05),
        q50_prediction_agreement=float(q50),
        q95_prediction_agreement=float(q95),
        per_bootstrap_agreement=per_bootstrap_agreement,
        per_sample_agreement=per_sample_agreement,
        reference_test_balanced_accuracy=reference_test_bacc,
        dataset_name=dataset_name,
        details={
            "n_failed": n_failed,
            "n_test": n_test,
            "n_train": n_train,
            "random_state": random_state,
        },
    )


def rashomon_stability(
    clf_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    c_grid: list[float] | None = None,
    delta_bacc: float = 0.02,
    dataset_name: str = "",
    y_test: np.ndarray | None = None,
) -> FiniteCGridResult:
    """Evaluate a finite C-grid and its near-optimal validation set.

    Implements the H3.2 analysis (paper §6.2). Fits the classifier at each C in
    *c_grid*. Evaluates balanced accuracy on (X_val, y_val). The near-optimal set:

        N(delta) = {C in C_grid : BAcc_val(C) >= best_BAcc_val - delta_bacc}.

    Then checks how often all near-optimal models agree on X_test.

    NaN values in training and evaluation sets are imputed with a single
    :class:`~sklearn.impute.SimpleImputer` fitted on X_train.

    Parameters
    ----------
    clf_factory:
        ``callable(C: float) -> FuzzyRuleSVM`` with the given C value set.
    X_train, y_train:
        Training data.
    X_val, y_val:
        Validation data for selecting the near-optimal set.
    X_test:
        Test data for measuring prediction agreement.
    c_grid:
        Pre-registered C values to evaluate.
    delta_bacc:
        Additive tolerance in balanced-accuracy units defining the near-optimal
        validation set.
    dataset_name:
        Stored in the result for identification.
    y_test:
        Optional held-out test labels. When supplied, reports the test balanced
        accuracy of the model selected by validation balanced accuracy.

    Returns
    -------
    FiniteCGridResult
    """
    if c_grid is None:
        c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val, dtype=np.float64)
    y_val = np.asarray(y_val)
    X_test = np.asarray(X_test, dtype=np.float64)
    if delta_bacc < 0:
        raise ValueError("delta_bacc must be non-negative.")
    if y_test is not None:
        y_test = np.asarray(y_test)
        if len(y_test) != len(X_test):
            raise ValueError("y_test must have the same number of samples as X_test.")

    # Single imputer fit on training data (shared across all C values)
    imputer = _make_imputer()
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    def _fit_one(c: float):
        clf = clf_factory(c)
        clf.fit(X_train_imp, y_train)
        bacc = balanced_accuracy_score(y_val, clf.predict(X_val_imp))
        test_preds = clf.predict(X_test_imp)
        return c, bacc, test_preds

    outcomes = Parallel(n_jobs=1)(
        delayed(_fit_one)(c) for c in c_grid
    )

    val_baccs: dict[float, float] = {}
    test_preds_map: dict[float, np.ndarray] = {}

    for c, bacc, preds in outcomes:
        val_baccs[c] = bacc
        test_preds_map[c] = preds

    best_bacc = max(val_baccs.values())
    best_c = max(val_baccs, key=lambda c: val_baccs[c])
    near_optimal_set = sorted(
        [c for c, bacc in val_baccs.items() if bacc >= best_bacc - delta_bacc]
    )

    # Prediction agreement on X_test: fraction where all near-optimal models agree.
    if len(near_optimal_set) <= 1:
        # Trivially agree if ≤1 model
        test_prediction_agreement = 1.0
    else:
        near_optimal_preds = np.stack(
            [test_preds_map[c] for c in near_optimal_set], axis=0
        )  # (n_near_optimal, n_test)
        # For each test sample: do all predictions match the first model's prediction?
        agree_mask = np.all(
            near_optimal_preds == near_optimal_preds[0:1, :], axis=0
        )  # (n_test,)
        test_prediction_agreement = float(np.mean(agree_mask))

    best_model_test_bacc = (
        float(balanced_accuracy_score(y_test, test_preds_map[best_c]))
        if y_test is not None
        else None
    )

    return FiniteCGridResult(
        c_grid=list(c_grid),
        val_balanced_accuracies=val_baccs,
        near_optimal_set=near_optimal_set,
        delta_bacc=delta_bacc,
        test_prediction_agreement=test_prediction_agreement,
        near_optimal_size=len(near_optimal_set),
        best_model_test_balanced_accuracy=best_model_test_bacc,
        details={
            "best_bacc": best_bacc,
            "best_c": best_c,
            "dataset_name": dataset_name,
        },
    )


def certificate_retention_bootstrap(
    clf_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_index: int,
    direction: str,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> CertificateRetentionResult:
    """Bootstrap certificate-status retention for the monotonicity certificate.

    Fits *n_bootstrap* bootstrap models.  For each, runs
    :func:`~fysvm.certificates.certificate_monotonicity` and records the status.
    Reports the complete status distribution, the exact reference-status
    retention rate, and the combined certified rate. This is a monotonicity
    secondary analysis; it is not an exclusion-certificate analysis.

    Parameters
    ----------
    clf_factory:
        Callable returning a fresh FuzzyRuleSVM.
    X_train, y_train:
        Training data (may contain NaN).
    feature_index:
        Original (pre-screening) feature index to certify.
    direction:
        ``'positive'`` or ``'negative'``.
    n_bootstrap:
        Number of bootstrap resamples.
    random_state:
        Base random seed.

    Returns
    -------
    CertificateRetentionResult
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)

    # --- Reference model ---
    imp_ref = _make_imputer()
    X_train_ref = imp_ref.fit_transform(X_train)
    clf_ref = clf_factory()
    clf_ref.fit(X_train_ref, y_train)
    ref_cert = certificate_monotonicity(clf_ref, feature_index, direction)
    reference_status = ref_cert.status

    n_train = len(X_train)

    def _one_bootstrap(b: int) -> str | None:
        rng = np.random.default_rng(random_state + b)
        idx = rng.choice(n_train, n_train, replace=True)
        return _fit_and_certificate_bootstrap(
            clf_factory,
            X_train[idx],
            y_train[idx],
            feature_index,
            direction,
            bootstrap_seed=b,
        )

    raw_results = Parallel(n_jobs=1)(
        delayed(_one_bootstrap)(b) for b in range(n_bootstrap)
    )

    valid_statuses = [r for r in raw_results if r is not None]
    n_valid = len(valid_statuses)
    n_failed = n_bootstrap - n_valid

    if n_failed > 0:
        warnings.warn(
            f"{n_failed}/{n_bootstrap} certificate bootstrap iterations failed.",
            RuntimeWarning,
            stacklevel=2,
        )

    if n_valid == 0:
        raise RuntimeError("All certificate bootstrap iterations failed.")

    known_statuses = ("CERTIFIED", "CERTIFIED-TRIVIAL", "COUNTEREXAMPLE", "UNKNOWN")
    status_counts: dict[str, int] = {status: 0 for status in known_statuses}
    for s in valid_statuses:
        status_counts[s] = status_counts.get(s, 0) + 1

    status_fractions = {
        status: count / n_valid for status, count in status_counts.items()
    }
    reference_status_retention_rate = status_counts.get(reference_status, 0) / n_valid
    certified_rate = (
        status_counts["CERTIFIED"] + status_counts["CERTIFIED-TRIVIAL"]
    ) / n_valid

    return CertificateRetentionResult(
        n_bootstrap=n_valid,
        certificate_type="monotonicity",
        reference_status=reference_status,
        reference_status_retention_rate=reference_status_retention_rate,
        certified_rate=certified_rate,
        status_counts=status_counts,
        status_fractions=status_fractions,
        details={
            "feature_index": feature_index,
            "direction": direction,
            "n_failed": n_failed,
            "n_train": n_train,
        },
    )
