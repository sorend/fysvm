# ============================================================
# Property Certification — Paper Section 4
# Implements: sound interval-arithmetic certificates for
# monotonicity, local measurement robustness, structural
# feature exclusion, and low-risk region boxes.
# Certificate outcomes: CERTIFIED / COUNTEREXAMPLE / UNKNOWN.
# Fail-closed when validity-domain conditions are unmet.
# ============================================================
"""Sound interval-arithmetic property certificates for FuzzyRuleSVM.

Property Certification — Four certificate types:
  1. Monotonicity   — decision function is non-decreasing/non-increasing in a feature.
  2. Robustness     — prediction is stable under ±ε perturbation.
  3. Feature exclusion — specified features are absent from every nonzero rule.
  4. Safe region    — model predicts target class throughout a box.

Interval-certificate validity domain (fail-closed):
  • Binary classification
  • and_operator ∈ {"min", "product"}

Structural feature exclusion is operator-independent and also supports softmin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.utils.validation import check_is_fitted

from fysvm.rule_svm import SparseMaxMarginFuzzyRuleMachine


# ---------------------------------------------------------------------------
# Pre-registered annotations (fixed before any model is fitted)
# ---------------------------------------------------------------------------

MONOTONE_FEATURE_ANNOTATIONS: dict[str, dict[str, list[str]]] = {
    "pima_diabetes": {
        "positive": ["Glucose", "BMI", "Age", "Insulin", "DiabetesPedigreeFunction"],
        "negative": [],
    },
    "heart_cleveland": {
        "positive": ["thalach"],       # max heart rate — higher = better prognosis
        "negative": ["trestbps", "chol", "oldpeak", "ca"],  # higher = worse
    },
    "mammographic_mass": {
        "positive": ["Age", "BI-RADS"],  # higher age + BI-RADS = higher malignancy
        "negative": [],
    },
}

# Note: Age appears in both lists for some datasets intentionally.
# Monotonicity checks direction IF the feature is used; exclusion checks whether L1 drops it.
EXCLUSION_FEATURE_ANNOTATIONS: dict[str, list[str]] = {
    "pima_diabetes": ["Age"],          # demographic proxy — L1 may exclude
    "heart_cleveland": ["age", "sex"], # protected attributes
    "mammographic_mass": ["Age"],      # age as proxy
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CertificateResult:
    """Result of a property certificate check."""

    status: str          # "CERTIFIED" | "CERTIFIED-TRIVIAL" | "COUNTEREXAMPLE" | "UNKNOWN"
    certificate_type: str  # "monotonicity" | "robustness" | "exclusion" | "safe_region"
    and_operator: str
    feature_index: int | None
    min_slack: float     # minimum margin from violation (NaN when not applicable)
    counterexample: dict | None   # concrete witness if COUNTEREXAMPLE
    details: dict


# ---------------------------------------------------------------------------
# Internal helpers — piecewise-linear membership arithmetic
# ---------------------------------------------------------------------------

_TERM_TO_INDEX = {"low": 0, "medium": 1, "high": 2}
_TERM_NAMES = ("low", "medium", "high")


def _down(value: float) -> float:
    """Round one binary64 operation toward negative infinity."""
    return float(np.nextafter(float(value), -np.inf))


def _up(value: float) -> float:
    """Round one binary64 operation toward positive infinity."""
    return float(np.nextafter(float(value), np.inf))


def _point_subtract(a: float, b: float) -> tuple[float, float]:
    value = float(a - b)
    return _down(value), _up(value)


def _add_intervals(
    left: tuple[float, float], right: tuple[float, float]
) -> tuple[float, float]:
    return _down(left[0] + right[0]), _up(left[1] + right[1])


def _multiply_intervals(
    left: tuple[float, float], right: tuple[float, float]
) -> tuple[float, float]:
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return _down(min(products)), _up(max(products))


def _divide_positive_intervals(
    numerator: tuple[float, float], denominator: tuple[float, float]
) -> tuple[float, float]:
    if denominator[0] <= 0.0:
        raise ValueError("denominator interval must be strictly positive")
    return _down(numerator[0] / denominator[1]), _up(
        numerator[1] / denominator[0]
    )


def _linear_down(values: np.ndarray, start: float, end: float) -> np.ndarray:
    """μ_low-style: 1 at ≤start, linear decrease to 0 at end, 0 at ≥end."""
    values = np.asarray(values, dtype=np.float64)
    if end <= start:
        return (values <= start).astype(np.float64)
    result = (end - values) / (end - start)
    result = np.where(values <= start, 1.0, result)
    result = np.where(values >= end, 0.0, result)
    return np.clip(result, 0.0, 1.0)


def _linear_up(values: np.ndarray, start: float, end: float) -> np.ndarray:
    """μ_high-style: 0 at ≤start, linear increase to 1 at end, 1 at ≥end."""
    values = np.asarray(values, dtype=np.float64)
    if end <= start:
        return (values >= end).astype(np.float64)
    result = (values - start) / (end - start)
    result = np.where(values <= start, 0.0, result)
    result = np.where(values >= end, 1.0, result)
    return np.clip(result, 0.0, 1.0)


def _eval_membership(term: str, values: np.ndarray, q_low: float, q_mid: float, q_high: float) -> np.ndarray:
    """Evaluate membership for a term at given values."""
    values = np.asarray(values, dtype=np.float64)
    if term == "low":
        return _linear_down(values, q_low, q_mid)
    elif term == "high":
        return _linear_up(values, q_mid, q_high)
    else:  # medium
        return np.minimum(
            _linear_up(values, q_low, q_mid),
            _linear_down(values, q_mid, q_high),
        )


def _linear_down_interval(value: float, start: float, end: float) -> tuple[float, float]:
    if end <= start:
        exact = float(value <= start)
        return exact, exact
    if value <= start:
        return 1.0, 1.0
    if value >= end:
        return 0.0, 0.0
    lower, upper = _divide_positive_intervals(
        _point_subtract(end, value), _point_subtract(end, start)
    )
    return max(0.0, lower), min(1.0, upper)


def _linear_up_interval(value: float, start: float, end: float) -> tuple[float, float]:
    if end <= start:
        exact = float(value >= end)
        return exact, exact
    if value <= start:
        return 0.0, 0.0
    if value >= end:
        return 1.0, 1.0
    lower, upper = _divide_positive_intervals(
        _point_subtract(value, start), _point_subtract(end, start)
    )
    return max(0.0, lower), min(1.0, upper)


def _membership_value_interval(
    term: str, value: float, q_low: float, q_mid: float, q_high: float
) -> tuple[float, float]:
    if term == "low":
        return _linear_down_interval(value, q_low, q_mid)
    if term == "high":
        return _linear_up_interval(value, q_mid, q_high)
    up = _linear_up_interval(value, q_low, q_mid)
    down = _linear_down_interval(value, q_mid, q_high)
    return min(up[0], down[0]), min(up[1], down[1])


def _membership_bounds(
    term: str, a: float, b: float, q_low: float, q_mid: float, q_high: float
) -> tuple[float, float]:
    """Compute tight (lo, hi) bounds on μ_term(v) for v ∈ [a, b].

    Evaluates at endpoints plus all breakpoints that fall inside [a, b].
    This is exact because each membership function is piecewise linear.
    """
    candidates = [a, b]
    for bp in (q_low, q_mid, q_high):
        if a < bp < b:
            candidates.append(bp)
    intervals = [
        _membership_value_interval(term, point, q_low, q_mid, q_high)
        for point in candidates
    ]
    return min(interval[0] for interval in intervals), max(
        interval[1] for interval in intervals
    )


def _rule_firing_bounds(
    clf: SparseMaxMarginFuzzyRuleMachine,
    rule_idx: int,
    box: dict[int, tuple[float, float]],
) -> tuple[float, float]:
    """Compute (lb_phi, ub_phi) for rule rule_idx over a box.

    box maps internal feature_index → (lo, hi).
    Features not in box are treated as unconstrained → membership ∈ [0, 1].

    For min t-norm:
        lb_phi = min_j lb_j       (exact: independent minimum)
        ub_phi = min_j ub_j       (sound upper bound: min_j max_j ≥ max min)

    For product t-norm:
        lb_phi = ∏_j lb_j         (exact: non-negative independent factors)
        ub_phi = ∏_j ub_j         (exact: non-negative independent factors)
    """
    rule = clf.rules_[rule_idx]
    and_op = clf.and_operator

    per_feature_lbs = []
    per_feature_ubs = []

    for cond in rule.conditions:
        j = cond.feature
        term = cond.term
        p = clf.partitions_[j]
        q_low, q_mid, q_high = p.low, p.medium, p.high

        if j in box:
            a, b = box[j]
            lb_j, ub_j = _membership_bounds(term, a, b, q_low, q_mid, q_high)
        else:
            # Unconstrained feature: membership ∈ [0, 1] always achievable.
            lb_j, ub_j = 0.0, 1.0

        per_feature_lbs.append(lb_j)
        per_feature_ubs.append(ub_j)

    if and_op == "min":
        lb_phi = float(min(per_feature_lbs))
        # Sound upper bound: min_j(max_j μ_j) ≥ max_x min_j μ_j(x_j) (proven in design doc)
        ub_phi = float(min(per_feature_ubs))
    else:  # product
        product_interval = (1.0, 1.0)
        for lb_j, ub_j in zip(per_feature_lbs, per_feature_ubs):
            product_interval = _multiply_intervals(product_interval, (lb_j, ub_j))
        lb_phi, ub_phi = product_interval

    return lb_phi, ub_phi


def _interval_f(
    clf: SparseMaxMarginFuzzyRuleMachine,
    box: dict[int, tuple[float, float]],
) -> tuple[float, float]:
    """Compute [lb_f, ub_f] of f(x) = Σ β_k φ_k(x) + b over a box.

    lb_f and ub_f are sound bounds (lb_f ≤ f(x) ≤ ub_f for all x in box).
    """
    coef = clf.coef_
    intercept = clf.intercept_

    decision_interval = (float(intercept), float(intercept))

    for k in range(len(clf.rules_)):
        if coef[k] == 0.0:
            continue
        lb_phi, ub_phi = _rule_firing_bounds(clf, k, box)
        beta_interval = (float(coef[k]), float(coef[k]))
        contribution = _multiply_intervals(beta_interval, (lb_phi, ub_phi))
        decision_interval = _add_intervals(decision_interval, contribution)

    return decision_interval


# ---------------------------------------------------------------------------
# Validity domain check
# ---------------------------------------------------------------------------

def check_validity_domain(clf: SparseMaxMarginFuzzyRuleMachine) -> bool:
    """Return whether ``clf`` supports interval-arithmetic certificates.

    Requirements:
      - Fitted (has classes_)
      - Binary classification (exactly 2 classes)
      - and_operator ∈ {"min", "product"}  (softmin is not supported)

    Individual certificate functions may impose additional requirements. In
    particular, monotonicity fails closed for a structurally used feature whose
    partition has tied anchors.
    """
    try:
        check_is_fitted(clf)
    except Exception:
        return False

    if len(clf.classes_) != 2:
        return False

    if clf.and_operator not in {"min", "product"}:
        return False

    return True


def _is_fitted_binary(clf: SparseMaxMarginFuzzyRuleMachine) -> bool:
    """Return whether the estimator is fitted for binary classification."""
    try:
        check_is_fitted(clf)
    except Exception:
        return False
    return len(clf.classes_) == 2


def _validate_feature_index(feature_index: int, n_features: int) -> int:
    """Validate and return one internal feature index."""
    if (
        isinstance(feature_index, (bool, np.bool_))
        or not isinstance(feature_index, (int, np.integer))
        or not 0 <= int(feature_index) < n_features
    ):
        raise ValueError(
            f"feature index must be an integer in [0, {n_features}), got {feature_index!r}."
        )
    return int(feature_index)


def _validate_feature_indices(
    feature_indices: list[int], n_features: int
) -> list[int]:
    """Validate and return internal feature indices."""
    try:
        return [_validate_feature_index(index, n_features) for index in feature_indices]
    except TypeError as exc:
        raise ValueError("feature_indices must be an iterable of feature indices.") from exc


# ---------------------------------------------------------------------------
# Certificate 1 — Monotonicity
# ---------------------------------------------------------------------------

def certificate_monotonicity(
    clf: SparseMaxMarginFuzzyRuleMachine,
    feature_index: int,
    direction: str = "positive",
) -> CertificateResult:
    """Soundly certify monotonicity in one internal feature.

    On each nondegenerate membership segment, a length-one rule contributes the
    exact derivative ``c = beta * slope``. A multi-condition min or product rule
    contributes the conservative interval ``[min(0, c), max(0, c)]`` because its
    other antecedents can suppress or scale the focal derivative. Per-rule
    intervals are summed. Positive monotonicity is certified only when every
    aggregate lower bound is nonnegative; negative monotonicity is certified only
    when every aggregate upper bound is nonpositive.

    A feature absent from every exactly nonzero rule is ``CERTIFIED-TRIVIAL``.
    Tied focal partition anchors return ``UNKNOWN`` when the feature is used.
    Inconclusive derivative bounds return ``UNKNOWN`` without a counterexample.
    Invalid feature indices or directions raise ``ValueError``.
    """
    and_op = getattr(clf, "and_operator", "unknown")

    if direction not in {"positive", "negative"}:
        raise ValueError("direction must be 'positive' or 'negative'.")

    if not _is_fitted_binary(clf):
        return CertificateResult(
            status="UNKNOWN",
            certificate_type="monotonicity",
            and_operator=and_op,
            feature_index=feature_index,
            min_slack=float("nan"),
            counterexample=None,
            details={"reason": "validity domain check failed"},
        )

    n_features = len(clf.partitions_)
    feature_index = _validate_feature_index(feature_index, n_features)

    if and_op not in {"min", "product"}:
        return CertificateResult(
            status="UNKNOWN",
            certificate_type="monotonicity",
            and_operator=and_op,
            feature_index=feature_index,
            min_slack=float("nan"),
            counterexample=None,
            details={"reason": "validity domain check failed"},
        )

    feature_rules: list[tuple[int, int]] = []
    for k, beta in enumerate(clf.coef_):
        if beta == 0.0:
            continue
        for cond in clf.rules_[k].conditions:
            if cond.feature == feature_index:
                feature_rules.append((k, _TERM_TO_INDEX[cond.term]))
                break

    if not feature_rules:
        return CertificateResult(
            status="CERTIFIED-TRIVIAL",
            certificate_type="monotonicity",
            and_operator=and_op,
            feature_index=feature_index,
            min_slack=0.0,
            counterexample=None,
            details={"reason": "feature is structurally absent from all nonzero rules"},
        )

    p = clf.partitions_[feature_index]
    q_low, q_mid, q_high = p.low, p.medium, p.high
    if q_low == q_mid or q_mid == q_high:
        return CertificateResult(
            status="UNKNOWN",
            certificate_type="monotonicity",
            and_operator=and_op,
            feature_index=feature_index,
            min_slack=float("nan"),
            counterexample=None,
            details={
                "reason": "tied partition",
                "breakpoints": [q_low, q_mid, q_high],
            },
        )

    dql = _point_subtract(q_mid, q_low)
    dqh = _point_subtract(q_high, q_mid)
    mag_ql = _divide_positive_intervals((1.0, 1.0), dql)
    mag_qh = _divide_positive_intervals((1.0, 1.0), dqh)

    # Directed slope intervals on the two nondegenerate focal segments.
    zero = (0.0, 0.0)
    slopes_by_interval = [
        [(-mag_ql[1], -mag_ql[0]), mag_ql, zero],
        [zero, (-mag_qh[1], -mag_qh[0]), mag_qh],
    ]

    derivative_intervals = [(0.0, 0.0), (0.0, 0.0)]
    for k, term_idx in feature_rules:
        beta_interval = (float(clf.coef_[k]), float(clf.coef_[k]))
        for interval_index, slopes in enumerate(slopes_by_interval):
            slope_interval = slopes[term_idx]
            if slope_interval == zero:
                continue
            contribution = _multiply_intervals(beta_interval, slope_interval)
            if clf.rules_[k].length == 1:
                bounded_contribution = contribution
            else:
                bounded_contribution = (
                    min(0.0, contribution[0]),
                    max(0.0, contribution[1]),
                )
            derivative_intervals[interval_index] = _add_intervals(
                derivative_intervals[interval_index], bounded_contribution
            )

    derivative_lower = np.array([interval[0] for interval in derivative_intervals])
    derivative_upper = np.array([interval[1] for interval in derivative_intervals])

    if direction == "positive":
        certified = bool(np.all(derivative_lower >= 0.0))
        min_slack = float(derivative_lower.min())
    else:
        certified = bool(np.all(derivative_upper <= 0.0))
        min_slack = float(-derivative_upper.max())

    return CertificateResult(
        status="CERTIFIED" if certified else "UNKNOWN",
        certificate_type="monotonicity",
        and_operator=and_op,
        feature_index=feature_index,
        min_slack=min_slack,
        counterexample=None,
        details={
            "derivative_lower_by_segment": derivative_lower.tolist(),
            "derivative_upper_by_segment": derivative_upper.tolist(),
            "breakpoints": [q_low, q_mid, q_high],
            **({} if certified else {"reason": "derivative bounds are inconclusive"}),
        },
    )


# ---------------------------------------------------------------------------
# Certificate 2 — Robustness
# ---------------------------------------------------------------------------

def certificate_robustness(
    clf: SparseMaxMarginFuzzyRuleMachine,
    x: np.ndarray,
    epsilon: float | np.ndarray,
    feature_indices: list[int] | None = None,
) -> CertificateResult:
    """Check prediction stability under ±ε perturbation of specified features.

    Implements Claim R (paper §5.3). Computes [ℓ_f(ε), u_f(ε)] via sound interval
    arithmetic over the perturbed box x ± ε:

        ℓ_f(ε) = Σ_{β_k>0} β_k · min_{x'∈box} φ_k(x')           [paper Eq. (6)]
                + Σ_{β_k<0} β_k · max_{x'∈box} φ_k(x') + b

    CERTIFIED  iff the interval does not straddle 0 (sign is preserved).
    UNKNOWN    iff the interval straddles 0.

    ``epsilon`` may be a finite nonnegative scalar or an array with one value per
    internal feature. ``x`` must be a finite vector with exactly that many values.
    ``feature_indices`` selects which internal features to perturb; ``None`` means
    all. Invalid public inputs raise ``ValueError``.
    """
    and_op = getattr(clf, "and_operator", "unknown")

    if not _is_fitted_binary(clf):
        return CertificateResult(
            status="UNKNOWN",
            certificate_type="robustness",
            and_operator=and_op,
            feature_index=None,
            min_slack=float("nan"),
            counterexample=None,
            details={"reason": "validity domain check failed"},
        )

    n_feat = len(clf.partitions_)

    try:
        x = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("x must be a finite numeric feature vector.") from exc
    if x.shape != (n_feat,) or not np.all(np.isfinite(x)):
        raise ValueError(f"x must be finite with shape ({n_feat},), got {x.shape}.")

    try:
        epsilon_input = np.asarray(epsilon, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("epsilon must be finite and nonnegative.") from exc
    if epsilon_input.ndim == 0:
        eps_arr = np.full(n_feat, float(epsilon_input), dtype=np.float64)
    elif epsilon_input.shape == (n_feat,):
        eps_arr = epsilon_input.copy()
    else:
        raise ValueError(
            f"epsilon must be a scalar or have shape ({n_feat},), got {epsilon_input.shape}."
        )
    if not np.all(np.isfinite(eps_arr)) or np.any(eps_arr < 0.0):
        raise ValueError("epsilon must contain only finite nonnegative values.")

    # Zero out epsilon for features not being perturbed
    if feature_indices is not None:
        feature_indices = _validate_feature_indices(feature_indices, n_feat)
        mask = np.zeros(n_feat, dtype=bool)
        for j in feature_indices:
            mask[j] = True
        eps_arr[~mask] = 0.0

    if and_op not in {"min", "product"}:
        return CertificateResult(
            status="UNKNOWN",
            certificate_type="robustness",
            and_operator=and_op,
            feature_index=None,
            min_slack=float("nan"),
            counterexample=None,
            details={"reason": "validity domain check failed"},
        )

    if not bool(np.any(eps_arr > 0.0)):
        f_x = float(
            np.dot(clf.coef_, clf.transform(x.reshape(1, -1))[0]) + clf.intercept_
        )
        return CertificateResult(
            status="CERTIFIED-TRIVIAL",
            certificate_type="robustness",
            and_operator=and_op,
            feature_index=None,
            min_slack=abs(f_x),
            counterexample=None,
            details={
                "lb_f": f_x,
                "ub_f": f_x,
                "f_x": f_x,
                "epsilon": eps_arr.tolist(),
                "reason": "zero perturbation vector",
            },
        )

    # Build box: feature_j → [x_j − ε_j, x_j + ε_j]
    box: dict[int, tuple[float, float]] = {}
    for j in range(n_feat):
        lo = _point_subtract(float(x[j]), float(eps_arr[j]))[0]
        hi = _add_intervals(
            (float(x[j]), float(x[j])),
            (float(eps_arr[j]), float(eps_arr[j])),
        )[1]
        box[j] = (lo, hi)

    lb_f, ub_f = _interval_f(clf, box)

    # Current prediction
    f_x = float(np.dot(clf.coef_, clf.transform(x.reshape(1, -1))[0]) + clf.intercept_)
    predicted_positive = f_x >= 0.0

    if predicted_positive:
        min_slack = lb_f   # CERTIFIED if lb_f > 0
        if lb_f > 0.0:
            status = "CERTIFIED"
        else:
            status = "UNKNOWN"
    else:
        min_slack = -ub_f  # CERTIFIED if ub_f < 0
        if ub_f < 0.0:
            status = "CERTIFIED"
        else:
            status = "UNKNOWN"

    return CertificateResult(
        status=status,
        certificate_type="robustness",
        and_operator=and_op,
        feature_index=None,
        min_slack=float(min_slack),
        counterexample=None,
        details={
            "lb_f": float(lb_f),
            "ub_f": float(ub_f),
            "f_x": float(f_x),
            "epsilon": eps_arr.tolist(),
            "reason": None,
        },
    )


# ---------------------------------------------------------------------------
# Certificate 3 — Feature exclusion
# ---------------------------------------------------------------------------

def certificate_feature_exclusion(
    clf: SparseMaxMarginFuzzyRuleMachine,
    feature_indices: list[int],
) -> CertificateResult:
    """Check structural non-reference of specified internal features.

    A feature is structurally excluded exactly when no rule with coefficient
    ``beta != 0.0`` references it. This property is deliberately not described as
    zero net influence: cancellation between referencing rules does not establish
    structural exclusion. The exact check is independent of interval arithmetic,
    so every fitted binary model is supported, including softmin. Invalid feature
    indices raise ``ValueError``.

    ``CERTIFIED`` means no nonzero rule references a forbidden feature.
    ``COUNTEREXAMPLE`` includes the nonzero referencing rules as explicit witnesses.
    ``UNKNOWN`` means the estimator is not a fitted binary model.
    """
    and_op = getattr(clf, "and_operator", "unknown")

    if not _is_fitted_binary(clf):
        return CertificateResult(
            status="UNKNOWN",
            certificate_type="exclusion",
            and_operator=and_op,
            feature_index=None,
            min_slack=float("nan"),
            counterexample=None,
            details={"reason": "validity domain check failed"},
        )

    feature_indices = _validate_feature_indices(feature_indices, len(clf.partitions_))
    forbidden = set(feature_indices)
    violating_rules: list[dict] = []

    nonzero_rule_indices = np.flatnonzero(clf.coef_ != 0.0)
    for k in nonzero_rule_indices:
        rule = clf.rules_[k]
        for cond in rule.conditions:
            if cond.feature in forbidden:
                violating_rules.append({
                    "rule_index": int(k),
                    "feature_index": cond.feature,
                    "feature_name": str(clf.feature_names_in_[cond.feature]),
                    "term": cond.term,
                    "beta": float(clf.coef_[k]),
                })
                break  # one violation per rule is enough

    if not violating_rules:
        return CertificateResult(
            status="CERTIFIED",
            certificate_type="exclusion",
            and_operator=and_op,
            feature_index=None,
            min_slack=0.0,
            counterexample=None,
            details={
                "n_nonzero_rules": int(len(nonzero_rule_indices)),
                "forbidden_features": list(feature_indices),
                "property": "structural non-reference",
            },
        )
    else:
        return CertificateResult(
            status="COUNTEREXAMPLE",
            certificate_type="exclusion",
            and_operator=and_op,
            feature_index=None,
            min_slack=float("nan"),
            counterexample={"violating_rules": violating_rules},
            details={
                "n_nonzero_rules": int(len(nonzero_rule_indices)),
                "forbidden_features": list(feature_indices),
                "n_violations": len(violating_rules),
                "property": "structural non-reference",
            },
        )


# ---------------------------------------------------------------------------
# Certificate 4 — Safe region
# ---------------------------------------------------------------------------

def certificate_safe_region(
    clf: SparseMaxMarginFuzzyRuleMachine,
    box: dict[int, tuple[float, float]],
    target_class_index: int = 0,
) -> CertificateResult:
    """Check if the model predicts target_class throughout a specified box.

    Implements Claim S (paper §5.5). Computes a sound interval upper bound u_f(R):

        u_f(R) = Σ_{β_k>0} β_k · max_{x∈R} φ_k(x)               [paper Eq. (7)]
               + Σ_{β_k<0} β_k · min_{x∈R} φ_k(x) + b

    box: {internal_feature_index: (lo, hi)} — finite input domain box
    target_class_index:
        0 → classes_[0] (f < 0), certified iff ub_f < 0
        1 → classes_[1] (f ≥ 0), certified iff lb_f > 0

    Uses interval arithmetic for a sound sufficient condition.
    CERTIFIED  — guaranteed prediction throughout box.
    UNKNOWN    — interval does not permit sound conclusion.

    Invalid box indices/bounds or target class indices raise ``ValueError``.
    """
    and_op = getattr(clf, "and_operator", "unknown")

    if (
        isinstance(target_class_index, (bool, np.bool_))
        or not isinstance(target_class_index, (int, np.integer))
        or int(target_class_index) not in {0, 1}
    ):
        raise ValueError("target_class_index must be 0 or 1.")
    target_class_index = int(target_class_index)

    if not _is_fitted_binary(clf):
        return CertificateResult(
            status="UNKNOWN",
            certificate_type="safe_region",
            and_operator=and_op,
            feature_index=None,
            min_slack=float("nan"),
            counterexample=None,
            details={"reason": "validity domain check failed"},
        )

    n_features = len(clf.partitions_)
    if not hasattr(box, "items"):
        raise ValueError("box must map feature indices to (lo, hi) bounds.")
    checked_box: dict[int, tuple[float, float]] = {}
    for raw_index, raw_bounds in box.items():
        index = _validate_feature_index(raw_index, n_features)
        try:
            bounds = np.asarray(raw_bounds, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("each box bound must be a finite (lo, hi) pair.") from exc
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
            raise ValueError("each box bound must be a finite (lo, hi) pair.")
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo > hi:
            raise ValueError(f"box lower bound must not exceed upper bound for feature {index}.")
        checked_box[index] = (lo, hi)

    if and_op not in {"min", "product"}:
        return CertificateResult(
            status="UNKNOWN",
            certificate_type="safe_region",
            and_operator=and_op,
            feature_index=None,
            min_slack=float("nan"),
            counterexample=None,
            details={"reason": "validity domain check failed"},
        )

    lb_f, ub_f = _interval_f(clf, checked_box)

    if target_class_index == 0:
        # Want f(x) < 0 everywhere in box → certified iff ub_f < 0
        min_slack = float(-ub_f)   # positive = margin below zero
        if ub_f < 0.0:
            status = "CERTIFIED"
        else:
            status = "UNKNOWN"
    else:
        # Want f(x) ≥ 0 everywhere in box → certified iff lb_f > 0
        min_slack = float(lb_f)
        if lb_f > 0.0:
            status = "CERTIFIED"
        else:
            status = "UNKNOWN"

    return CertificateResult(
        status=status,
        certificate_type="safe_region",
        and_operator=and_op,
        feature_index=None,
        min_slack=min_slack,
        counterexample=None,
        details={
            "lb_f": float(lb_f),
            "ub_f": float(ub_f),
            "target_class": str(clf.classes_[target_class_index]),
            "target_class_index": target_class_index,
            "box_feature_count": len(checked_box),
        },
    )


# ---------------------------------------------------------------------------
# Utility: fuzzy feature name matching
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Lowercase and strip underscores/hyphens/spaces for fuzzy matching."""
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def find_feature_index(clf: SparseMaxMarginFuzzyRuleMachine, feature_name: str) -> int | None:
    """Find internal feature index matching feature_name via case-insensitive substring match.

    Normalises both names (lowercase, strip underscores/hyphens) then checks
    bidirectional substring: annotation ⊆ dataset-name OR dataset-name ⊆ annotation.

    To avoid false-positive short matches (e.g., "thal" matching "thalach"),
    the matching token must be at least 5 characters long.  For shorter query
    strings an exact match is required instead.

    Returns None if no match is found.
    """
    query = _normalize_name(feature_name)
    for i, fn in enumerate(clf.feature_names_in_):
        target = _normalize_name(str(fn))
        if len(query) >= 5 and len(target) >= 5:
            if query in target or target in query:
                return i
        else:
            # Short names: require exact match
            if query == target:
                return i
    return None
