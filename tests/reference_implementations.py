"""
Independent scalar executable specification for FuzzyRuleSVM core mathematics.

Written from the mathematical specification in README and docstrings,
independently of rule_svm.py implementation internals.

ASSUMPTIONS DOCUMENTED (see also docs/spec_fidelity_checkpoint.md):

A1. Softmin formula: -T * log((1/n) * sum(exp(-v_i/T))).
    VERIFIED against production: production uses a numerically stable
    max-subtraction equivalent that is mathematically identical.
    NO DISCREPANCY found.

A2. Degenerate partitions (start >= end in linear_up/linear_down): treated as
    step functions. linear_up returns 1.0 when v >= end, 0.0 otherwise.
    linear_down returns 1.0 when v <= start, 0.0 otherwise.
    This matches production _linear_up/_linear_down behaviour.

A3. Membership clipping: all membership values are clipped to [0, 1] after
    computation, matching production _concept_membership_tensor.

A4. Medium membership at q_low: linear_up(q_low, q_low, q_mid) = 0.0,
    linear_down(q_low, q_mid, q_high) = 1.0, so min = 0.0.  Correct.
    Medium membership at q_high: linear_up(q_high, q_low, q_mid) = 1.0,
    linear_down(q_high, q_mid, q_high) = 0.0, so min = 0.0.  Correct.

Reference is vectorisable over samples (accepts numpy arrays) and also works
on Python scalars.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Primitive ramp functions
# ---------------------------------------------------------------------------


def linear_up(v, start: float, end: float):
    """Rising ramp: 0 at start, 1 at end, clipped outside.

    Formula: clip((v - start) / (end - start), 0, 1)           [paper §3.1, Eq. (2)]

    Degenerate (end <= start): step — 1.0 for v >= end, else 0.0.
    """
    v = np.asarray(v, dtype=np.float64)
    if end <= start:
        return np.where(v >= end, 1.0, 0.0)
    return np.clip((v - start) / (end - start), 0.0, 1.0)


def linear_down(v, start: float, end: float):
    """Falling ramp: 1 at start, 0 at end, clipped outside.

    Formula: clip((end - v) / (end - start), 0, 1)             [paper §3.1, Eq. (2)]

    Degenerate (end <= start): step — 1.0 for v <= start, else 0.0.
    """
    v = np.asarray(v, dtype=np.float64)
    if end <= start:
        return np.where(v <= start, 1.0, 0.0)
    return np.clip((end - v) / (end - start), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Triangular membership functions
# ---------------------------------------------------------------------------


def triangular_low(v, q_low: float, q_mid: float, q_high: float):
    """Low linguistic term: 1 at q_low, falls to 0 at q_mid, 0 beyond.

    Formula: linear_down(v, q_low, q_mid)                       [paper §3.1, Eq. (2a)]
    """
    return linear_down(v, q_low, q_mid)


def triangular_medium(v, q_low: float, q_mid: float, q_high: float):
    """Medium linguistic term: 0 at q_low, 1 at q_mid, 0 at q_high.

    Formula: min(linear_up(v, q_low, q_mid),                    [paper §3.1, Eq. (2b)]
                 linear_down(v, q_mid, q_high))
    """
    return np.minimum(
        linear_up(v, q_low, q_mid),
        linear_down(v, q_mid, q_high),
    )


def triangular_high(v, q_low: float, q_mid: float, q_high: float):
    """High linguistic term: 0 at q_mid, rises to 1 at q_high, 1 beyond.

    Formula: linear_up(v, q_mid, q_high)                        [paper §3.1, Eq. (2c)]
    """
    return linear_up(v, q_mid, q_high)


# ---------------------------------------------------------------------------
# T-norm aggregation operators
# ---------------------------------------------------------------------------


def min_tnorm(*activations):
    """Min t-norm T_min: element-wise minimum over conjunct activations.

    Formula: min(a_1, ..., a_L)  element-wise.                  [paper §3.2, Eq. (3)]
    """
    stacked = np.stack([np.asarray(a, dtype=np.float64) for a in activations], axis=-1)
    return np.min(stacked, axis=-1)


def product_tnorm(*activations):
    """Product t-norm T_Pi: element-wise product over conjunct activations.

    Formula: prod(a_1, ..., a_L)  element-wise.                 [paper §3.2, Eq. (3)]
    """
    result = np.ones_like(np.asarray(activations[0], dtype=np.float64))
    for a in activations:
        result = result * np.asarray(a, dtype=np.float64)
    return result


def softmin_tnorm(*activations, temperature: float = 0.1):
    """Soft-min operator T_soft via numerically stable log-mean-exp.

    Formula (mathematically):                                    [paper §3.2, Eq. (3)]
      T_soft(a_1,...,a_L) = -T * log((1/L) * sum(exp(-a_i / T)))
    Equivalently: -T * (logsumexp(-a/T) - log(L))

    Uses max-subtraction stabilisation to avoid overflow/underflow,
    identical to production implementation (verified — see module docstring).

    Activation values are clipped to [0, 1] after computation.
    Note: T_soft is outside the sound certificate domain (paper §4.1).
    """
    stacked = np.stack(
        [np.asarray(a, dtype=np.float64) for a in activations], axis=-1
    )  # shape (..., n)
    n = len(activations)
    scaled = -stacked / temperature                                  # -a/T
    max_scaled = np.max(scaled, axis=-1, keepdims=True)              # max per row
    log_mean_exp = np.log(
        np.mean(np.exp(scaled - max_scaled), axis=-1)
    )                                                                # log(mean(exp(-a/T - max)))
    result = -temperature * (log_mean_exp + max_scaled[..., 0])     # = -T*log(mean(exp(-a/T)))
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Decision function
# ---------------------------------------------------------------------------


def decision_function_ref(Z, coef, intercept: float):
    """Linear decision function in rule activation space.

    Formula: f(x) = Z @ coef + intercept                        [paper §3, Eq. (1)]
             i.e., f(x) = sum_k beta_k * phi_k(x) + b

    Parameters
    ----------
    Z : array-like, shape (n, K)
        Rule activation matrix.
    coef : array-like, shape (K,)
        Learned rule coefficients.
    intercept : float
        Bias term.

    Returns
    -------
    np.ndarray, shape (n,)
        Decision function values.
    """
    return np.asarray(Z, dtype=np.float64) @ np.asarray(coef, dtype=np.float64) + float(intercept)


# ---------------------------------------------------------------------------
# Vectorised membership tensor and activation matrix
# ---------------------------------------------------------------------------


def compute_membership_matrix_ref(X, partitions):
    """Compute the (n, d, 3) membership tensor over all features.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Input samples (already projected to the screened feature subset).
    partitions : list of length d
        Each element is either a ``(q_low, q_mid, q_high)`` tuple or an
        object with ``.low``, ``.medium``, ``.high`` attributes (accepts
        production ``_FuzzyPartition`` directly).

    Returns
    -------
    np.ndarray, shape (n, d, 3)
        Clipped membership values.
        Axis 2 order: [μ_low, μ_medium, μ_high].
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    tensor = np.empty((n, d, 3), dtype=np.float64)
    for j, partition in enumerate(partitions):
        if hasattr(partition, "low"):
            q_low, q_mid, q_high = partition.low, partition.medium, partition.high
        else:
            q_low, q_mid, q_high = float(partition[0]), float(partition[1]), float(partition[2])
        v = X[:, j]
        tensor[:, j, 0] = triangular_low(v, q_low, q_mid, q_high)
        tensor[:, j, 1] = triangular_medium(v, q_low, q_mid, q_high)
        tensor[:, j, 2] = triangular_high(v, q_low, q_mid, q_high)
    return np.clip(tensor, 0.0, 1.0)


def compute_activation_matrix_ref(
    membership_tensor,
    rules,
    and_operator: str = "min",
    temperature: float = 0.1,
):
    """Compute the (n, K) rule activation matrix from a membership tensor.

    Parameters
    ----------
    membership_tensor : np.ndarray, shape (n, d, 3)
        Membership tensor from ``compute_membership_matrix_ref``.
    rules : sequence of FuzzyRule
        Rules with ``.conditions`` attribute; each condition has ``.feature``
        (int) and ``.term`` (str in {"low", "medium", "high"}).
    and_operator : {"min", "product", "softmin"}
        T-norm for combining conjunct memberships.
    temperature : float
        Temperature for softmin.

    Returns
    -------
    np.ndarray, shape (n, K)
        Rule activation matrix.
    """
    _term_index = {"low": 0, "medium": 1, "high": 2}
    membership_tensor = np.asarray(membership_tensor, dtype=np.float64)
    n = membership_tensor.shape[0]
    n_rules = len(rules)
    Z = np.empty((n, n_rules), dtype=np.float64)

    for k, rule in enumerate(rules):
        conjuncts = [
            membership_tensor[:, cond.feature, _term_index[cond.term]]
            for cond in rule.conditions
        ]
        if len(conjuncts) == 1:
            Z[:, k] = conjuncts[0]
        elif and_operator == "min":
            Z[:, k] = min_tnorm(*conjuncts)
        elif and_operator == "product":
            Z[:, k] = product_tnorm(*conjuncts)
        else:  # softmin
            Z[:, k] = softmin_tnorm(*conjuncts, temperature=temperature)

    return Z
