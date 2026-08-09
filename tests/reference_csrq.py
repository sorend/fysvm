"""Independent scalar reference implementation for CSRQ-Train mathematics.

Written independently of quotient.py internals, from the mathematical
specification in the proposal docs/proposals-quotient-invariant-fysvm.md.

Used in test_quotient.py and test_csrq.py as an independent oracle.

Key reference implementations:
    - ref_canonical_dimension: D_{d,r} formula
    - ref_expand_rule: NF(phi_R) via scalar iteration
    - ref_canonical_feature_vector: evaluate psi_bar(x)
    - ref_check_ruspini: verify L_j + M_j + H_j = 1 and L_j * H_j = 0
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product as iterproduct
from math import comb

import numpy as np


# ---------------------------------------------------------------------------
# Dimension formulas
# ---------------------------------------------------------------------------

def ref_canonical_dimension(d: int, r: int) -> int:
    """Reference D_{d,r} = sum_{l=0}^{r} C(d,l) * 2^l."""
    return sum(comb(d, l) * (2 ** l) for l in range(r + 1))


def ref_original_grammar_size(d: int, r: int) -> int:
    """Reference N_{d,r} = sum_{l=0}^{r} C(d,l) * 3^l."""
    return sum(comb(d, l) * (3 ** l) for l in range(r + 1))


# ---------------------------------------------------------------------------
# Ruspini check (T1)
# ---------------------------------------------------------------------------

def ref_check_ruspini(a: float, b: float, c: float, x: float) -> dict:
    """Check partition identities L+M+H=1 and L*H=0 for strict anchors.

    Returns {'L+M+H': float, 'L*H': float} for a given x.
    Anchors must satisfy a < b < c (strict).
    """
    assert a < b < c, "Anchors must be strictly ordered."

    def L(v):
        if v <= a:
            return 1.0
        if v >= b:
            return 0.0
        return (b - v) / (b - a)

    def M(v):
        return min(
            max(0.0, (v - a) / (b - a)) if b > a else float(v >= a),
            max(0.0, (c - v) / (c - b)) if c > b else float(v <= c),
        )

    def H(v):
        if v <= b:
            return 0.0
        if v >= c:
            return 1.0
        return (v - b) / (c - b)

    lv = L(x)
    mv = M(x)
    hv = H(x)
    return {"L+M+H": lv + mv + hv, "L*H": lv * hv}


# ---------------------------------------------------------------------------
# Rule expansion reference (NF algorithm)
# ---------------------------------------------------------------------------

def ref_expand_rule(
    conditions: list[tuple[int, str]],
    n_features: int,
    max_degree: int,
) -> dict[tuple[tuple[int, str], ...], int]:
    """Expand a rule into canonical low/high monomials.

    conditions: list of (feature_index, term) where term in {low, medium, high}
    Returns: dict mapping frozenset-like canonical monomial key -> integer coefficient

    Key format: sorted tuple of (feature_index, 'low'/'high') pairs.
    """
    # Validate
    if len(conditions) == 0:
        raise ValueError("Empty rule is reserved for the constant atom.")
    if len(conditions) > max_degree:
        raise ValueError(f"Rule length {len(conditions)} > max_degree {max_degree}.")
    features_seen: set[int] = set()
    for f, t in conditions:
        if f in features_seen:
            raise ValueError(f"Feature {f} repeated.")
        features_seen.add(f)
        if t not in ("low", "medium", "high"):
            raise ValueError(f"Unknown term {t!r}.")

    # Separate fixed and medium conditions
    fixed: list[tuple[int, str]] = []
    medium_features: list[int] = []
    for f, t in conditions:
        if t == "medium":
            medium_features.append(f)
        else:
            fixed.append((f, t))

    # Current state: mapping of sorted (feature, term) tuple -> coefficient
    current: dict[tuple, int] = {tuple(sorted(fixed)): 1}

    # Expand each medium condition: M_j = 1 - L_j - H_j
    for mf in medium_features:
        new: dict[tuple, int] = {}
        for lits, coeff in current.items():
            lits_list = list(lits)

            # Original: (lits, coeff)
            key = tuple(sorted(lits_list))
            new[key] = new.get(key, 0) + coeff

            # -L_j: (lits + (mf, low), -coeff)
            key_low = tuple(sorted(lits_list + [(mf, "low")]))
            new[key_low] = new.get(key_low, 0) - coeff

            # -H_j: (lits + (mf, high), -coeff)
            key_high = tuple(sorted(lits_list + [(mf, "high")]))
            new[key_high] = new.get(key_high, 0) - coeff

        current = {k: v for k, v in new.items() if v != 0}

    return current


# ---------------------------------------------------------------------------
# Canonical feature evaluation reference
# ---------------------------------------------------------------------------

def ref_membership_L(x: float, a: float, b: float) -> float:
    """L_j(x) = linear_down(x, a, b)."""
    if b <= a:
        return float(x <= a)
    return float(np.clip((b - x) / (b - a), 0.0, 1.0))


def ref_membership_H(x: float, b: float, c: float) -> float:
    """H_j(x) = linear_up(x, b, c)."""
    if c <= b:
        return float(x >= c)
    return float(np.clip((x - b) / (c - b), 0.0, 1.0))


def ref_canonical_feature_vector(
    x: np.ndarray,
    anchors: list[tuple[float, float, float]],
    d: int,
    r: int,
) -> np.ndarray:
    """Compute the canonical feature vector psi_bar(x) using reference formulas.

    anchors: list of (a_j, b_j, c_j) for each selected feature.
    Returns: 1D array of length D_{d,r}.
    """
    D = ref_canonical_dimension(d, r)
    psi = np.zeros(D)

    # Build ordered list of monomials (same order as canonical_basis)
    monomials: list[tuple[tuple[tuple[int, str], ...], int]] = []  # (literals, index)
    idx = 0

    # Empty monomial (constant)
    monomials.append(((), idx))
    idx += 1

    for degree in range(1, r + 1):
        for feature_subset in combinations(range(d), degree):
            for term_combo in iterproduct(("low", "high"), repeat=degree):
                lits = tuple(zip(feature_subset, term_combo))
                monomials.append((lits, idx))
                idx += 1

    assert len(monomials) == D

    for lits, i in monomials:
        if len(lits) == 0:
            psi[i] = 1.0
        else:
            val = 1.0
            for f, t in lits:
                a_j, b_j, c_j = anchors[f]
                if t == "low":
                    val *= ref_membership_L(float(x[f]), a_j, b_j)
                else:
                    val *= ref_membership_H(float(x[f]), b_j, c_j)
            psi[i] = val

    return psi


# ---------------------------------------------------------------------------
# Parent/children identity check (T5 corollary)
# ---------------------------------------------------------------------------

def ref_parent_children_identity(
    parent_conditions: list[tuple[int, str]],
    extra_feature: int,
    n_features: int,
    max_degree: int,
) -> dict[str, dict]:
    """Verify phi_C - phi_{C+L_j} - phi_{C+M_j} - phi_{C+H_j} = 0 in NF.

    Returns the expansions of all four rules and the difference.
    The identity holds only when |parent| + 1 <= max_degree.
    """
    exp_parent = ref_expand_rule(parent_conditions, n_features, max_degree)

    conditions_low = parent_conditions + [(extra_feature, "low")]
    conditions_med = parent_conditions + [(extra_feature, "medium")]
    conditions_high = parent_conditions + [(extra_feature, "high")]

    exp_low = ref_expand_rule(conditions_low, n_features, max_degree)
    exp_med = ref_expand_rule(conditions_med, n_features, max_degree)
    exp_high = ref_expand_rule(conditions_high, n_features, max_degree)

    # Compute difference: parent - child_low - child_med - child_high
    all_keys = (
        set(exp_parent)
        | set(exp_low)
        | set(exp_med)
        | set(exp_high)
    )
    diff = {}
    for k in all_keys:
        v = (exp_parent.get(k, 0)
             - exp_low.get(k, 0)
             - exp_med.get(k, 0)
             - exp_high.get(k, 0))
        if v != 0:
            diff[k] = v

    return {
        "parent": exp_parent,
        "child_low": exp_low,
        "child_med": exp_med,
        "child_high": exp_high,
        "difference": diff,
    }
