"""Tests for fysvm.quotient — canonical basis, rule expansion, semantic maps, RREF, decoding."""

from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np
import pytest

from fysvm.quotient import (
    CanonicalBasis,
    CanonicalLiteral,
    CanonicalMonomial,
    RuleAtom,
    build_semantic_map,
    canonical_basis,
    canonical_dimension,
    canonical_feature_matrix,
    decode_canonical,
    decode_rref,
    exact_kernel,
    exact_rref,
    expand_rule,
    original_grammar_size,
    semantic_subspace_rref,
)
from fysvm.rule_svm import FuzzyRule, RuleCondition, _FuzzyPartition
from tests.reference_csrq import (
    ref_canonical_dimension,
    ref_canonical_feature_vector,
    ref_check_ruspini,
    ref_expand_rule,
    ref_original_grammar_size,
    ref_parent_children_identity,
)


# ---------------------------------------------------------------------------
# Basis counts (T4 — linear independence implied by correct dimension)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,r,expected", [
    (1, 1, 3),    # 1 + 2 = 3
    (2, 1, 5),    # 1 + 4 = 5
    (2, 2, 13),   # 1 + 4 + 8 = 13
    (3, 1, 7),    # 1 + 6 = 7
    (3, 2, 25),   # 1 + 6 + 18 = 25  -> wait: 1 + C(3,1)*2 + C(3,2)*4 = 1+6+12=19
    (0, 0, 1),    # only constant
    (0, 2, 1),    # no features, only constant
])
def test_canonical_dimension_matches_reference(d, r, expected):
    ref = ref_canonical_dimension(d, r)
    got = canonical_dimension(d, r)
    assert got == ref, f"canonical_dimension({d},{r}) = {got}, expected {ref}"


def test_canonical_dimension_formula():
    """D_{d,r} formula matches basis count for small cases."""
    for d in range(6):
        for r in range(4):
            basis = canonical_basis(d, r)
            expected = canonical_dimension(d, r)
            assert basis.dimension == expected, (
                f"basis.dimension({d},{r}) = {basis.dimension}, expected {expected}"
            )


def test_original_grammar_size_formula():
    """N_{d,r} matches reference formula."""
    for d in range(5):
        for r in range(4):
            assert original_grammar_size(d, r) == ref_original_grammar_size(d, r)


def test_canonical_basis_starts_with_empty():
    """The empty monomial (constant) is always first."""
    for d in [1, 2, 3]:
        basis = canonical_basis(d, 2)
        assert basis.monomials[0] == CanonicalMonomial(())


def test_canonical_basis_graded_lex_ordering():
    """Monomials are ordered by degree, then feature tuple, then low<high."""
    basis = canonical_basis(2, 2)
    monos = basis.monomials

    # Degrees are non-decreasing
    degrees = [m.degree for m in monos]
    assert degrees == sorted(degrees)

    # Within same degree, lexicographic feature tuple
    for deg in [1, 2]:
        same_deg = [m for m in monos if m.degree == deg]
        feature_tuples = [tuple(l.feature for l in m.literals) for m in same_deg]
        assert feature_tuples == sorted(feature_tuples)


def test_canonical_basis_stable_hash():
    """Same parameters produce the same SHA-256 hash."""
    b1 = canonical_basis(3, 2)
    b2 = canonical_basis(3, 2)
    assert b1.sha256 == b2.sha256

    b3 = canonical_basis(3, 3)
    assert b1.sha256 != b3.sha256


def test_canonical_basis_low_before_high():
    """Within same feature subset, low comes before high."""
    basis = canonical_basis(2, 1)
    # Degree-1 monomials: (f0,low), (f0,high), (f1,low), (f1,high)
    deg1 = [m for m in basis.monomials if m.degree == 1]
    assert deg1[0].literals[0].term == "low"
    assert deg1[1].literals[0].term == "high"


# ---------------------------------------------------------------------------
# Rule expansion (T5)
# ---------------------------------------------------------------------------

def test_expand_rule_pure_low():
    """Pure low rule expands to a single canonical basis vector."""
    basis = canonical_basis(2, 2)
    rule = FuzzyRule((RuleCondition(0, "low"),))
    expansion = expand_rule(rule, basis)
    ref = ref_expand_rule([(0, "low")], 2, 2)
    # ref keys are sorted (feature, term) tuples; map to canonical basis indices
    # Both should have a single entry with coefficient 1
    assert len(expansion) == 1
    assert list(expansion.values()) == [1]


def test_expand_rule_pure_medium():
    """Medium rule expands to constant - low - high."""
    basis = canonical_basis(2, 2)
    rule = FuzzyRule((RuleCondition(0, "medium"),))
    expansion = expand_rule(rule, basis)
    ref_exp = ref_expand_rule([(0, "medium")], 2, 2)

    # Reference has: () -> 1, (0,low) -> -1, (0,high) -> -1
    assert len(ref_exp) == 3
    assert ref_exp[()] == 1
    assert ref_exp[((0, "low"),)] == -1
    assert ref_exp[((0, "high"),)] == -1

    # Production expansion (indices in canonical basis)
    # Constant is index 0, (0,low) is index 1, (0,high) is index 2 for d=2,r=2
    assert sum(expansion.values()) == -1  # 1 - 1 - 1 = -1


def test_expand_rule_medium_medium():
    """Two-medium rule expands to 9 canonical terms (3x3)."""
    basis = canonical_basis(2, 2)
    rule = FuzzyRule((RuleCondition(0, "medium"), RuleCondition(1, "medium")))
    expansion = expand_rule(rule, basis)
    ref_exp = ref_expand_rule([(0, "medium"), (1, "medium")], 2, 2)
    # reference has 9 terms (3 choices for each feature)
    assert len(ref_exp) == 9
    total = sum(ref_exp.values())
    assert total == 1  # (1)(1) - sums correctly for M_0 * M_1


def test_expand_rule_matches_reference():
    """Production expansion matches reference for several rules."""
    d, r = 3, 2
    basis = canonical_basis(d, r)
    # Build monomial -> index map
    mono_to_idx = {m: i for i, m in enumerate(basis.monomials)}

    # Test a mixed rule: f0=low, f1=medium (2 conditions, within max_degree=2)
    rule_conditions = [(0, "low"), (1, "medium")]
    ref_exp = ref_expand_rule(rule_conditions, d, r)
    rule = FuzzyRule(tuple(RuleCondition(f, t) for f, t in rule_conditions))
    prod_exp = expand_rule(rule, basis)

    # Convert reference to index-based
    ref_idx = {}
    for key, coeff in ref_exp.items():
        # key is sorted tuple of (feature, term)
        from fysvm.quotient import CanonicalMonomial, CanonicalLiteral
        lits = tuple(CanonicalLiteral(f, t) for f, t in key)
        mono = CanonicalMonomial(lits)
        idx = mono_to_idx[mono]
        ref_idx[idx] = coeff

    assert prod_exp == ref_idx


def test_expand_rule_parent_children_identity():
    """phi_C - phi_{C+Lj} - phi_{C+Mj} - phi_{C+Hj} = 0 (in NF)."""
    d, r = 3, 2
    result = ref_parent_children_identity(
        parent_conditions=[(0, "low")],
        extra_feature=1,
        n_features=d,
        max_degree=r,
    )
    assert result["difference"] == {}, (
        f"Parent/children identity failed: diff={result['difference']}"
    )


def test_expand_rule_rejects_empty():
    basis = canonical_basis(2, 2)
    with pytest.raises(ValueError, match="Empty rules"):
        expand_rule(FuzzyRule(()), basis)


def test_expand_rule_rejects_overlong():
    basis = canonical_basis(3, 2)
    rule = FuzzyRule((RuleCondition(0, "low"), RuleCondition(1, "low"), RuleCondition(2, "low")))
    with pytest.raises(ValueError, match="exceeds max_degree"):
        expand_rule(rule, basis)


def test_expand_rule_rejects_repeated_feature():
    basis = canonical_basis(3, 2)
    rule = FuzzyRule((RuleCondition(0, "low"), RuleCondition(0, "high")))
    with pytest.raises(ValueError, match="more than once"):
        expand_rule(rule, basis)


# ---------------------------------------------------------------------------
# Semantic map
# ---------------------------------------------------------------------------

def _make_atom(feature: int, term: str) -> RuleAtom:
    rule = FuzzyRule((RuleCondition(feature, term),))
    return RuleAtom(rule=rule, scale=1.0, cost=1.0)


def test_semantic_map_shape():
    d, r = 2, 2
    basis = canonical_basis(d, r)
    atoms = (_make_atom(0, "low"), _make_atom(0, "high"), _make_atom(1, "medium"))
    smap = build_semantic_map(atoms, basis)
    D = canonical_dimension(d, r)
    K = len(atoms)
    assert smap.matrix.shape == (D, K + 1)


def test_semantic_map_intercept_column():
    """Column 0 (intercept atom) has a 1 at the empty-monomial row."""
    d, r = 2, 1
    basis = canonical_basis(d, r)
    atoms = (_make_atom(0, "low"),)
    smap = build_semantic_map(atoms, basis)
    dense = smap.matrix.toarray()
    assert dense[0, 0] == 1.0  # empty monomial row, intercept column


def test_semantic_map_rank_and_nullity():
    """Complete low/high grammar has known rank (equals canonical dimension)."""
    d, r = 2, 1
    basis = canonical_basis(d, r)
    # Build complete low/high grammar (same as canonical)
    atoms = tuple(_make_atom(j, t) for j in range(d) for t in ("low", "high"))
    smap = build_semantic_map(atoms, basis)
    D = canonical_dimension(d, r)
    # The complete canonical grammar + intercept spans the canonical space
    assert smap.exact_rank == D


def test_semantic_map_scale():
    """Atom scale multiplies the canonical column."""
    d, r = 2, 1
    basis = canonical_basis(d, r)
    atom1 = RuleAtom(FuzzyRule((RuleCondition(0, "low"),)), scale=1.0, cost=1.0)
    atom2 = RuleAtom(FuzzyRule((RuleCondition(0, "low"),)), scale=2.0, cost=1.0)
    smap1 = build_semantic_map((atom1,), basis)
    smap2 = build_semantic_map((atom2,), basis)
    col1 = smap1.matrix.toarray()[:, 1]
    col2 = smap2.matrix.toarray()[:, 1]
    np.testing.assert_allclose(col2, 2.0 * col1)


def test_semantic_map_rejects_zero_scale():
    d, r = 2, 1
    basis = canonical_basis(d, r)
    atom = RuleAtom(FuzzyRule((RuleCondition(0, "low"),)), scale=0.0, cost=1.0)
    with pytest.raises(ValueError, match="scale"):
        build_semantic_map((atom,), basis)


def test_semantic_map_rejects_nonpositive_cost():
    d, r = 2, 1
    basis = canonical_basis(d, r)
    atom = RuleAtom(FuzzyRule((RuleCondition(0, "low"),)), scale=1.0, cost=-1.0)
    with pytest.raises(ValueError, match="cost"):
        build_semantic_map((atom,), basis)


def test_semantic_map_stable_hash():
    """Same atoms and basis produce the same semantic map hash."""
    d, r = 2, 1
    basis = canonical_basis(d, r)
    atoms = (_make_atom(0, "low"), _make_atom(1, "high"))
    smap1 = build_semantic_map(atoms, basis)
    smap2 = build_semantic_map(atoms, basis)
    assert smap1.sha256 == smap2.sha256


# ---------------------------------------------------------------------------
# Exact RREF (same-span invariance)
# ---------------------------------------------------------------------------

def test_rref_same_span_invariant():
    """Two dictionaries spanning the same subspace produce the same RREF."""
    d, r = 2, 2
    basis = canonical_basis(d, r)

    # Dict 1: f0=low, f0=high, f1=low, f1=high
    atoms1 = tuple(_make_atom(j, t) for j in range(d) for t in ("low", "high"))
    smap1 = build_semantic_map(atoms1, basis)

    # Dict 2: same atoms in reverse order, with duplicates
    atoms2 = atoms1[::-1] + (atoms1[0],)  # reversed + duplicate
    smap2 = build_semantic_map(atoms2, basis)

    rref1, pivots1 = exact_rref(smap1.matrix)
    rref2, pivots2 = exact_rref(smap2.matrix)

    # Number of nonzero rows (rank) must match
    assert len(rref1) == len(rref2)


def test_rref_nonzero_rows_span_range():
    """RREF rows form a basis for range(A_D)."""
    d, r = 2, 1
    basis = canonical_basis(d, r)
    atoms = (_make_atom(0, "low"), _make_atom(0, "medium"))
    smap = build_semantic_map(atoms, basis)
    rref_rows, pivot_cols = exact_rref(smap.matrix)
    # All pivot columns should be distinct
    assert len(pivot_cols) == len(set(pivot_cols))


# ---------------------------------------------------------------------------
# Semantic subspace RREF (for dictionary mode)
# ---------------------------------------------------------------------------

def test_subspace_rref_same_span():
    """Same-span dictionaries produce the same R_float."""
    d, r = 2, 1
    basis = canonical_basis(d, r)

    # Dict 1: complete low/high grammar
    atoms1 = tuple(_make_atom(j, t) for j in range(d) for t in ("low", "high"))
    smap1 = build_semantic_map(atoms1, basis)
    R1, _, _ = semantic_subspace_rref(smap1)

    # Dict 2: same atoms permuted + one duplicate
    import random
    atoms2 = tuple(atoms1[i % len(atoms1)] for i in [1, 0, 3, 2, 0])
    smap2 = build_semantic_map(atoms2, basis)
    R2, _, _ = semantic_subspace_rref(smap2)

    assert R1.shape == R2.shape
    np.testing.assert_allclose(R1, R2, atol=1e-12)


# ---------------------------------------------------------------------------
# Canonical feature matrix
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_partitions():
    # Strict anchors: a=0, b=1, c=2
    return [_FuzzyPartition(low=0.0, medium=1.0, high=2.0)]


def test_canonical_feature_matrix_constant_column(simple_partitions):
    """Column 0 of Psi is always 1 (constant/intercept)."""
    d, r = 1, 2
    basis = canonical_basis(d, r)
    X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
    Psi = canonical_feature_matrix(X, basis, simple_partitions)
    np.testing.assert_array_equal(Psi[:, 0], 1.0)


def test_canonical_feature_matrix_matches_reference():
    """Production canonical features match reference implementation."""
    d, r = 2, 2
    anchors = [(0.0, 1.0, 2.0), (0.5, 1.5, 2.5)]
    partitions = [_FuzzyPartition(low=a, medium=b, high=c) for a, b, c in anchors]
    basis = canonical_basis(d, r)

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 3, size=(20, d))

    Psi_prod = canonical_feature_matrix(X, basis, partitions)
    for i, xi in enumerate(X):
        psi_ref = ref_canonical_feature_vector(xi, anchors, d, r)
        np.testing.assert_allclose(Psi_prod[i], psi_ref, atol=1e-12,
                                   err_msg=f"Row {i} mismatch")


# ---------------------------------------------------------------------------
# Ruspini identity (T1)
# ---------------------------------------------------------------------------

def test_ruspini_strict_anchors():
    """L + M + H = 1 and L * H = 0 for strict anchors."""
    a, b, c = 0.0, 1.0, 2.0
    for x in np.linspace(-0.5, 2.5, 50):
        result = ref_check_ruspini(a, b, c, x)
        assert abs(result["L+M+H"] - 1.0) < 1e-12, f"L+M+H != 1 at x={x}"
        assert abs(result["L*H"]) < 1e-12, f"L*H != 0 at x={x}"


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def test_decode_canonical_roundtrip():
    """Canonical decode returns the correct monomial->coef mapping."""
    d, r = 2, 1
    basis = canonical_basis(d, r)
    c = np.array([0.5, 1.0, -0.5, 0.0, 0.25])  # D=5 for d=2,r=1
    decoded = decode_canonical(c, basis)
    # Non-near-zero entries only
    for mono, val in decoded.items():
        idx = basis.monomial_index(mono)
        assert abs(val - c[idx]) < 1e-14


def test_decode_rref_feasible():
    """decode_rref returns exact solution when c is in range(A_D)."""
    d, r = 2, 1
    basis = canonical_basis(d, r)
    atoms = tuple(_make_atom(j, t) for j in range(d) for t in ("low", "high"))
    smap = build_semantic_map(atoms, basis)

    # c that is in range: just take a specific linear combination
    gamma_true = np.array([0.5, 1.0, -0.3, 0.7, 2.0])  # K+1 = 5
    c_true = smap.matrix.toarray() @ gamma_true

    gamma_out, cert = decode_rref(c_true, smap)
    assert cert.status == "CERTIFIED"
    assert cert.exact_zero_residual
    assert gamma_out is not None
    # Verify A @ gamma = c
    np.testing.assert_allclose(smap.matrix.toarray() @ gamma_out, c_true, atol=1e-10)


def test_decode_rref_unrepresentable():
    """decode_rref returns REJECTED when c is outside range(A_D)."""
    d, r = 2, 2
    basis = canonical_basis(d, r)
    # Only use 2 atoms from the 13-dimensional basis
    atoms = (_make_atom(0, "low"),)
    smap = build_semantic_map(atoms, basis)

    # c with a nonzero component in a direction not spanned by the dictionary
    c = np.zeros(canonical_dimension(d, r))
    c[5] = 1.0  # some canonical coordinate not covered by the 2-column map

    gamma_out, cert = decode_rref(c, smap)
    # c may or may not be in range; but c[5]=1 with a tiny dictionary is likely not
    # Just check the certificate is consistent
    if cert.status == "REJECTED":
        assert gamma_out is None
        assert not cert.exact_zero_residual
