"""Canonical semantic quotient space for product fuzzy rule classifiers.

Implements the canonical low/high monomial basis, exact rule expansion via the
Ruspini identity M_j = 1 - L_j - H_j, sparse semantic maps, exact rank/RREF
operations backed by SymPy, and decoding from canonical coefficients back to
dictionary representations.

The finite quotient space is the image of the multiaffine product-rule grammar
(at most one term per feature, degree <= r) modulo the ideal

    I = sum_j < L_j + M_j + H_j - 1,  L_j * H_j >.

Every medium term expands exactly as: M_j -> 1 - L_j - H_j.

Exact linear algebra (rank, RREF, kernel, decode) requires the ``csrq``
optional dependency (sympy).  An ImportError with instructions is raised at the
point of first use when sympy is not installed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations, product as iterproduct
from typing import Literal

import numpy as np
import scipy.sparse as sp

from fysvm.rule_svm import FuzzyRule, RuleCondition


# ---------------------------------------------------------------------------
# API data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class CanonicalLiteral:
    """A single low or high condition on one feature."""

    feature: int
    term: Literal["low", "high"]


@dataclass(frozen=True, order=True)
class CanonicalMonomial:
    """A product of canonical literals — one term per feature, only low/high."""

    literals: tuple[CanonicalLiteral, ...]

    @property
    def degree(self) -> int:
        return len(self.literals)

    def __str__(self) -> str:
        if not self.literals:
            return "1"
        return " * ".join(
            f"{lit.term}[{lit.feature}]" for lit in self.literals
        )


@dataclass(frozen=True)
class CanonicalBasis:
    """The graded-lex ordered canonical basis for Q_{d,r}."""

    n_features: int
    max_degree: int
    monomials: tuple[CanonicalMonomial, ...]
    ordering: Literal["graded_lex"] = "graded_lex"
    sha256: str = ""

    @property
    def dimension(self) -> int:
        return len(self.monomials)

    def monomial_index(self, mono: CanonicalMonomial) -> int:
        """Return the index of a monomial in the basis (linear scan)."""
        for i, m in enumerate(self.monomials):
            if m == mono:
                return i
        raise KeyError(f"Monomial {mono} not in basis.")


@dataclass(frozen=True)
class RuleAtom:
    """A fuzzy rule with an optional positive scale and cost."""

    rule: FuzzyRule
    scale: float = 1.0
    cost: float = 1.0


@dataclass
class SemanticMap:
    """Sparse integer matrix mapping rule atoms to canonical coefficients.

    Column k of ``matrix`` is the canonical expansion of atom k (scaled by
    atom.scale), with the intercept atom in column 0.
    """

    basis: CanonicalBasis
    atoms: tuple[RuleAtom, ...]
    matrix: sp.csc_matrix           # shape (D, K+1); column 0 = intercept
    exact_rank: int
    nullity: int
    sha256: str


@dataclass
class SemanticEqualityCertificate:
    """Rational exact-equality certificate for A_D gamma = c.

    When status is 'REJECTED' and c is outside range(A_D), the optional
    ``left_nullspace_witness`` field contains a vector z (as list of Fraction)
    such that z^T A_D = 0 and z^T c != 0, proving c is unrepresentable.
    """

    status: Literal["CERTIFIED", "REJECTED", "UNKNOWN", "INVALID"]
    semantic_contract_sha256: str
    basis_sha256: str
    map_sha256: str
    exact_zero_residual: bool
    residual_nonzero_indices: tuple[int, ...]
    details: dict
    left_nullspace_witness: list[Fraction] | None = None


# ---------------------------------------------------------------------------
# Basis generation
# ---------------------------------------------------------------------------

def _basis_ordering_key(mono: CanonicalMonomial) -> tuple:
    """Graded-lex key: (degree, feature_tuple, term_tuple)."""
    feats = tuple(lit.feature for lit in mono.literals)
    terms = tuple(0 if lit.term == "low" else 1 for lit in mono.literals)
    return (len(mono.literals), feats, terms)


def canonical_basis(n_features: int, max_degree: int) -> CanonicalBasis:
    """Build the graded-lex canonical basis for Q_{d,r}.

    The empty monomial (constant) is always first.  Remaining monomials are
    ordered by increasing degree, then lexicographic feature tuple, then
    lexicographic term string with ``low < high``.

    Canonical dimension:
        D_{d,r} = sum_{l=0}^{r}  C(d, l) * 2^l
    """
    if n_features < 0:
        raise ValueError("n_features must be non-negative.")
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative.")

    monomials: list[CanonicalMonomial] = []

    # Empty monomial (constant / intercept)
    monomials.append(CanonicalMonomial(()))

    for degree in range(1, max_degree + 1):
        for feature_subset in combinations(range(n_features), degree):
            for term_combo in iterproduct(("low", "high"), repeat=degree):
                literals = tuple(
                    CanonicalLiteral(feature=f, term=t)
                    for f, t in zip(feature_subset, term_combo, strict=True)
                )
                monomials.append(CanonicalMonomial(literals))

    mono_tuple = tuple(monomials)
    # Hash of basis is stable across runs
    basis_str = json.dumps(
        {
            "n_features": n_features,
            "max_degree": max_degree,
            "ordering": "graded_lex",
            "monomials": [
                [(lit.feature, lit.term) for lit in m.literals]
                for m in mono_tuple
            ],
        },
        sort_keys=True,
    )
    sha = hashlib.sha256(basis_str.encode()).hexdigest()

    return CanonicalBasis(
        n_features=n_features,
        max_degree=max_degree,
        monomials=mono_tuple,
        ordering="graded_lex",
        sha256=sha,
    )


def canonical_dimension(n_features: int, max_degree: int) -> int:
    """Compute D_{d,r} = sum_{l=0}^{r} C(d,l) * 2^l."""
    from math import comb
    return sum(comb(n_features, l) * (2 ** l) for l in range(max_degree + 1))


def original_grammar_size(n_features: int, max_degree: int) -> int:
    """Compute N_{d,r} = sum_{l=0}^{r} C(d,l) * 3^l (full LMH grammar)."""
    from math import comb
    return sum(comb(n_features, l) * (3 ** l) for l in range(max_degree + 1))


# ---------------------------------------------------------------------------
# Rule expansion: NF(phi_R)
# ---------------------------------------------------------------------------

def _validate_rule_for_expansion(
    rule: FuzzyRule,
    n_features: int,
    max_degree: int,
) -> None:
    """Raise ValueError for invalid rules (proposal §Validation Rules)."""
    if rule.length == 0:
        raise ValueError(
            "Empty rules are reserved for the constant atom; do not supply "
            "them as dictionary rules."
        )
    if rule.length > max_degree:
        raise ValueError(
            f"Rule length {rule.length} exceeds max_degree {max_degree}."
        )
    features_seen: set[int] = set()
    for cond in rule.conditions:
        if cond.feature < 0 or cond.feature >= n_features:
            raise ValueError(
                f"Feature index {cond.feature} out of range [0, {n_features})."
            )
        if cond.feature in features_seen:
            raise ValueError(
                f"Feature {cond.feature} appears more than once in rule."
            )
        features_seen.add(cond.feature)
        if cond.term not in {"low", "medium", "high"}:
            raise ValueError(
                f"Unknown term {cond.term!r}; must be 'low', 'medium', or 'high'."
            )


def expand_rule(
    rule: FuzzyRule,
    basis: CanonicalBasis,
) -> dict[int, int]:
    """Expand a fuzzy rule into canonical basis coefficients (sparse int vector).

    Uses the specialized medium expansion:
        M_j = 1 - L_j - H_j

    Algorithm (proposal §Exact Rule Mapping):
        coordinates = {fixed low/high monomial: 1}
        for each medium condition on feature j:
            replace every (monomial, coeff) with:
                (same monomial,             coeff)
                (monomial + j=low,  -coeff)
                (monomial + j=high, -coeff)
        aggregate identical monomials.

    Returns a dict mapping canonical basis index -> integer coefficient.
    Monomials with coefficient 0 are omitted.
    """
    _validate_rule_for_expansion(rule, basis.n_features, basis.max_degree)

    # Build index for fast monomial lookup
    mono_to_idx: dict[CanonicalMonomial, int] = {
        m: i for i, m in enumerate(basis.monomials)
    }

    # Separate fixed (low/high) and medium conditions
    fixed: list[tuple[int, str]] = []   # (feature, "low" or "high")
    medium_features: list[int] = []

    for cond in rule.conditions:
        if cond.term == "medium":
            medium_features.append(cond.feature)
        else:
            fixed.append((cond.feature, cond.term))

    # Start with the fixed-literal monomial
    fixed_literals = tuple(
        CanonicalLiteral(feature=f, term=t) for f, t in sorted(fixed)
    )
    current: dict[tuple[CanonicalLiteral, ...], int] = {fixed_literals: 1}

    # Expand each medium condition
    for mf in medium_features:
        new: dict[tuple[CanonicalLiteral, ...], int] = {}
        for lits, coeff in current.items():
            # Original term: +coeff
            _add_to(new, lits, coeff)
            # Replace M_j with -L_j: -coeff * L_j
            lits_low = _insert_literal(lits, CanonicalLiteral(feature=mf, term="low"))
            _add_to(new, lits_low, -coeff)
            # Replace M_j with -H_j: -coeff * H_j
            lits_high = _insert_literal(lits, CanonicalLiteral(feature=mf, term="high"))
            _add_to(new, lits_high, -coeff)
        current = new

    # Map to basis indices
    result: dict[int, int] = {}
    for lits, coeff in current.items():
        if coeff == 0:
            continue
        mono = CanonicalMonomial(lits)
        if mono not in mono_to_idx:
            raise ValueError(
                f"Expanded monomial {mono} not in basis (degree cap exceeded?). "
                f"Basis n_features={basis.n_features}, max_degree={basis.max_degree}."
            )
        idx = mono_to_idx[mono]
        result[idx] = result.get(idx, 0) + coeff
    # Remove zeros
    return {k: v for k, v in result.items() if v != 0}


def _add_to(
    d: dict[tuple[CanonicalLiteral, ...], int],
    key: tuple[CanonicalLiteral, ...],
    coeff: int,
) -> None:
    """Add coeff to d[key], leaving out zero entries."""
    d[key] = d.get(key, 0) + coeff


def _insert_literal(
    lits: tuple[CanonicalLiteral, ...],
    new_lit: CanonicalLiteral,
) -> tuple[CanonicalLiteral, ...]:
    """Insert new_lit into lits maintaining sorted order."""
    lst = list(lits) + [new_lit]
    lst.sort()
    return tuple(lst)


# ---------------------------------------------------------------------------
# Semantic map
# ---------------------------------------------------------------------------

def build_semantic_map(
    atoms: tuple[RuleAtom, ...],
    basis: CanonicalBasis,
    *,
    resource_cap: int = 10_000_000,
) -> SemanticMap:
    """Build the semantic map matrix A_D for a dictionary.

    The intercept atom (empty rule, scale=1, cost=1) is always prepended as
    column 0.  The resulting matrix has shape (D, K+1) where K = len(atoms).

    Each column k corresponds to atom k-1 (0-indexed from the supplied list),
    scaled by atom.scale.  Column 0 is the explicit constant atom e_empty.

    ``resource_cap`` limits D * (K+1) to prevent runaway allocation.
    """
    D = basis.dimension
    K = len(atoms)
    if D * (K + 1) > resource_cap:
        raise ValueError(
            f"Semantic map size D={D} * (K+1)={K+1} = {D*(K+1)} exceeds "
            f"resource_cap={resource_cap}. Reduce dictionary size or degree."
        )

    # Validate atoms before building
    for i, atom in enumerate(atoms):
        if not np.isfinite(atom.scale) or atom.scale == 0.0:
            raise ValueError(f"Atom {i}: scale must be finite and nonzero, got {atom.scale}.")
        if not np.isfinite(atom.cost) or atom.cost <= 0.0:
            raise ValueError(f"Atom {i}: cost must be finite and positive, got {atom.cost}.")
        _validate_rule_for_expansion(atom.rule, basis.n_features, basis.max_degree)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    # Column 0: constant atom (e_empty), coefficient 1 at index 0
    rows.append(0)
    cols.append(0)
    vals.append(1.0)

    # Columns 1..K: dictionary atoms
    for col_offset, atom in enumerate(atoms):
        col = col_offset + 1
        expansion = expand_rule(atom.rule, basis)
        for row_idx, int_coeff in expansion.items():
            rows.append(row_idx)
            cols.append(col)
            vals.append(float(int_coeff) * atom.scale)

    matrix = sp.csc_matrix(
        (vals, (rows, cols)),
        shape=(D, K + 1),
        dtype=np.float64,
    )

    # Exact rank and nullity via SymPy
    exact_rank, nullity = _exact_rank_nullity(matrix)

    # Hash of the matrix for artifact integrity
    mat_dense = matrix.toarray()
    sha = hashlib.sha256(mat_dense.tobytes()).hexdigest()

    return SemanticMap(
        basis=basis,
        atoms=atoms,
        matrix=matrix,
        exact_rank=exact_rank,
        nullity=nullity,
        sha256=sha,
    )


# ---------------------------------------------------------------------------
# Exact linear algebra (requires sympy)
# ---------------------------------------------------------------------------

def _require_sympy() -> None:
    try:
        import sympy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Exact linear algebra (rank, RREF, decode, certificates) requires "
            "sympy. Install it with: pip install fysvm[csrq]"
        ) from exc


def _exact_rank_nullity(matrix: sp.spmatrix) -> tuple[int, int]:
    """Compute exact rank and nullity using SymPy rational arithmetic."""
    _require_sympy()
    import sympy

    arr = matrix.toarray()
    sym_matrix = sympy.Matrix(arr.tolist()).applyfunc(sympy.Rational)
    rank = sym_matrix.rank()
    _, ncols = arr.shape
    nullity = ncols - rank
    return int(rank), int(nullity)


def exact_rref(
    matrix: sp.spmatrix,
    *,
    tolerance: float = 0.0,
) -> "tuple[list[list[Fraction]], list[int]]":
    """Compute exact RREF of matrix using SymPy rational arithmetic.

    Returns (rref_rows, pivot_cols) where rref_rows is a list of nonzero rows
    as lists of Fraction, and pivot_cols is the list of pivot column indices.

    Same-span dictionaries produce the same RREF (identical pivot structure and
    values), making this the canonical subspace representation used by
    Variant A dictionary mode.
    """
    _require_sympy()
    import sympy

    arr = matrix.toarray()
    sym_matrix = sympy.Matrix(arr.tolist()).applyfunc(sympy.Rational)
    rref_mat, pivot_cols = sym_matrix.rref()

    # Extract nonzero rows
    nrows, _ = rref_mat.shape
    nonzero_rows: list[list[Fraction]] = []
    for r in range(nrows):
        row = [Fraction(int(rref_mat[r, c].p), int(rref_mat[r, c].q)) for c in range(rref_mat.shape[1])]
        if any(v != 0 for v in row):
            nonzero_rows.append(row)

    return nonzero_rows, list(pivot_cols)


def exact_kernel(matrix: sp.spmatrix) -> "list[list[Fraction]]":
    """Compute exact rational null space of matrix.

    Returns a list of vectors (as list[Fraction]) forming a basis of ker(matrix).
    """
    _require_sympy()
    import sympy

    arr = matrix.toarray()
    sym_matrix = sympy.Matrix(arr.tolist()).applyfunc(sympy.Rational)
    null_vecs = sym_matrix.nullspace()
    result: list[list[Fraction]] = []
    for v in null_vecs:
        # Normalize so entries are Fractions
        row = [Fraction(int(v[i].p), int(v[i].q)) for i in range(v.shape[0])]
        result.append(row)
    return result


def semantic_subspace_rref(
    semantic_map: SemanticMap,
) -> "tuple[np.ndarray, list[list[Fraction]], list[int]]":
    """Compute R such that range(R) = range(A_D) using exact RREF.

    Following the proposal: R^T = nonzero_rows(rref(A_D^T)).
    So R is (D x rank) where each column is a canonical basis vector spanning
    the dictionary semantic subspace.

    Returns:
        R_float: (D, rank) float64 numpy array for numerical computation
        R_exact: list of rank D-vectors as Fraction (rows of R^T)
        pivot_cols: pivot column indices from the RREF
    """
    _require_sympy()

    A = semantic_map.matrix  # (D, K+1)
    # Compute RREF of A^T  (K+1, D)
    AT = A.T
    rref_rows, pivot_cols = exact_rref(AT)
    # rref_rows are the nonzero rows of rref(A^T), each has length D
    # R^T has these as rows, so R has them as columns

    rank = len(rref_rows)
    D = A.shape[0]

    R_float = np.zeros((D, rank), dtype=np.float64)
    for col_idx, row in enumerate(rref_rows):
        for row_idx, val in enumerate(row):
            R_float[row_idx, col_idx] = float(val)

    return R_float, rref_rows, pivot_cols


# ---------------------------------------------------------------------------
# Evaluation: canonical feature matrix
# ---------------------------------------------------------------------------

def canonical_feature_matrix(
    X: np.ndarray,
    basis: CanonicalBasis,
    partitions: "list",
    selected_feature_indices: "list[int] | None" = None,
) -> np.ndarray:
    """Compute the canonical feature matrix Psi_bar(X) of shape (n, D).

    partitions: list of partition objects with .low, .medium (used as b_j), .high
    Each partition provides the anchors (a_j, b_j, c_j):
        - a_j = partition.low
        - b_j = partition.medium
        - c_j = partition.high

    L_j(x) = linear_down(x_j, a_j, b_j)
    H_j(x) = linear_up(x_j, b_j, c_j)

    selected_feature_indices: if provided, X columns are already screened;
    the feature index in CanonicalLiteral refers to position in partitions.
    """
    from fysvm.rule_svm import _linear_down, _linear_up

    n_samples = X.shape[0]
    D = basis.dimension
    n_sel = len(partitions)

    # Precompute low and high memberships for all selected features
    low_mem = np.empty((n_samples, n_sel), dtype=np.float64)
    high_mem = np.empty((n_samples, n_sel), dtype=np.float64)

    for j, part in enumerate(partitions):
        col = X[:, j]
        a_j, b_j, c_j = float(part.low), float(part.medium), float(part.high)
        low_mem[:, j] = _linear_down(col, a_j, b_j)
        high_mem[:, j] = _linear_up(col, b_j, c_j)

    # Build canonical feature matrix
    Psi = np.empty((n_samples, D), dtype=np.float64)
    for i, mono in enumerate(basis.monomials):
        if mono.degree == 0:
            Psi[:, i] = 1.0
        else:
            val = np.ones(n_samples, dtype=np.float64)
            for lit in mono.literals:
                if lit.term == "low":
                    val = val * low_mem[:, lit.feature]
                else:
                    val = val * high_mem[:, lit.feature]
            Psi[:, i] = val

    return Psi


# ---------------------------------------------------------------------------
# Decoding: canonical c -> dictionary coefficients
# ---------------------------------------------------------------------------

def decode_canonical(
    c: np.ndarray,
    basis: CanonicalBasis,
) -> dict[CanonicalMonomial, float]:
    """Return canonical low/high monomials with their coefficients.

    The intercept is stored as the empty monomial coefficient.
    Monomials with near-zero coefficients (abs < 1e-15) are omitted.
    """
    result: dict[CanonicalMonomial, float] = {}
    for i, mono in enumerate(basis.monomials):
        if abs(c[i]) > 1e-15:
            result[mono] = float(c[i])
    return result


def _find_left_nullspace_witness(
    A: np.ndarray,
    c: np.ndarray,
    c_exact: np.ndarray,
) -> "list[Fraction] | None":
    """Find a left-nullspace vector z of A with z^T c != 0.

    A vector z is a left-nullspace vector of A iff z^T A = 0, i.e.
    z is in the nullspace of A^T.  If z^T c != 0, then c is outside range(A).

    Returns z as list[Fraction] if found, else None.
    """
    _require_sympy()
    import sympy

    D, K = A.shape
    AT_sym = sympy.Matrix(A.tolist()).T.applyfunc(sympy.Rational)
    null_vecs = AT_sym.nullspace()

    c_frac = [Fraction(*float(ci).as_integer_ratio()) for ci in c]

    for nv in null_vecs:
        z = [Fraction(int(nv[i].p), int(nv[i].q)) for i in range(D)]
        dot = sum(z[i] * c_frac[i] for i in range(D))
        if dot != Fraction(0):
            return z

    return None


def decode_rref(
    c: np.ndarray,
    semantic_map: SemanticMap,
) -> "tuple[np.ndarray | None, SemanticEqualityCertificate]":
    """Decode c into dictionary atom coefficients via exact RREF.

    Solves A_D gamma = c exactly in rational arithmetic, with free variables
    set to zero.

    Returns:
        gamma: float64 array of shape (K+1,) if representable, else None
        certificate: SemanticEqualityCertificate with CERTIFIED / UNREPRESENTABLE
    """
    _require_sympy()
    import sympy

    A = semantic_map.matrix
    D, Kp1 = A.shape

    # Build augmented system [A_D | c] and solve via RREF
    A_dense = A.toarray()
    c_col = c.reshape(-1, 1)
    augmented = np.hstack([A_dense, c_col])

    sym_aug = sympy.Matrix(augmented.tolist()).applyfunc(sympy.Rational)
    rref_aug, pivots = sym_aug.rref()

    # Check consistency: if any row has all-zero A portion but nonzero c entry
    # => c is outside range(A_D) (unrepresentable)
    nrows = rref_aug.shape[0]
    for r in range(nrows):
        row_A = [rref_aug[r, k] for k in range(Kp1)]
        row_c = rref_aug[r, Kp1]
        if all(v == 0 for v in row_A) and row_c != 0:
            # c is outside range(A_D); compute left-nullspace witness
            # z^T A_D = 0 and z^T c != 0
            witness = _find_left_nullspace_witness(A_dense, c, c_col.reshape(-1))
            return None, SemanticEqualityCertificate(
                status="REJECTED",
                semantic_contract_sha256="",
                basis_sha256=semantic_map.basis.sha256,
                map_sha256=semantic_map.sha256,
                exact_zero_residual=False,
                residual_nonzero_indices=tuple(
                    i for i in range(D) if abs(c[i]) > 0
                ),
                details={
                    "reason": "c is outside range(A_D)",
                    "inconsistency_row": r,
                    "rref_c_value": str(row_c),
                },
                left_nullspace_witness=witness,
            )

    # Extract solution: free variables = 0, pivot variables solved
    gamma_sym = [sympy.Rational(0)] * Kp1
    for r, piv in enumerate(pivots):
        if piv < Kp1:
            gamma_sym[piv] = rref_aug[r, Kp1]
        # Skip pivot in augmented (c) column — means system is determined

    gamma = np.array([float(g) for g in gamma_sym], dtype=np.float64)

    # Verify residual exactly
    gamma_frac = [Fraction(int(g.p), int(g.q)) for g in gamma_sym]
    A_frac = [
        [Fraction(int(sympy.Rational(A_dense[r, k]).p), int(sympy.Rational(A_dense[r, k]).q))
         for k in range(Kp1)]
        for r in range(D)
    ]
    c_frac = [Fraction(*float(cv).as_integer_ratio()) for cv in c]

    residual = []
    nonzero_indices = []
    for r in range(D):
        val = sum(A_frac[r][k] * gamma_frac[k] for k in range(Kp1)) - c_frac[r]
        residual.append(val)
        if val != 0:
            nonzero_indices.append(r)

    exact_zero = len(nonzero_indices) == 0

    cert = SemanticEqualityCertificate(
        status="CERTIFIED" if exact_zero else "REJECTED",
        semantic_contract_sha256="",
        basis_sha256=semantic_map.basis.sha256,
        map_sha256=semantic_map.sha256,
        exact_zero_residual=exact_zero,
        residual_nonzero_indices=tuple(nonzero_indices),
        details={
            "gamma_exact": [str(g) for g in gamma_frac],
            "n_free_variables": Kp1 - len(pivots),
            "pivots": list(pivots),
        },
    )
    return gamma if exact_zero else None, cert


def decode_minimum_l2(
    c: np.ndarray,
    semantic_map: SemanticMap,
) -> "tuple[np.ndarray | None, str]":
    """Decode c into minimum-L2-norm atom coefficients.

    Uses scipy least-squares. Returns (gamma, status) where status is
    'OK', 'UNREPRESENTABLE', or 'NUMERICAL_ERROR'.
    """
    A = semantic_map.matrix.toarray()
    try:
        gamma, residuals, rank, sv = np.linalg.lstsq(A, c, rcond=None)
    except np.linalg.LinAlgError:
        return None, "NUMERICAL_ERROR"

    residual_norm = float(np.linalg.norm(A @ gamma - c))
    if residual_norm > 1e-8 * (1 + np.linalg.norm(c)):
        return None, "UNREPRESENTABLE"
    return gamma, "OK"


# ---------------------------------------------------------------------------
# Utility: contract hash
# ---------------------------------------------------------------------------

def semantic_contract_hash(
    n_features: int,
    selected_feature_indices: list[int],
    anchors: list[tuple[float, float, float]],
    max_degree: int,
    intercept_penalty: float,
    degree_penalty: float,
) -> str:
    """Stable SHA-256 hash of the semantic contract for artifact linking."""
    contract = {
        "n_features": n_features,
        "selected_feature_indices": list(selected_feature_indices),
        "anchors": [[float(a), float(b), float(c_)] for a, b, c_ in anchors],
        "max_degree": max_degree,
        "and_operator": "product",
        "intercept_penalty": float(intercept_penalty),
        "degree_penalty": float(degree_penalty),
    }
    s = json.dumps(contract, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()
