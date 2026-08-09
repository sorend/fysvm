"""Quotient Atomic Fuzzy SVM (Variant B) — optional experimental estimator.

Trains with a quotient atomic norm penalizing the minimum weighted cost
representation of the canonical coefficient vector c in the dictionary atom
basis.  A positive Tikhonov tie-break epsilon ensures uniqueness of the
semantic output.

Requires the ``csrq-atomic`` optional dependency group (sympy + osqp).
Import errors are raised lazily at construction time.

Hull certification uses exact rational arithmetic and provides one of:
    CERTIFIED_EQUAL     — exact rational inclusion witnesses in both directions
    CERTIFIED_DIFFERENT — an exact separating or minimum-gauge witness
    UNKNOWN             — numerical result without exact witnesses
    INVALID             — bad inputs
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y

from fysvm.quotient import (
    CanonicalBasis,
    RuleAtom,
    SemanticMap,
    build_semantic_map,
    canonical_basis,
    canonical_dimension,
    canonical_feature_matrix,
)
from fysvm.rule_svm import (
    FuzzyRule,
    RuleCondition,
    _FuzzyPartition,
    _linear_down,
    _linear_up,
)


def _require_osqp() -> None:
    try:
        import osqp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "QuotientAtomicFuzzySVM requires osqp. "
            "Install it with: pip install fysvm[csrq-atomic]"
        ) from exc


def _require_sympy() -> None:
    try:
        import sympy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Exact hull certification requires sympy. "
            "Install it with: pip install fysvm[csrq-atomic]"
        ) from exc


# ---------------------------------------------------------------------------
# Hull certification result
# ---------------------------------------------------------------------------

@dataclass
class HullCertificationResult:
    """Result of exact hull equality certification between two dictionaries."""

    status: Literal["CERTIFIED_EQUAL", "CERTIFIED_DIFFERENT", "UNKNOWN", "INVALID"]
    details: dict[str, Any]


def certify_hull_equality(
    dict1: tuple[RuleAtom, ...],
    dict2: tuple[RuleAtom, ...],
    basis: CanonicalBasis,
    *,
    costs1: list[float] | None = None,
    costs2: list[float] | None = None,
    tol: float = 1e-8,
) -> HullCertificationResult:
    """Certify whether two atom dictionaries have equal weighted signed atom hulls.

    The weighted signed atom hull for dictionary D with costs w is:
        H_{D,w} = conv{ ±b_k / w_k : k = 0, ..., K }

    where b_k is the canonical expansion column of atom k.

    Two dictionaries have the same atomic norm iff their weighted signed hulls
    are equal.  Hull equality is checked by verifying generator inclusion in
    both directions via LP feasibility.

    Exact rational feasibility witnesses promote UNKNOWN -> CERTIFIED_EQUAL.
    A left-nullspace range-exclusion certificate promotes UNKNOWN -> CERTIFIED_DIFFERENT.

    Parameters
    ----------
    dict1, dict2 : tuple[RuleAtom, ...]
        The two atom dictionaries to compare (costs taken from RuleAtom.cost).
    basis : CanonicalBasis
        Shared canonical basis.
    costs1, costs2 : list[float] or None
        Override atom costs (uses RuleAtom.cost by default).
    tol : float
        Numerical feasibility tolerance for LP.
    """
    try:
        smap1 = build_semantic_map(dict1, basis)
        smap2 = build_semantic_map(dict2, basis)
    except (ValueError, ImportError) as exc:
        return HullCertificationResult(
            status="INVALID",
            details={"reason": str(exc)},
        )

    w1 = np.array(
        [1.0] + [a.cost if costs1 is None else costs1[i] for i, a in enumerate(dict1)],
        dtype=np.float64,
    )
    w2 = np.array(
        [1.0] + [a.cost if costs2 is None else costs2[i] for i, a in enumerate(dict2)],
        dtype=np.float64,
    )

    # Validate costs
    if np.any(w1 <= 0) or np.any(w2 <= 0):
        return HullCertificationResult(
            status="INVALID",
            details={"reason": "All atom costs (including intercept) must be positive."},
        )

    A1 = smap1.matrix.toarray()  # (D, K1+1)
    A2 = smap2.matrix.toarray()  # (D, K2+1)

    # Normalized generator atoms: v_k = b_k / w_k, both signs
    gens1 = [(A1[:, k] / w1[k], k, 1) for k in range(A1.shape[1])]
    gens1 += [(- A1[:, k] / w1[k], k, -1) for k in range(A1.shape[1])]

    gens2 = [(A2[:, k] / w2[k], k, 1) for k in range(A2.shape[1])]
    gens2 += [(- A2[:, k] / w2[k], k, -1) for k in range(A2.shape[1])]

    try:
        from scipy.optimize import linprog

        # Check every generator of dict1 lies in the hull of dict2
        for v, k, sign in gens1:
            lp = _atom_in_hull_lp_with_sol(v, A2, w2, linprog, tol=tol)
            if not lp.in_hull:
                if not lp.feasible:
                    # v is not in range(B); try left-nullspace certificate
                    cert = _try_range_exclusion_witness(v, A2)
                    if cert is not None:
                        return HullCertificationResult(
                            status="CERTIFIED_DIFFERENT",
                            details={
                                "reason": "dict1 generator not in range of dict2",
                                "generator_index": k,
                                "generator_sign": sign,
                                "range_exclusion_witness": [str(z) for z in cert],
                            },
                        )
                    return HullCertificationResult(
                        status="UNKNOWN",
                        details={
                            "reason": "LP suggests dict1 generator outside range of dict2, "
                                      "but no exact rational witness computed",
                            "generator_index": k,
                            "generator_sign": sign,
                        },
                    )
                else:
                    # v is in range but cost > 1; try dual certificate
                    cert = _try_dual_certificate(v, A2, w2, lp.dual_lambda, tol=tol)
                    if cert is not None:
                        return HullCertificationResult(
                            status="CERTIFIED_DIFFERENT",
                            details={
                                "reason": "dict1 generator in range of dict2 but cost > 1",
                                "generator_index": k,
                                "generator_sign": sign,
                                "min_cost": lp.min_cost,
                                "dual_witness": [str(z) for z in cert],
                            },
                        )
                    return HullCertificationResult(
                        status="UNKNOWN",
                        details={
                            "reason": "LP suggests dict1 generator has cost > 1 in hull of dict2, "
                                      "but no exact dual witness computed",
                            "generator_index": k,
                            "generator_sign": sign,
                            "min_cost": lp.min_cost,
                        },
                    )

            # Verify exact rational witness for inclusion
            cert_ok = _verify_rational_inclusion(v, A2, w2, lp.delta, tol=tol)
            if not cert_ok:
                # LP says in hull but can't rationalize exactly — remain at UNKNOWN
                return HullCertificationResult(
                    status="UNKNOWN",
                    details={
                        "reason": "LP suggests dict1 generator in hull of dict2, "
                                  "but exact rational verification failed",
                        "generator_index": k,
                        "generator_sign": sign,
                    },
                )

        # Check every generator of dict2 lies in the hull of dict1
        for v, k, sign in gens2:
            lp = _atom_in_hull_lp_with_sol(v, A1, w1, linprog, tol=tol)
            if not lp.in_hull:
                if not lp.feasible:
                    cert = _try_range_exclusion_witness(v, A1)
                    if cert is not None:
                        return HullCertificationResult(
                            status="CERTIFIED_DIFFERENT",
                            details={
                                "reason": "dict2 generator not in range of dict1",
                                "generator_index": k,
                                "generator_sign": sign,
                                "range_exclusion_witness": [str(z) for z in cert],
                            },
                        )
                    return HullCertificationResult(
                        status="UNKNOWN",
                        details={
                            "reason": "LP suggests dict2 generator outside range of dict1, "
                                      "but no exact rational witness computed",
                            "generator_index": k,
                            "generator_sign": sign,
                        },
                    )
                else:
                    cert = _try_dual_certificate(v, A1, w1, lp.dual_lambda, tol=tol)
                    if cert is not None:
                        return HullCertificationResult(
                            status="CERTIFIED_DIFFERENT",
                            details={
                                "reason": "dict2 generator in range of dict1 but cost > 1",
                                "generator_index": k,
                                "generator_sign": sign,
                                "min_cost": lp.min_cost,
                                "dual_witness": [str(z) for z in cert],
                            },
                        )
                    return HullCertificationResult(
                        status="UNKNOWN",
                        details={
                            "reason": "LP suggests dict2 generator has cost > 1 in hull of dict1, "
                                      "but no exact dual witness computed",
                            "generator_index": k,
                            "generator_sign": sign,
                            "min_cost": lp.min_cost,
                        },
                    )

            cert_ok = _verify_rational_inclusion(v, A1, w1, lp.delta, tol=tol)
            if not cert_ok:
                return HullCertificationResult(
                    status="UNKNOWN",
                    details={
                        "reason": "LP suggests dict2 generator in hull of dict1, "
                                  "but exact rational verification failed",
                        "generator_index": k,
                        "generator_sign": sign,
                    },
                )

        return HullCertificationResult(
            status="CERTIFIED_EQUAL",
            details={
                "reason": "All generators verified by exact rational inclusion witnesses",
                "n_generators_dict1": len(gens1),
                "n_generators_dict2": len(gens2),
            },
        )

    except Exception as exc:
        return HullCertificationResult(
            status="UNKNOWN",
            details={"reason": f"LP solver error: {exc}"},
        )


@dataclass
class _LPResult:
    """Internal LP result for hull inclusion check."""

    in_hull: bool
    feasible: bool
    min_cost: float
    delta: np.ndarray | None   # [delta_plus (K), delta_minus (K)]
    dual_lambda: np.ndarray | None  # dual for equality constraints (D,)


def _atom_in_hull_lp_with_sol(
    v: np.ndarray,
    B: np.ndarray,
    w: np.ndarray,
    linprog: Any,
    tol: float = 1e-8,
) -> _LPResult:
    """Check whether v is in the weighted signed atom hull of B with weights w.

    Solves: min ||delta||_{1,w}  s.t. B (delta_plus - delta_minus) = v,
                                       delta_plus, delta_minus >= 0.

    Returns an _LPResult with:
      - in_hull: True if the min weighted L1 norm is <= 1
      - feasible: True if the LP is feasible (v is in range(B))
      - min_cost: the LP optimal value (inf if infeasible)
      - delta: [delta_plus; delta_minus] at optimum (None if infeasible)
      - dual_lambda: dual variables for equality constraints (None if infeasible)
    """
    D, K = B.shape
    n_vars = 2 * K
    c_obj = np.concatenate([w, w])
    A_eq = np.hstack([B, -B])
    b_eq = v
    bounds = [(0.0, None)] * n_vars

    result = linprog(
        c_obj,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if result.status != 0:
        return _LPResult(in_hull=False, feasible=False, min_cost=float("inf"),
                         delta=None, dual_lambda=None)

    min_cost = float(result.fun)
    dual = None
    if hasattr(result, "eqlin") and result.eqlin is not None:
        dual = np.asarray(result.eqlin.marginals, dtype=np.float64)

    if min_cost > 1.0 + tol:
        return _LPResult(in_hull=False, feasible=True, min_cost=min_cost,
                         delta=result.x, dual_lambda=dual)

    return _LPResult(in_hull=True, feasible=True, min_cost=min_cost,
                     delta=result.x, dual_lambda=dual)


# Keep original function for backward compatibility
def _atom_in_hull_lp(
    v: np.ndarray,
    B: np.ndarray,
    w: np.ndarray,
    linprog: Any,
    tol: float = 1e-8,
) -> bool:
    """Check whether v is in the weighted signed atom hull of B with weights w."""
    lp = _atom_in_hull_lp_with_sol(v, B, w, linprog, tol=tol)
    return lp.in_hull


def _verify_rational_inclusion(
    v: np.ndarray,
    B: np.ndarray,
    w: np.ndarray,
    delta_float: np.ndarray | None,
    *,
    tol: float = 1e-8,
) -> bool:
    """Verify inclusion using exact rational arithmetic.

    Rationalizes delta_float, checks B*(d_p - d_m) == v and ||delta||_{1,w} <= 1.
    Returns True if exact rational verification passes.
    """
    if delta_float is None:
        return False

    K = B.shape[1]
    # delta_float = [delta_plus (K), delta_minus (K)]
    dp = delta_float[:K]
    dm = delta_float[K:]

    # Rationalize each entry
    dp_frac = [Fraction(*float(x).as_integer_ratio()).limit_denominator(10 ** 9) for x in dp]
    dm_frac = [Fraction(*float(x).as_integer_ratio()).limit_denominator(10 ** 9) for x in dm]

    # Build B as Fraction matrix
    D = B.shape[0]
    B_frac = [
        [Fraction(*float(B[r, c]).as_integer_ratio()) for c in range(K)]
        for r in range(D)
    ]
    v_frac = [Fraction(*float(vi).as_integer_ratio()) for vi in v]
    w_frac = [Fraction(*float(wi).as_integer_ratio()) for wi in w]

    # Check residual: B*(dp - dm) - v == 0
    for r in range(D):
        val = sum(B_frac[r][c] * (dp_frac[c] - dm_frac[c]) for c in range(K)) - v_frac[r]
        if abs(val) > Fraction(0):
            return False

    # Check cost: sum_k w_k * (dp_k + dm_k) <= 1
    cost = sum(w_frac[c] * (dp_frac[c] + dm_frac[c]) for c in range(K))
    if cost > Fraction(1):
        return False

    return True


def _try_dual_certificate(
    v: np.ndarray,
    B: np.ndarray,
    w: np.ndarray,
    lambda_float: np.ndarray | None,
    *,
    tol: float = 1e-8,
) -> list[Fraction] | None:
    """Try to construct an exact rational dual certificate proving cost > 1.

    A dual certificate lambda satisfies:
        |B^T lambda|_k <= w_k  for all k  (dual feasibility)
        v^T lambda > 1                     (dual objective proves cost > 1)

    Returns the rationalized lambda as list[Fraction] if verified, else None.
    """
    if lambda_float is None:
        return None

    D, K = B.shape
    lambda_frac = [
        Fraction(*float(x).as_integer_ratio()).limit_denominator(10 ** 9)
        for x in lambda_float
    ]
    v_frac = [Fraction(*float(vi).as_integer_ratio()) for vi in v]
    B_frac = [
        [Fraction(*float(B[r, c]).as_integer_ratio()) for c in range(K)]
        for r in range(D)
    ]
    w_frac = [Fraction(*float(wi).as_integer_ratio()) for wi in w]

    # Check dual feasibility: |B^T lambda|_k <= w_k
    for c in range(K):
        bt_lambda = sum(B_frac[r][c] * lambda_frac[r] for r in range(D))
        if abs(bt_lambda) > w_frac[c]:
            return None

    # Check v^T lambda > 1
    obj = sum(v_frac[r] * lambda_frac[r] for r in range(D))
    if obj <= Fraction(1):
        return None

    return lambda_frac


def _try_range_exclusion_witness(
    v: np.ndarray,
    B: np.ndarray,
) -> list[Fraction] | None:
    """Try to find a left-nullspace vector z of B with z^T v != 0.

    If successful, this is an exact certificate that v is not in range(B).
    Returns the witness as a list of Fractions, or None if not found.
    """
    try:
        _require_sympy()
    except ImportError:
        return None

    import sympy

    D, K = B.shape
    B_frac = [[Fraction(*float(B[r, c]).as_integer_ratio()) for c in range(K)] for r in range(D)]
    v_frac = [Fraction(*float(vi).as_integer_ratio()) for vi in v]

    # Left nullspace of B = nullspace of B^T
    BT_sym = sympy.Matrix([[int(B_frac[r][c].numerator) * int(B_frac[r][c].denominator)
                            if False else
                            sympy.Rational(B_frac[r][c].numerator, B_frac[r][c].denominator)
                            for r in range(D)]
                           for c in range(K)])
    null_vecs = BT_sym.nullspace()

    for nv in null_vecs:
        # nv is a D-vector (left null of B)
        z_frac = [Fraction(int(nv[i].p), int(nv[i].q)) for i in range(D)]
        dot = sum(z_frac[r] * v_frac[r] for r in range(D))
        if dot != Fraction(0):
            return z_frac

    return None


# ---------------------------------------------------------------------------
# QuotientAtomicFuzzySVM
# ---------------------------------------------------------------------------

class QuotientAtomicFuzzySVM:
    """Variant B: Quotient atomic norm fuzzy SVM (experimental).

    The quotient atomic norm Omega_{D,w}(c) is the infimum over atom
    coefficients gamma with B_D gamma = c of the weighted L1 norm:
        sum_k w_k |gamma_k|.

    The optimization objective is:
        min_{c in S_D} [Omega_{D,w}(c) + (eps/2) c^T G c
                        + C sum_i s_i [1 - y_i psi_bar(x_i)^T c]_+^2]

    with eps > 0 for uniqueness.

    This is a convex QP after splitting gamma = gamma_plus - gamma_minus:
        min (eps/2) c^T G c + w^T(gamma_plus + gamma_minus) + C sum_i s_i xi_i^2
        s.t.  B_D (gamma_plus - gamma_minus) = c
              y_i Psi[i,:] c + xi_i >= 1
              gamma_plus, gamma_minus, xi >= 0

    OSQP is the recommended backend (optional dependency).

    The fitted object exposes the unique semantic c_, not the nonunique gamma.

    Parameters
    ----------
    atom_dictionary : tuple[RuleAtom, ...]
        Dictionary of atoms with positive costs.
    C : float
        Hinge loss regularization strength.
    max_rule_length : int
        Maximum canonical monomial degree.
    partition_quantiles : (float, float, float)
        Quantiles for low/medium/high anchors.
    degree_penalty : float
        eta >= 0; weight increment per extra degree above 1.
    intercept_cost : float
        Positive cost w_0 for the explicit intercept atom.
    intercept_penalty : float
        p_0 > 0; degree weight of the empty monomial in G.
    atomic_tie_break : float
        Tikhonov epsilon > 0 ensuring unique semantic minimiser.
    class_weight : dict or "balanced" or None
        Class weights folded into sample weights.
    max_semantic_terms : int
        Hard cap on canonical dimension D.
    max_iter : int
        Maximum OSQP iterations.
    tol : float
        OSQP absolute and relative tolerance.
    """

    def __init__(
        self,
        *,
        atom_dictionary: tuple[RuleAtom, ...],
        C: float = 1.0,
        max_rule_length: int = 2,
        partition_quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95),
        degree_penalty: float = 0.35,
        intercept_cost: float = 1.0,
        intercept_penalty: float = 1.0,
        atomic_tie_break: float = 1e-6,
        feature_names: tuple[str, ...] | None = None,
        class_weight: dict[object, float] | Literal["balanced"] | None = None,
        max_semantic_terms: int = 1024,
        max_iter: int = 100000,
        tol: float = 1e-6,
    ) -> None:
        _require_osqp()
        if atomic_tie_break <= 0:
            raise ValueError("atomic_tie_break must be strictly positive.")
        if intercept_cost <= 0:
            raise ValueError("intercept_cost must be strictly positive.")
        if intercept_penalty <= 0:
            raise ValueError("intercept_penalty must be strictly positive.")
        if C <= 0:
            raise ValueError("C must be strictly positive.")

        self.atom_dictionary = atom_dictionary
        self.C = C
        self.max_rule_length = max_rule_length
        self.partition_quantiles = partition_quantiles
        self.degree_penalty = degree_penalty
        self.intercept_cost = intercept_cost
        self.intercept_penalty = intercept_penalty
        self.atomic_tie_break = atomic_tie_break
        self.feature_names = feature_names
        self.class_weight = class_weight
        self.max_semantic_terms = max_semantic_terms
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "QuotientAtomicFuzzySVM":
        """Fit the quotient atomic fuzzy SVM via OSQP convex QP.

        Parameters
        ----------
        X : (n_samples, n_features) array
        y : (n_samples,) binary labels
        sample_weight : (n_samples,) non-negative weights or None
        """
        import osqp

        X_check, y_check = check_X_y(X, y, dtype=np.float64)
        n_samples, n_features = X_check.shape

        classes = unique_labels(y_check)
        if len(classes) != 2:
            raise ValueError(
                f"QuotientAtomicFuzzySVM requires binary targets; "
                f"got {len(classes)} classes."
            )
        self.classes_ = classes
        self.n_features_in_ = n_features

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64)
            if sw.shape != (n_samples,):
                raise ValueError("sample_weight must have shape (n_samples,).")
            if np.any(sw < 0):
                raise ValueError("sample_weight must be non-negative.")
        else:
            sw = np.ones(n_samples, dtype=np.float64)

        # Fit partitions (strict anchors required — no 'drop' policy)
        partitions = self._fit_partitions(X_check)
        for j, p in enumerate(partitions):
            if not (p.low < p.medium < p.high):
                raise ValueError(
                    f"Tied anchors on feature {j} (a={p.low}, b={p.medium}, "
                    f"c={p.high}). QuotientAtomicFuzzySVM always rejects tied "
                    "anchors; use CSRQClassifier with strict_anchor_policy='drop' "
                    "for automatic dropping."
                )
        self.partitions_ = partitions

        # Build canonical basis
        d = n_features
        r = self.max_rule_length
        D = canonical_dimension(d, r)
        if D > self.max_semantic_terms:
            raise ValueError(
                f"Canonical dimension D={D} exceeds max_semantic_terms="
                f"{self.max_semantic_terms}. "
                "Reduce max_rule_length or set a higher max_semantic_terms."
            )
        self.basis_ = canonical_basis(d, r)

        # Build semantic map from atom dictionary
        smap = build_semantic_map(self.atom_dictionary, self.basis_)
        self.semantic_map_ = smap

        B = smap.matrix.toarray()  # (D, K+1)
        K1 = B.shape[1]  # K+1 atoms including intercept

        # Atom costs: w_0 = intercept_cost, w_k = atom.cost
        w_atoms = np.array(
            [self.intercept_cost] + [a.cost for a in self.atom_dictionary],
            dtype=np.float64,
        )
        if np.any(w_atoms <= 0):
            raise ValueError("All atom costs (including intercept_cost) must be positive.")

        # Degree weights p_q
        p_vec = self._degree_weights(self.basis_)  # shape (D,)
        G_diag = p_vec ** 2  # G = diag(p_q^2)

        # Canonical feature matrix Psi, shape (n, D)
        Psi = canonical_feature_matrix(X_check, self.basis_, self.partitions_)

        # Signed labels and effective sample weights
        y_signed = self._signed_labels(y_check)
        sw_eff = self._apply_class_weights(y_check, sw)

        # ------------------------------------------------------------------
        # Build OSQP problem
        # Variables z = [c (D), gamma_plus (K+1), gamma_minus (K+1), xi (n)]
        # ------------------------------------------------------------------
        n_c = D
        n_k = K1
        n_n = n_samples
        n_total = n_c + 2 * n_k + n_n

        # Variable index offsets
        i_c = 0
        i_gp = n_c
        i_gm = n_c + n_k
        i_xi = n_c + 2 * n_k

        # P matrix (diagonal): (eps/2)*G for c, C*sw for xi (factor 2 from
        # 0.5*x^T P x => P entry = 2*coeff_on_x^2)
        eps = self.atomic_tie_break
        P_diag = np.zeros(n_total)
        P_diag[i_c:i_c + n_c] = eps * G_diag          # (eps/2)*G -> eps*G in P
        P_diag[i_xi:i_xi + n_n] = 2.0 * self.C * sw_eff  # C*sw*xi^2 -> 2*C*sw in P
        P_osqp = sp.diags(P_diag, format="csc")

        # q vector (linear costs)
        q_osqp = np.zeros(n_total)
        q_osqp[i_gp:i_gp + n_k] = w_atoms   # atomic norm gamma_plus
        q_osqp[i_gm:i_gm + n_k] = w_atoms   # atomic norm gamma_minus

        # ------------------------------------------------------------------
        # Constraint matrix A_osqp
        # Row group 1 [n_c]: c - B*gamma_plus + B*gamma_minus = 0
        # Row group 2 [n_n]: y*Psi*c + xi >= 1
        # Row group 3 [n_k]: gamma_plus >= 0
        # Row group 4 [n_k]: gamma_minus >= 0
        # Row group 5 [n_n]: xi >= 0
        # ------------------------------------------------------------------
        n_rows = n_c + n_n + 2 * n_k + n_n

        rows_list: list[int] = []
        cols_list: list[int] = []
        vals_list: list[float] = []

        ro = 0  # row offset

        # Group 1: c - B*gamma_plus + B*gamma_minus = 0
        # I_{n_c} at c
        for i in range(n_c):
            rows_list.append(ro + i)
            cols_list.append(i_c + i)
            vals_list.append(1.0)
        # -B at gamma_plus
        B_coo = sp.coo_matrix(B)
        for r, c_idx, val in zip(B_coo.row, B_coo.col, B_coo.data):
            rows_list.append(ro + r)
            cols_list.append(i_gp + c_idx)
            vals_list.append(-float(val))
        # +B at gamma_minus
        for r, c_idx, val in zip(B_coo.row, B_coo.col, B_coo.data):
            rows_list.append(ro + r)
            cols_list.append(i_gm + c_idx)
            vals_list.append(float(val))
        l_1 = np.zeros(n_c)
        u_1 = np.zeros(n_c)
        ro += n_c

        # Group 2: y_i * Psi[i,:] @ c + xi_i >= 1
        yPsi = y_signed[:, np.newaxis] * Psi  # (n, D)
        yPsi_coo = sp.coo_matrix(yPsi)
        for r, c_idx, val in zip(yPsi_coo.row, yPsi_coo.col, yPsi_coo.data):
            rows_list.append(ro + r)
            cols_list.append(i_c + c_idx)
            vals_list.append(float(val))
        for i in range(n_n):
            rows_list.append(ro + i)
            cols_list.append(i_xi + i)
            vals_list.append(1.0)
        l_2 = np.ones(n_n)
        u_2 = np.full(n_n, np.inf)
        ro += n_n

        # Group 3: gamma_plus >= 0
        for i in range(n_k):
            rows_list.append(ro + i)
            cols_list.append(i_gp + i)
            vals_list.append(1.0)
        l_3 = np.zeros(n_k)
        u_3 = np.full(n_k, np.inf)
        ro += n_k

        # Group 4: gamma_minus >= 0
        for i in range(n_k):
            rows_list.append(ro + i)
            cols_list.append(i_gm + i)
            vals_list.append(1.0)
        l_4 = np.zeros(n_k)
        u_4 = np.full(n_k, np.inf)
        ro += n_k

        # Group 5: xi >= 0
        for i in range(n_n):
            rows_list.append(ro + i)
            cols_list.append(i_xi + i)
            vals_list.append(1.0)
        l_5 = np.zeros(n_n)
        u_5 = np.full(n_n, np.inf)

        A_osqp = sp.csc_matrix(
            (vals_list, (rows_list, cols_list)),
            shape=(n_rows, n_total),
        )
        l_osqp = np.concatenate([l_1, l_2, l_3, l_4, l_5])
        u_osqp = np.concatenate([u_1, u_2, u_3, u_4, u_5])

        # ------------------------------------------------------------------
        # Solve with OSQP
        # ------------------------------------------------------------------
        prob = osqp.OSQP()
        prob.setup(
            P_osqp,
            q_osqp,
            A_osqp,
            l_osqp,
            u_osqp,
            warm_starting=True,
            max_iter=self.max_iter,
            eps_abs=self.tol,
            eps_rel=self.tol,
            verbose=False,
        )
        res = prob.solve()

        status_str = str(res.info.status)
        solved_statuses = {"solved", "solved_inaccurate"}
        # OSQP v1 may use enum status; convert to lowercase string
        if hasattr(res.info.status, "name"):
            status_str = res.info.status.name.lower().replace("osqp_", "")

        if status_str not in solved_statuses:
            raise RuntimeError(
                f"OSQP did not converge (status={status_str!r}). "
                "Increase max_iter or relax tol."
            )

        if status_str == "solved_inaccurate":
            warnings.warn(
                "QuotientAtomicFuzzySVM: OSQP returned 'solved_inaccurate'. "
                "Increase max_iter or tol for better accuracy.",
                RuntimeWarning,
                stacklevel=2,
            )

        z = res.x  # solution vector
        c_fitted = z[i_c:i_c + n_c].copy()

        # Expose the unique semantic c (not gamma)
        self.c_ = c_fitted
        self.intercept_ = float(c_fitted[0])
        self.coef_ = c_fitted[1:]
        self.n_iter_ = int(res.info.iter)
        self.solver_status_ = status_str

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not hasattr(self, "c_"):
            raise RuntimeError(
                "QuotientAtomicFuzzySVM is not fitted. Call fit() first."
            )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return canonical feature matrix Psi_bar(X), shape (n, D)."""
        self._check_fitted()
        from sklearn.utils.validation import check_array
        X_check = check_array(X, dtype=np.float64)
        if X_check.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_check.shape[1]} features but model was fitted "
                f"with {self.n_features_in_}."
            )
        return canonical_feature_matrix(X_check, self.basis_, self.partitions_)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return signed canonical margin f(x) = psi_bar(x)^T c."""
        self._check_fitted()
        Psi = self.transform(X)
        return Psi @ self.c_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._check_fitted()
        margins = self.decision_function(X)
        return np.where(margins >= 0.0, self.classes_[1], self.classes_[0])

    # ------------------------------------------------------------------
    # Fixed property
    # ------------------------------------------------------------------

    @property
    def and_operator(self) -> str:
        return "product"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_partitions(self, X: np.ndarray) -> list[_FuzzyPartition]:
        """Fit low/medium/high partition anchors from training data."""
        quantiles = np.quantile(X, self.partition_quantiles, axis=0)
        return [
            _FuzzyPartition(
                low=float(quantiles[0, j]),
                medium=float(quantiles[1, j]),
                high=float(quantiles[2, j]),
            )
            for j in range(X.shape[1])
        ]

    def _degree_weights(self, basis: CanonicalBasis) -> np.ndarray:
        """Compute degree weights p_q for each canonical monomial."""
        D = basis.dimension
        p = np.empty(D, dtype=np.float64)
        for i, mono in enumerate(basis.monomials):
            deg = mono.degree
            if deg == 0:
                p[i] = float(self.intercept_penalty)
            else:
                p[i] = 1.0 + float(self.degree_penalty) * float(deg - 1)
        return p

    def _signed_labels(self, y: np.ndarray) -> np.ndarray:
        """Convert {class0, class1} labels to {-1, +1}."""
        signed = np.empty(y.shape[0], dtype=np.float64)
        signed[y == self.classes_[1]] = 1.0
        signed[y == self.classes_[0]] = -1.0
        return signed

    def _apply_class_weights(
        self, y: np.ndarray, sample_weight: np.ndarray
    ) -> np.ndarray:
        """Fold class weights into sample weights."""
        if self.class_weight is None:
            return sample_weight
        if self.class_weight == "balanced":
            n = len(y)
            n_classes = len(self.classes_)
            weights = {
                cls: n / (n_classes * max(int(np.sum(y == cls)), 1))
                for cls in self.classes_
            }
        else:
            weights = dict(self.class_weight)
        sw = sample_weight.copy()
        for i, yi in enumerate(y):
            if yi in weights:
                sw[i] *= weights[yi]
        return sw
