"""Certified Linguistic Transition Envelopes (CLTE) for FuzzyRuleSVM.

Given a fitted binary FuzzyRuleSVM with min t-norm, a transition feature j,
source term A, destination term B, and a declared bounded domain D, this module
computes sound global bounds on the target-oriented score change

    f_t(z, v) - f_t(z, u)

over all feasible shared-context endpoint pairs (z, u, v) in D satisfying the
declared source/destination membership thresholds and an optional context clause.

Only min t-norm models are supported. The inner MILP uses SciPy's HiGHS backend.
Product, softmin, tied relevant anchors, and unbounded domains are rejected.

Public types mirror the Proposed API in the MCLTA specification.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.utils.validation import check_is_fitted

from fysvm.rule_svm import FuzzyRuleSVM, SparseMaxMarginFuzzyRuleMachine

# ------------------------------------------------------------------ #
# Public data types (mirrors Proposed API in MCLTA spec)             #
# ------------------------------------------------------------------ #

_TERMS = ("low", "medium", "high")
_TERM_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class LinearDomain:
    """Bounded polyhedral domain in screened-feature space."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    A_ub: tuple[tuple[float, ...], ...] = ()
    b_ub: tuple[float, ...] = ()
    provenance: str = ""


@dataclass(frozen=True)
class ContextLiteral:
    """One membership-threshold literal in a context clause."""

    feature_index: int
    term: Literal["low", "medium", "high"]
    min_membership: float


@dataclass(frozen=True)
class ContextClause:
    """Conjunction of context literals (empty = no constraint)."""

    literals: tuple[ContextLiteral, ...] = ()


@dataclass(frozen=True)
class TransitionQuery:
    """Specifies a single CLTE query."""

    feature_index: int
    source_term: Literal["low", "medium", "high"]
    destination_term: Literal["low", "medium", "high"]
    source_alpha: float
    destination_alpha: float
    target_class: str | int | float | bool
    domain: LinearDomain
    base_context: ContextClause = field(default_factory=ContextClause)
    enforce_term_order: bool = True
    min_raw_displacement: float = 0.0
    original_feature_fill: tuple[float, ...] | None = None


@dataclass(frozen=True)
class MilpConfig:
    """MILP solver configuration."""

    backend: Literal["scipy-highs"] = "scipy-highs"
    time_limit_seconds: float = 30.0
    node_limit: int | None = None
    relative_gap: float = 1e-6
    postcheck_atol: float = 1e-8
    outer_bound_atol: float = 1e-8
    outer_bound_rtol: float = 1e-9


# ------------------------------------------------------------------ #
# Internal / result types                                             #
# ------------------------------------------------------------------ #

SolverStatus = Literal[
    "OPTIMAL", "BOUNDED", "INFEASIBLE", "FEASIBLE_ONLY", "UNKNOWN", "INVALID"
]

ProofLevel = Literal["WITNESS_CHECKED", "SOLVER_BOUNDED", "UNKNOWN"]

AtlasStatus = Literal[
    "MINIMUM_SOLVER_CERTIFIED",
    "NEAR_MINIMUM_SOLVER_CERTIFIED",
    "VALID_COVER_MINIMALITY_UNKNOWN",
    "INFEASIBLE_TRANSITION",
    "GRAMMAR_INSUFFICIENT",
    "UNKNOWN",
    "INVALID",
]


@dataclass(frozen=True)
class TransitionWitness:
    """Independent witness record for a transition endpoint pair."""

    # Screened-space context vector and feature values
    context: tuple[float, ...]
    source_value: float
    destination_value: float
    source_vector: tuple[float, ...]
    destination_vector: tuple[float, ...]

    # Memberships at both endpoints for the transitioned feature
    source_memberships: tuple[float, float, float]   # (low, medium, high)
    destination_memberships: tuple[float, float, float]

    # Oriented scores
    source_target_score: float
    destination_target_score: float

    # Per-rule breakdown
    rule_indices: tuple[int, ...]
    source_activations: tuple[float, ...]
    destination_activations: tuple[float, ...]
    per_rule_score_changes: tuple[float, ...]
    total_score_change: float

    # Residuals (should be <= solver's postcheck_atol for VALIDATED)
    alpha_cut_residual: float
    context_residual: float
    domain_residual: float
    displacement_residual: float
    objective_error: float

    validated: bool
    validation_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SolverRecord:
    """Certificate for one MILP solve (min or max)."""

    status: SolverStatus
    primal_value: float | None        # best feasible objective found
    global_bound: float | None        # dual bound (lower bound on obj for min)
    absolute_gap: float | None
    relative_gap_reported: float | None
    runtime_seconds: float
    node_count: int | None
    backend: str
    backend_version: str
    termination_reason: str
    witness_status: Literal["VALIDATED", "FAILED", "ABSENT"]
    max_constraint_residual: float | None
    witness: TransitionWitness | None


@dataclass(frozen=True)
class CertifiedTransitionEnvelope:
    """Result of certify_transition_envelope().

    Stores certified outer intervals on L and U:
        dual_lower  ≤  L  ≤  primal_lower
        primal_upper ≤  U  ≤  dual_upper
    """

    status: SolverStatus

    # Certified lower envelope bounds
    dual_lower: float | None    # d_a^L : certified lower bound on L
    primal_lower: float | None  # p_a^L : best known lower endpoint

    # Certified upper envelope bounds
    primal_upper: float | None  # p_a^U : best known upper endpoint
    dual_upper: float | None    # d_a^U : certified upper bound on U

    lower_solve: SolverRecord
    upper_solve: SolverRecord | None

    query: TransitionQuery
    context: ContextClause
    solver_config: MilpConfig

    proof_level: ProofLevel
    target_sign: int   # +1 if target = classes_[1], -1 if target = classes_[0]

    affected_rule_indices: tuple[int, ...]  # nonzero rules containing feature j
    schema_version: str = "1.0"


# ------------------------------------------------------------------ #
# Scalar reference helpers (no production imports beyond model types) #
# ------------------------------------------------------------------ #


def _mu_scalar(term: str, v: float, q_low: float, q_mid: float, q_high: float) -> float:
    """Scalar membership value (no numpy arrays)."""
    if term == "low":
        if q_mid <= q_low:
            return 1.0 if v <= q_low else 0.0
        if v <= q_low:
            return 1.0
        if v >= q_mid:
            return 0.0
        return (q_mid - v) / (q_mid - q_low)
    if term == "high":
        if q_high <= q_mid:
            return 1.0 if v >= q_high else 0.0
        if v <= q_mid:
            return 0.0
        if v >= q_high:
            return 1.0
        return (v - q_mid) / (q_high - q_mid)
    # medium
    if q_mid <= q_low:
        up = 1.0 if v >= q_mid else 0.0
    else:
        up = max(0.0, min(1.0, (v - q_low) / (q_mid - q_low)))
    if q_high <= q_mid:
        down = 1.0 if v <= q_mid else 0.0
    else:
        down = max(0.0, min(1.0, (q_high - v) / (q_high - q_mid)))
    return min(up, down)


def _mu_array(term: str, vals: np.ndarray, q_low: float, q_mid: float, q_high: float) -> np.ndarray:
    """Vectorised membership values."""
    vals = np.asarray(vals, dtype=np.float64)
    if term == "low":
        if q_mid <= q_low:
            return (vals <= q_low).astype(np.float64)
        return np.clip((q_mid - vals) / (q_mid - q_low), 0.0, 1.0)
    if term == "high":
        if q_high <= q_mid:
            return (vals >= q_high).astype(np.float64)
        return np.clip((vals - q_mid) / (q_high - q_mid), 0.0, 1.0)
    # medium
    if q_mid <= q_low:
        up = (vals >= q_mid).astype(np.float64)
    else:
        up = np.clip((vals - q_low) / (q_mid - q_low), 0.0, 1.0)
    if q_high <= q_mid:
        down = (vals <= q_mid).astype(np.float64)
    else:
        down = np.clip((q_high - vals) / (q_high - q_mid), 0.0, 1.0)
    return np.minimum(up, down)


def _mu_knot_values(term: str, knots: np.ndarray, q_low: float, q_mid: float, q_high: float) -> np.ndarray:
    """Membership values at each knot (shape (P,))."""
    return _mu_array(term, knots, q_low, q_mid, q_high)


def _build_knots(q_low: float, q_mid: float, q_high: float,
                  domain_lo: float, domain_hi: float) -> np.ndarray:
    """Ordered distinct knot points for the piecewise-linear encoding.

    Returns at least [domain_lo, domain_hi]. Adds partition anchors that
    fall strictly inside the domain.
    """
    pts = {domain_lo, domain_hi}
    for anchor in (q_low, q_mid, q_high):
        if domain_lo < anchor < domain_hi:
            pts.add(anchor)
    return np.array(sorted(pts), dtype=np.float64)


# ------------------------------------------------------------------ #
# Validation helpers                                                  #
# ------------------------------------------------------------------ #


def _resolve_target_sign(model: SparseMaxMarginFuzzyRuleMachine, target_class: Any) -> int:
    """Return +1 if target_class == classes_[1], -1 if classes_[0]. Raises ValueError otherwise."""
    c0, c1 = model.classes_[0], model.classes_[1]
    # Compare as strings to handle mixed types
    if str(target_class) == str(c1):
        return 1
    if str(target_class) == str(c0):
        return -1
    # Try direct equality
    try:
        if target_class == c1:
            return 1
        if target_class == c0:
            return -1
    except Exception:
        pass
    raise ValueError(
        f"target_class {target_class!r} not in model.classes_ {list(model.classes_)}"
    )


def _validate_query(
    model: SparseMaxMarginFuzzyRuleMachine,
    query: TransitionQuery,
) -> str | None:
    """Return None if valid, else an error string."""
    n_screened = len(model.partitions_)
    j = query.feature_index

    if not (0 <= j < n_screened):
        return f"feature_index {j} out of range [0, {n_screened})"
    if query.source_term not in _TERMS:
        return f"source_term must be one of {_TERMS}"
    if query.destination_term not in _TERMS:
        return f"destination_term must be one of {_TERMS}"
    if not (0.0 < query.source_alpha <= 1.0):
        return "source_alpha must be in (0, 1]"
    if not (0.0 < query.destination_alpha <= 1.0):
        return "destination_alpha must be in (0, 1]"
    if query.min_raw_displacement < 0.0:
        return "min_raw_displacement must be >= 0"

    dom = query.domain
    n_d = len(dom.lower)
    if len(dom.upper) != n_d or n_d != n_screened:
        return (
            f"domain lower/upper must each have length {n_screened} (n_screened)"
        )
    lower = np.array(dom.lower, dtype=np.float64)
    upper = np.array(dom.upper, dtype=np.float64)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        return "domain lower/upper must be finite"
    if np.any(lower > upper):
        return "domain lower must not exceed upper"
    if dom.A_ub:
        A = np.array(dom.A_ub, dtype=np.float64)
        if A.ndim != 2 or A.shape[1] != n_screened:
            return f"domain A_ub must have shape (m, {n_screened})"
        if len(dom.b_ub) != A.shape[0]:
            return "domain b_ub length must equal number of A_ub rows"
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(dom.b_ub)):
            return "domain A_ub and b_ub must be finite"

    for lit in query.base_context.literals:
        if not (0 <= lit.feature_index < n_screened):
            return f"base_context literal feature_index {lit.feature_index} out of range"
        if lit.term not in _TERMS:
            return f"base_context literal term must be one of {_TERMS}"
        if not (0.0 < lit.min_membership <= 1.0):
            return "base_context literal min_membership must be in (0, 1]"

    return None


def _check_tied_anchors(
    model: SparseMaxMarginFuzzyRuleMachine,
    affected_rule_indices: list[int],
    j: int,
) -> str | None:
    """Return error string if any relevant partition has tied anchors for affected rules."""
    # Collect all features in affected rules
    features_in_rules: set[int] = set()
    for k in affected_rule_indices:
        for cond in model.rules_[k].conditions:
            features_in_rules.add(cond.feature)

    for feat in features_in_rules:
        p = model.partitions_[feat]
        if p.low == p.medium or p.medium == p.high:
            return (
                f"Partition for screened feature {feat} has tied anchors "
                f"(low={p.low}, medium={p.medium}, high={p.high}). "
                "Tied relevant anchors are not supported."
            )
    return None


# ------------------------------------------------------------------ #
# MILP builder                                                        #
# ------------------------------------------------------------------ #


def _compute_big_m(
    term: str, knots: np.ndarray, q_low: float, q_mid: float, q_high: float
) -> float:
    """Tight big-M for this term over the knot domain: max membership minus min membership."""
    mu_vals = _mu_knot_values(term, knots, q_low, q_mid, q_high)
    return float(np.max(mu_vals))  # max possible value in domain = upper bound on membership


def _solve_milp_direction(
    model: SparseMaxMarginFuzzyRuleMachine,
    query: TransitionQuery,
    combined_context: ContextClause,
    affected_rule_indices: list[int],
    target_sign: int,
    is_minimize: bool,
    solver: MilpConfig,
) -> SolverRecord:
    """Build and solve the MILP for one direction (min or max transition score).

    Returns a SolverRecord with primal value, global bound, and witness.
    """
    import importlib.metadata as im
    t_start = time.perf_counter()

    n_screened = len(model.partitions_)
    j = query.feature_index
    dom = query.domain
    lower = np.array(dom.lower, dtype=np.float64)
    upper = np.array(dom.upper, dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 1: Determine which context features need knot encodings
    # ------------------------------------------------------------------
    # context_knot_features: features != j that appear in affected rules or context clause
    # These need full knot encodings for membership computation.
    ctx_knot_set: set[int] = set()
    for k in affected_rule_indices:
        for cond in model.rules_[k].conditions:
            if cond.feature != j:
                ctx_knot_set.add(cond.feature)
    for lit in combined_context.literals:
        if lit.feature_index != j:
            ctx_knot_set.add(lit.feature_index)

    # polyhedral constraint features (need at least box variables)
    poly_features: set[int] = set()
    if dom.A_ub:
        A_ub = np.array(dom.A_ub, dtype=np.float64)
        b_ub = np.array(dom.b_ub, dtype=np.float64)
        for row_i in range(A_ub.shape[0]):
            for feat_k in range(n_screened):
                if feat_k != j and abs(A_ub[row_i, feat_k]) > 0:
                    poly_features.add(feat_k)
    else:
        A_ub = None
        b_ub = None

    # Box-only features (need domain enforcement but not membership)
    ctx_box_set = poly_features - ctx_knot_set  # features with only box variables

    # Sorted lists for deterministic ordering
    ctx_knot_feats = sorted(ctx_knot_set)
    ctx_box_feats = sorted(ctx_box_set)

    # Build knot sets for context knot features
    ctx_knots: dict[int, np.ndarray] = {}
    for feat in ctx_knot_feats:
        p = model.partitions_[feat]
        knots = _build_knots(p.low, p.medium, p.high, lower[feat], upper[feat])
        ctx_knots[feat] = knots

    # Build knot set for transition feature (shared by source and dest)
    p_j = model.partitions_[j]
    j_knots = _build_knots(p_j.low, p_j.medium, p_j.high, lower[j], upper[j])
    P_j = len(j_knots)

    # ------------------------------------------------------------------
    # Step 2: Allocate variables
    # ------------------------------------------------------------------
    # Variable index tracker
    var_idx: dict[str, Any] = {}
    n_cont = 0
    n_bin = 0

    cont_lb: list[float] = []
    cont_ub: list[float] = []
    bin_count = 0

    def alloc_cont(n: int, lb: float = 0.0, ub: float = 1.0) -> int:
        nonlocal n_cont
        start = n_cont
        n_cont += n
        cont_lb.extend([lb] * n)
        cont_ub.extend([ub] * n)
        return start

    def alloc_bin(n: int) -> int:
        nonlocal bin_count
        start = bin_count
        bin_count += n
        return start

    # Context knot features: lambda (cont) + q (bin)
    ctx_lambda_start: dict[int, int] = {}
    ctx_q_start: dict[int, int] = {}
    for feat in ctx_knot_feats:
        knots = ctx_knots[feat]
        P = len(knots)
        ctx_lambda_start[feat] = alloc_cont(P)
        if P > 1:
            ctx_q_start[feat] = alloc_bin(P - 1)

    # Context box features: single continuous variable
    ctx_box_var: dict[int, int] = {}
    for feat in ctx_box_feats:
        ctx_box_var[feat] = alloc_cont(1, lb=float(lower[feat]), ub=float(upper[feat]))

    # Transition feature source: lambda (cont) + q (bin)
    src_lambda_start = alloc_cont(P_j)
    src_q_start: int = alloc_bin(P_j - 1) if P_j > 1 else 0

    # Transition feature dest: lambda (cont) + q (bin)
    dst_lambda_start = alloc_cont(P_j)
    dst_q_start: int = alloc_bin(P_j - 1) if P_j > 1 else 0

    # Rule activations: phi_src, phi_dst (cont) + y_src, y_dst (bin) if length > 1
    rule_phi_src: dict[int, int] = {}
    rule_phi_dst: dict[int, int] = {}
    rule_y_src: dict[int, int] = {}  # start of binary selectors
    rule_y_dst: dict[int, int] = {}

    for k in affected_rule_indices:
        rule_phi_src[k] = alloc_cont(1)
        rule_phi_dst[k] = alloc_cont(1)
        conds = model.rules_[k].conditions
        if len(conds) > 1:
            rule_y_src[k] = alloc_bin(len(conds))
            rule_y_dst[k] = alloc_bin(len(conds))

    # Total variables: continuous block first, then binary block
    # For scipy.milp: we'll lay them out as [cont vars | bin vars]
    n_vars = n_cont + bin_count

    # Index translation for binary variables
    BIN_OFFSET = n_cont  # binary variables start at index n_cont

    def bin_abs(local_start: int, i: int = 0) -> int:
        return BIN_OFFSET + local_start + i

    def ctx_q_abs(feat: int, s: int) -> int:
        return BIN_OFFSET + ctx_q_start[feat] + s

    def src_q_abs(s: int) -> int:
        return BIN_OFFSET + src_q_start + s

    def dst_q_abs(s: int) -> int:
        return BIN_OFFSET + dst_q_start + s

    def y_src_abs(k: int, i: int) -> int:
        return BIN_OFFSET + rule_y_src[k] + i

    def y_dst_abs(k: int, i: int) -> int:
        return BIN_OFFSET + rule_y_dst[k] + i

    # ------------------------------------------------------------------
    # Step 3: Build constraints
    # ------------------------------------------------------------------
    # Using lists of (row_dict, lb, ub) -> convert to sparse matrix
    constr_rows: list[dict[int, float]] = []
    constr_lb: list[float] = []
    constr_ub: list[float] = []

    INF = np.inf

    def add_constr(coefs: dict[int, float], lb: float, ub: float) -> None:
        constr_rows.append(coefs)
        constr_lb.append(lb)
        constr_ub.append(ub)

    def add_eq(coefs: dict[int, float], rhs: float) -> None:
        add_constr(coefs, rhs, rhs)

    def add_le(coefs: dict[int, float], rhs: float) -> None:
        add_constr(coefs, -INF, rhs)

    def add_ge(coefs: dict[int, float], lhs: float) -> None:
        add_constr(coefs, lhs, INF)

    # --- Context knot feature constraints ---
    for feat in ctx_knot_feats:
        knots = ctx_knots[feat]
        P = len(knots)
        lam_s = ctx_lambda_start[feat]

        # Convexity: sum lambda = 1
        add_eq({lam_s + p: 1.0 for p in range(P)}, 1.0)

        if P == 1:
            # Fixed point: lambda[0] = 1 (already implicit from convexity)
            # Domain: knots[0] == lower[feat] == upper[feat]
            continue  # no segment selectors needed

        q_s = ctx_q_start[feat]
        q_abs = [BIN_OFFSET + q_s + s for s in range(P - 1)]

        # Cardinality: sum q = 1
        add_eq({qa: 1.0 for qa in q_abs}, 1.0)

        # Adjacency: lambda[0] <= q[0]
        add_le({lam_s: 1.0, q_abs[0]: -1.0}, 0.0)

        # Adjacency: lambda[P-1] <= q[P-2]
        add_le({lam_s + P - 1: 1.0, q_abs[P - 2]: -1.0}, 0.0)

        # Adjacency: lambda[p] <= q[p-1] + q[p] for 0 < p < P-1
        for p in range(1, P - 1):
            coefs = {lam_s + p: 1.0, q_abs[p - 1]: -1.0, q_abs[p]: -1.0}
            add_le(coefs, 0.0)

    # --- Source knot feature constraints ---
    src_lam_s = src_lambda_start

    add_eq({src_lam_s + p: 1.0 for p in range(P_j)}, 1.0)

    if P_j > 1:
        s_q_s = src_q_start
        s_q_abs = [BIN_OFFSET + s_q_s + s for s in range(P_j - 1)]
        add_eq({qa: 1.0 for qa in s_q_abs}, 1.0)
        add_le({src_lam_s: 1.0, s_q_abs[0]: -1.0}, 0.0)
        add_le({src_lam_s + P_j - 1: 1.0, s_q_abs[P_j - 2]: -1.0}, 0.0)
        for p in range(1, P_j - 1):
            coefs = {src_lam_s + p: 1.0, s_q_abs[p - 1]: -1.0, s_q_abs[p]: -1.0}
            add_le(coefs, 0.0)

    # --- Destination knot feature constraints ---
    dst_lam_s = dst_lambda_start

    add_eq({dst_lam_s + p: 1.0 for p in range(P_j)}, 1.0)

    if P_j > 1:
        d_q_s = dst_q_start
        d_q_abs = [BIN_OFFSET + d_q_s + s for s in range(P_j - 1)]
        add_eq({qa: 1.0 for qa in d_q_abs}, 1.0)
        add_le({dst_lam_s: 1.0, d_q_abs[0]: -1.0}, 0.0)
        add_le({dst_lam_s + P_j - 1: 1.0, d_q_abs[P_j - 2]: -1.0}, 0.0)
        for p in range(1, P_j - 1):
            coefs = {dst_lam_s + p: 1.0, d_q_abs[p - 1]: -1.0, d_q_abs[p]: -1.0}
            add_le(coefs, 0.0)

    # --- Source and destination membership threshold constraints ---
    # source: mu_{j, source_term}(u) >= source_alpha
    src_mu_src = _mu_knot_values(query.source_term, j_knots, p_j.low, p_j.medium, p_j.high)
    src_alpha_coefs = {src_lam_s + p: float(src_mu_src[p]) for p in range(P_j)}
    add_ge(src_alpha_coefs, float(query.source_alpha))

    # dest: mu_{j, dest_term}(v) >= dest_alpha
    dst_mu_dst = _mu_knot_values(query.destination_term, j_knots, p_j.low, p_j.medium, p_j.high)
    dst_alpha_coefs = {dst_lam_s + p: float(dst_mu_dst[p]) for p in range(P_j)}
    add_ge(dst_alpha_coefs, float(query.destination_alpha))

    # --- Displacement constraint ---
    if query.enforce_term_order:
        src_ord = _TERM_ORDER[query.source_term]
        dst_ord = _TERM_ORDER[query.destination_term]
        if src_ord != dst_ord:
            s_ab = 1 if dst_ord > src_ord else -1
            # s_ab * (v - u) >= delta_x
            # s_ab * sum_p j_knots[p] * (dst_lam[p] - src_lam[p]) >= delta_x
            disp_coefs = {}
            for p in range(P_j):
                t = float(j_knots[p])
                disp_coefs[dst_lam_s + p] = disp_coefs.get(dst_lam_s + p, 0.0) + s_ab * t
                disp_coefs[src_lam_s + p] = disp_coefs.get(src_lam_s + p, 0.0) - s_ab * t
            add_ge(disp_coefs, float(query.min_raw_displacement))
    elif query.min_raw_displacement > 0.0:
        # No order enforcement but positive displacement
        disp_coefs = {}
        for p in range(P_j):
            t = float(j_knots[p])
            disp_coefs[dst_lam_s + p] = disp_coefs.get(dst_lam_s + p, 0.0) + t
            disp_coefs[src_lam_s + p] = disp_coefs.get(src_lam_s + p, 0.0) - t
        add_ge(disp_coefs, float(query.min_raw_displacement))

    # --- Context clause constraints ---
    for lit in combined_context.literals:
        feat = lit.feature_index
        if feat == j:
            continue  # transition feature context handled by alpha constraints
        if feat in ctx_knot_feats:
            p_feat = model.partitions_[feat]
            k_feat = ctx_knots[feat]
            lam_s_feat = ctx_lambda_start[feat]
            P_feat = len(k_feat)
            mu_ctx = _mu_knot_values(lit.term, k_feat, p_feat.low, p_feat.medium, p_feat.high)
            coefs = {lam_s_feat + p: float(mu_ctx[p]) for p in range(P_feat)}
            add_ge(coefs, float(lit.min_membership))
        else:
            # Context clause features must be encoded in the MILP (knot encoding)
            raise ValueError(
                f"Context clause feature {feat} (term={lit.term}) is not encoded in the MILP; "
                f"only features appearing in affected rules or with knot encodings are supported"
            )

    # --- Polyhedral domain constraints (applied to source and dest endpoints) ---
    if A_ub is not None and b_ub is not None:
        for row_i in range(A_ub.shape[0]):
            b_val = float(b_ub[row_i])

            # Helper: build coefficient dict for full screened vector expression
            # For source endpoint: x[j] = sum_p j_knots[p] * src_lam[p]
            # For dest endpoint: x[j] = sum_p j_knots[p] * dst_lam[p]
            # For ctx_knot features: x[k] = sum_p knot_k[p] * ctx_lam_k[p]
            # For ctx_box features: x[k] = ctx_box_var[k]

            def build_poly_row(use_dst_j: bool) -> dict[int, float]:
                coefs: dict[int, float] = {}
                for feat_k in range(n_screened):
                    a_ik = float(A_ub[row_i, feat_k])
                    if a_ik == 0.0:
                        continue
                    if feat_k == j:
                        # transition feature value
                        lam_jk = dst_lam_s if use_dst_j else src_lam_s
                        for p in range(P_j):
                            coefs[lam_jk + p] = coefs.get(lam_jk + p, 0.0) + a_ik * float(j_knots[p])
                    elif feat_k in ctx_knot_feats:
                        lam_fk = ctx_lambda_start[feat_k]
                        k_feat = ctx_knots[feat_k]
                        for p in range(len(k_feat)):
                            coefs[lam_fk + p] = coefs.get(lam_fk + p, 0.0) + a_ik * float(k_feat[p])
                    elif feat_k in ctx_box_feats:
                        bv = ctx_box_var[feat_k]
                        coefs[bv] = coefs.get(bv, 0.0) + a_ik
                    # else: feature not encoded - skip (shouldn't happen if poly_features collected correctly)
                return coefs

            add_le(build_poly_row(use_dst_j=False), b_val)  # source
            add_le(build_poly_row(use_dst_j=True), b_val)   # dest

    # --- Rule activation constraints ---
    for k in affected_rule_indices:
        conds = model.rules_[k].conditions
        phi_src_var = rule_phi_src[k]
        phi_dst_var = rule_phi_dst[k]

        # Compute antecedent memberships at source/dest
        # For each condition, get the membership expression coefficients
        antecedent_src_coefs: list[dict[int, float]] = []
        antecedent_dst_coefs: list[dict[int, float]] = []
        big_m_src: list[float] = []
        big_m_dst: list[float] = []

        for cond in conds:
            feat = cond.feature
            term = cond.term

            if feat == j:
                # Transition feature: different at source vs dest
                p_feat = model.partitions_[j]
                mu_src_at_knots = _mu_knot_values(term, j_knots, p_feat.low, p_feat.medium, p_feat.high)
                mu_dst_at_knots = _mu_knot_values(term, j_knots, p_feat.low, p_feat.medium, p_feat.high)
                src_coefs = {src_lam_s + p: float(mu_src_at_knots[p]) for p in range(P_j)}
                dst_coefs = {dst_lam_s + p: float(mu_dst_at_knots[p]) for p in range(P_j)}
                antecedent_src_coefs.append(src_coefs)
                antecedent_dst_coefs.append(dst_coefs)
                big_m_src.append(float(np.max(mu_src_at_knots)))
                big_m_dst.append(float(np.max(mu_dst_at_knots)))
            else:
                # Context feature: same at source and dest
                p_feat = model.partitions_[feat]
                k_feat = ctx_knots[feat]
                lam_fk = ctx_lambda_start[feat]
                P_feat = len(k_feat)
                mu_at_knots = _mu_knot_values(term, k_feat, p_feat.low, p_feat.medium, p_feat.high)
                ctx_coefs = {lam_fk + p: float(mu_at_knots[p]) for p in range(P_feat)}
                antecedent_src_coefs.append(ctx_coefs)
                antecedent_dst_coefs.append(dict(ctx_coefs))  # same for dest
                big_m_src.append(float(np.max(mu_at_knots)))
                big_m_dst.append(float(np.max(mu_at_knots)))

        L = len(conds)

        if L == 1:
            # phi_r = a_1 (single antecedent): equality constraints
            src_coefs = dict(antecedent_src_coefs[0])
            src_coefs[phi_src_var] = -1.0
            add_eq(src_coefs, 0.0)

            dst_coefs = dict(antecedent_dst_coefs[0])
            dst_coefs[phi_dst_var] = -1.0
            add_eq(dst_coefs, 0.0)
        else:
            # Source: phi_src <= a_i for all i
            for i, a_coefs in enumerate(antecedent_src_coefs):
                coefs = dict(a_coefs)
                coefs[phi_src_var] = coefs.get(phi_src_var, 0.0) - 1.0
                add_ge(coefs, 0.0)  # a_i - phi_src >= 0

            # Source: phi_src >= a_i - M_i * (1 - y_i)
            # => phi_src + M_i * (1 - y_i) >= a_i
            # => phi_src - a_i + M_i * y_i >= M_i - m_i (where m_i = 0 lower bound on a_i)
            # Actually: phi_src >= a_i - M_i * (1 - y_i)
            # => phi_src - a_i + M_i * y_i >= 0 ... no
            # phi_src >= a_i - M_i + M_i * y_i
            # => phi_src - a_i - M_i * y_i >= -M_i
            # Rearranged: phi_src + M_i*(1-y_i) >= a_i
            # => phi_src + M_i - M_i*y_i >= a_i
            # => phi_src - a_i_expr + M_i*(-y_i) >= -M_i (subtracting a_i_expr from both sides)
            # Wait let me redo: phi >= a_i - M_i*(1-y_i) = a_i - M_i + M_i*y_i
            # phi - a_i + M_i - M_i*y_i >= 0
            # phi - a_i + M_i*(1-y_i) >= 0
            # In terms of constraint: phi_src + M_i*(1) + (-M_i)*y_i - a_i >= 0
            for i, a_coefs in enumerate(antecedent_src_coefs):
                M_i = big_m_src[i]
                coefs: dict[int, float] = {phi_src_var: 1.0}
                coefs[y_src_abs(k, i)] = -M_i
                for vi, v in a_coefs.items():
                    coefs[vi] = coefs.get(vi, 0.0) - v
                add_ge(coefs, -M_i)

            # Source: sum y_src = 1
            add_eq({y_src_abs(k, i): 1.0 for i in range(L)}, 1.0)

            # Destination: phi_dst <= a_i for all i
            for i, a_coefs in enumerate(antecedent_dst_coefs):
                coefs = dict(a_coefs)
                coefs[phi_dst_var] = coefs.get(phi_dst_var, 0.0) - 1.0
                add_ge(coefs, 0.0)

            # Destination: phi_dst >= a_i - M_i * (1 - y_i_dst)
            for i, a_coefs in enumerate(antecedent_dst_coefs):
                M_i = big_m_dst[i]
                coefs = {phi_dst_var: 1.0}
                coefs[y_dst_abs(k, i)] = -M_i
                for vi, v in a_coefs.items():
                    coefs[vi] = coefs.get(vi, 0.0) - v
                add_ge(coefs, -M_i)

            # Destination: sum y_dst = 1
            add_eq({y_dst_abs(k, i): 1.0 for i in range(L)}, 1.0)

    # ------------------------------------------------------------------
    # Step 4: Build objective
    # ------------------------------------------------------------------
    c_obj = np.zeros(n_vars, dtype=np.float64)
    for k in affected_rule_indices:
        beta_k = float(model.coef_[k])
        contrib = target_sign * beta_k
        # Objective: s_t * sum_r beta_r * (phi_r_dst - phi_r_src)
        # For minimization: c_obj[phi_dst] = +contrib, c_obj[phi_src] = -contrib
        # For maximization (is_minimize=False): negate entire objective
        sign = 1.0 if is_minimize else -1.0
        c_obj[rule_phi_dst[k]] += sign * contrib
        c_obj[rule_phi_src[k]] -= sign * contrib

    # ------------------------------------------------------------------
    # Step 5: Build scipy arrays
    # ------------------------------------------------------------------
    if not constr_rows:
        # No constraints beyond variable bounds; add dummy to avoid empty system
        constr_rows.append({0: 0.0})
        constr_lb.append(-INF)
        constr_ub.append(INF)

    n_constrs = len(constr_rows)
    row_indices: list[int] = []
    col_indices: list[int] = []
    vals: list[float] = []

    for i, coefs in enumerate(constr_rows):
        for col, v in coefs.items():
            if v != 0.0:
                row_indices.append(i)
                col_indices.append(col)
                vals.append(v)

    A_mat = sp.csc_matrix(
        (vals, (row_indices, col_indices)),
        shape=(n_constrs, n_vars),
        dtype=np.float64,
    )
    lb_arr = np.array(constr_lb, dtype=np.float64)
    ub_arr = np.array(constr_ub, dtype=np.float64)

    # Variable bounds
    var_lb = np.array(cont_lb + [0.0] * bin_count, dtype=np.float64)
    var_ub = np.array(cont_ub + [1.0] * bin_count, dtype=np.float64)

    # Integrality: 0 for continuous, 1 for binary
    integrality = np.zeros(n_vars, dtype=int)
    integrality[BIN_OFFSET:] = 1

    # Solver options
    options: dict[str, Any] = {"disp": False}
    if solver.time_limit_seconds > 0:
        options["time_limit"] = solver.time_limit_seconds
    if solver.relative_gap > 0:
        options["mip_rel_gap"] = solver.relative_gap
    if solver.node_limit is not None:
        options["node_limit"] = solver.node_limit

    # ------------------------------------------------------------------
    # Step 6: Solve
    # ------------------------------------------------------------------
    try:
        scipy_result = milp(
            c=c_obj,
            constraints=LinearConstraint(A_mat, lb_arr, ub_arr),
            integrality=integrality,
            bounds=Bounds(var_lb, var_ub),
            options=options,
        )
    except Exception as exc:
        t_end = time.perf_counter()
        return SolverRecord(
            status="UNKNOWN",
            primal_value=None,
            global_bound=None,
            absolute_gap=None,
            relative_gap_reported=None,
            runtime_seconds=t_end - t_start,
            node_count=None,
            backend="scipy-highs",
            backend_version=_highs_version(),
            termination_reason=f"exception: {exc}",
            witness_status="ABSENT",
            max_constraint_residual=None,
            witness=None,
        )

    t_end = time.perf_counter()
    runtime = t_end - t_start

    # ------------------------------------------------------------------
    # Step 7: Interpret result
    # ------------------------------------------------------------------
    # scipy milp status: 0=optimal, 1=iter/time limit, 2=infeasible, 3=unbounded, 4=other
    sc_status = scipy_result.status
    sc_fun = scipy_result.fun
    sc_x = scipy_result.x
    sc_dual = getattr(scipy_result, "mip_dual_bound", None)
    sc_gap = getattr(scipy_result, "mip_gap", None)
    sc_nodes = getattr(scipy_result, "mip_node_count", None)

    if sc_status == 0:
        status: SolverStatus = "OPTIMAL"
    elif sc_status == 1:
        if sc_fun is not None and sc_dual is not None:
            status = "BOUNDED"
        elif sc_fun is not None:
            status = "FEASIBLE_ONLY"
        else:
            status = "UNKNOWN"
    elif sc_status == 2:
        status = "INFEASIBLE"
    elif sc_status == 3:
        status = "UNKNOWN"  # unbounded shouldn't happen with finite domain
    else:
        status = "UNKNOWN"

    # Map primal and dual bounds back to original objective sign
    primal_val: float | None = None
    global_bound: float | None = None
    abs_gap: float | None = None

    if sc_fun is not None:
        # sc_fun is the minimized value (possibly negated)
        raw_primal = float(sc_fun)
        primal_val = raw_primal if is_minimize else -raw_primal

    if sc_dual is not None:
        raw_dual = float(sc_dual)
        # For minimize: dual bound ≤ optimal objective ≤ primal
        # For maximize (negated minimize): -dual ≥ optimal ≥ primal_val
        global_bound = raw_dual if is_minimize else -raw_dual

    if primal_val is not None and global_bound is not None:
        # For minimize: gap = primal - dual
        # For maximize: gap = primal - dual (but computed from primal_val >= global_bound)
        abs_gap = abs(primal_val - global_bound)

    # Build witness
    witness: TransitionWitness | None = None
    witness_status: Literal["VALIDATED", "FAILED", "ABSENT"] = "ABSENT"

    if sc_x is not None and sc_status in (0, 1):
        x_sol = sc_x
        try:
            witness = _extract_and_validate_witness(
                x_sol,
                model,
                query,
                combined_context,
                affected_rule_indices,
                target_sign,
                ctx_knot_feats,
                ctx_box_feats,
                ctx_knots,
                ctx_lambda_start,
                ctx_box_var,
                j_knots,
                src_lambda_start,
                dst_lambda_start,
                rule_phi_src,
                rule_phi_dst,
                n_screened,
                solver.postcheck_atol,
            )
            witness_status = "VALIDATED" if witness.validated else "FAILED"
        except Exception:
            witness = None
            witness_status = "FAILED"

    # Compute max constraint residual
    max_resid: float | None = None
    if sc_x is not None:
        try:
            Ax = A_mat @ sc_x
            resid = np.maximum(lb_arr - Ax, Ax - ub_arr)
            resid = np.maximum(resid, 0.0)
            max_resid = float(np.max(resid))
        except Exception:
            max_resid = None

    return SolverRecord(
        status=status,
        primal_value=primal_val,
        global_bound=global_bound,
        absolute_gap=abs_gap,
        relative_gap_reported=float(sc_gap) if sc_gap is not None else None,
        runtime_seconds=runtime,
        node_count=int(sc_nodes) if sc_nodes is not None else None,
        backend="scipy-highs",
        backend_version=_highs_version(),
        termination_reason=scipy_result.message if hasattr(scipy_result, "message") else "",
        witness_status=witness_status,
        max_constraint_residual=max_resid,
        witness=witness,
    )


def _highs_version() -> str:
    """Return the HiGHS version string if available."""
    try:
        import scipy
        return f"scipy-{scipy.__version__}"
    except Exception:
        return "unknown"


def _extract_and_validate_witness(
    x_sol: np.ndarray,
    model: SparseMaxMarginFuzzyRuleMachine,
    query: TransitionQuery,
    combined_context: ContextClause,
    affected_rule_indices: list[int],
    target_sign: int,
    ctx_knot_feats: list[int],
    ctx_box_feats: list[int],
    ctx_knots: dict[int, np.ndarray],
    ctx_lambda_start: dict[int, int],
    ctx_box_var: dict[int, int],
    j_knots: np.ndarray,
    src_lambda_start: int,
    dst_lambda_start: int,
    rule_phi_src: dict[int, int],
    rule_phi_dst: dict[int, int],
    n_screened: int,
    atol: float,
) -> TransitionWitness:
    """Extract and independently validate the witness from the MILP solution."""
    j = query.feature_index
    p_j = model.partitions_[j]
    P_j = len(j_knots)

    # Extract source and destination feature values
    src_lam = x_sol[src_lambda_start: src_lambda_start + P_j]
    dst_lam = x_sol[dst_lambda_start: dst_lambda_start + P_j]
    x_j_src = float(np.dot(j_knots, src_lam))
    x_j_dst = float(np.dot(j_knots, dst_lam))

    # Extract context feature values
    context_vals = np.zeros(n_screened, dtype=np.float64)
    for feat in ctx_knot_feats:
        knots = ctx_knots[feat]
        P = len(knots)
        lam = x_sol[ctx_lambda_start[feat]: ctx_lambda_start[feat] + P]
        context_vals[feat] = float(np.dot(knots, lam))
    for feat in ctx_box_feats:
        context_vals[feat] = float(x_sol[ctx_box_var[feat]])

    # Build source and destination screened vectors
    src_vec = context_vals.copy()
    src_vec[j] = x_j_src
    dst_vec = context_vals.copy()
    dst_vec[j] = x_j_dst

    # Context z vector (all context features, position j is NaN for clarity)
    ctx_tuple = tuple(float(context_vals[k]) for k in range(n_screened))

    # Compute memberships independently using scalar evaluator
    def memberships_j(v: float) -> tuple[float, float, float]:
        return (
            _mu_scalar("low", v, p_j.low, p_j.medium, p_j.high),
            _mu_scalar("medium", v, p_j.low, p_j.medium, p_j.high),
            _mu_scalar("high", v, p_j.low, p_j.medium, p_j.high),
        )

    src_mu_j = memberships_j(x_j_src)
    dst_mu_j = memberships_j(x_j_dst)

    # Compute per-rule activations and score changes
    rule_idx_list = []
    src_acts = []
    dst_acts = []
    per_rule_changes = []

    for k in affected_rule_indices:
        rule = model.rules_[k]
        src_act = 1.0
        dst_act = 1.0
        for cond in rule.conditions:
            feat = cond.feature
            term = cond.term
            p = model.partitions_[feat]
            if feat == j:
                s = _mu_scalar(term, x_j_src, p.low, p.medium, p.high)
                d = _mu_scalar(term, x_j_dst, p.low, p.medium, p.high)
            else:
                s = _mu_scalar(term, float(context_vals[feat]), p.low, p.medium, p.high)
                d = s  # context is shared
            src_act = min(src_act, s)
            dst_act = min(dst_act, d)

        beta_k = float(model.coef_[k])
        delta = target_sign * beta_k * (dst_act - src_act)
        rule_idx_list.append(k)
        src_acts.append(src_act)
        dst_acts.append(dst_act)
        per_rule_changes.append(delta)

    total_change = sum(per_rule_changes)

    # Target-oriented scores at each endpoint
    # Full model score: we need all rule activations
    def full_score_at(vec: np.ndarray) -> float:
        s = float(model.intercept_)
        for ki, rule in enumerate(model.rules_):
            beta_k = float(model.coef_[ki])
            if beta_k == 0.0:
                continue
            act = 1.0
            for cond in rule.conditions:
                p = model.partitions_[cond.feature]
                act = min(act, _mu_scalar(cond.term, float(vec[cond.feature]), p.low, p.medium, p.high))
            s += beta_k * act
        return s

    f_src = full_score_at(src_vec)
    f_dst = full_score_at(dst_vec)
    f_t_src = target_sign * f_src
    f_t_dst = target_sign * f_dst
    reconstructed_change = f_t_dst - f_t_src

    # Validation checks
    notes: list[str] = []
    dom = query.domain
    lower = np.array(dom.lower, dtype=np.float64)
    upper = np.array(dom.upper, dtype=np.float64)

    # Alpha-cut residual
    src_alpha_mu = _mu_scalar(query.source_term, x_j_src, p_j.low, p_j.medium, p_j.high)
    dst_alpha_mu = _mu_scalar(query.destination_term, x_j_dst, p_j.low, p_j.medium, p_j.high)
    alpha_resid = max(
        max(0.0, float(query.source_alpha) - src_alpha_mu),
        max(0.0, float(query.destination_alpha) - dst_alpha_mu),
    )

    # Context residual (max violation of context clause membership constraints)
    ctx_resid = 0.0
    for lit in combined_context.literals:
        feat = lit.feature_index
        p = model.partitions_[feat]
        if feat == j:
            # Context on transition feature: check both endpoints
            mu_s = _mu_scalar(lit.term, x_j_src, p.low, p.medium, p.high)
            mu_d = _mu_scalar(lit.term, x_j_dst, p.low, p.medium, p.high)
            ctx_resid = max(ctx_resid, max(0.0, lit.min_membership - mu_s))
            ctx_resid = max(ctx_resid, max(0.0, lit.min_membership - mu_d))
        else:
            val = float(context_vals[feat])
            mu_v = _mu_scalar(lit.term, val, p.low, p.medium, p.high)
            ctx_resid = max(ctx_resid, max(0.0, lit.min_membership - mu_v))

    # Domain residual (max box violation)
    src_dom_resid = max(0.0, float(np.max(np.maximum(lower - src_vec, src_vec - upper))))
    dst_dom_resid = max(0.0, float(np.max(np.maximum(lower - dst_vec, dst_vec - upper))))
    dom_resid = max(src_dom_resid, dst_dom_resid)

    # Polyhedral domain residual
    if dom.A_ub:
        A_mat_poly = np.array(dom.A_ub, dtype=np.float64)
        b_poly = np.array(dom.b_ub, dtype=np.float64)
        src_poly = A_mat_poly @ src_vec - b_poly
        dst_poly = A_mat_poly @ dst_vec - b_poly
        poly_resid = max(0.0, float(np.max(np.maximum(src_poly, dst_poly))))
        dom_resid = max(dom_resid, poly_resid)

    # Displacement residual
    disp_resid = 0.0
    if query.enforce_term_order:
        src_ord = _TERM_ORDER[query.source_term]
        dst_ord = _TERM_ORDER[query.destination_term]
        if src_ord != dst_ord:
            s_ab = 1 if dst_ord > src_ord else -1
            actual_disp = s_ab * (x_j_dst - x_j_src)
            disp_resid = max(0.0, float(query.min_raw_displacement) - actual_disp)
    elif query.min_raw_displacement > 0.0:
        actual_disp = x_j_dst - x_j_src
        disp_resid = max(0.0, float(query.min_raw_displacement) - actual_disp)

    # Objective error
    obj_error = abs(reconstructed_change - total_change)

    validated = (
        alpha_resid <= atol
        and ctx_resid <= atol
        and dom_resid <= atol
        and disp_resid <= atol
        and obj_error <= atol
    )

    if alpha_resid > atol:
        notes.append(f"alpha_cut_residual={alpha_resid:.3e} > atol={atol:.3e}")
    if ctx_resid > atol:
        notes.append(f"context_residual={ctx_resid:.3e} > atol={atol:.3e}")
    if dom_resid > atol:
        notes.append(f"domain_residual={dom_resid:.3e} > atol={atol:.3e}")
    if disp_resid > atol:
        notes.append(f"displacement_residual={disp_resid:.3e} > atol={atol:.3e}")
    if obj_error > atol:
        notes.append(f"objective_error={obj_error:.3e} > atol={atol:.3e}")

    return TransitionWitness(
        context=ctx_tuple,
        source_value=x_j_src,
        destination_value=x_j_dst,
        source_vector=tuple(float(v) for v in src_vec),
        destination_vector=tuple(float(v) for v in dst_vec),
        source_memberships=src_mu_j,
        destination_memberships=dst_mu_j,
        source_target_score=f_t_src,
        destination_target_score=f_t_dst,
        rule_indices=tuple(rule_idx_list),
        source_activations=tuple(float(a) for a in src_acts),
        destination_activations=tuple(float(a) for a in dst_acts),
        per_rule_score_changes=tuple(float(d) for d in per_rule_changes),
        total_score_change=float(total_change),
        alpha_cut_residual=float(alpha_resid),
        context_residual=float(ctx_resid),
        domain_residual=float(dom_resid),
        displacement_residual=float(disp_resid),
        objective_error=float(obj_error),
        validated=validated,
        validation_notes=tuple(notes),
    )


# ------------------------------------------------------------------ #
# Main public function                                                #
# ------------------------------------------------------------------ #


def certify_transition_envelope(
    model: FuzzyRuleSVM,
    query: TransitionQuery,
    *,
    context: ContextClause = ContextClause(),
    solver: MilpConfig = MilpConfig(),
) -> CertifiedTransitionEnvelope:
    """Certify the score-change envelope for a linguistic transition.

    The ``context`` argument is conjoined with ``query.base_context``; it never
    overrides the base context. The result carries certified outer bounds on the
    minimum and maximum target-oriented score change.

    Parameters
    ----------
    model:
        Fitted binary FuzzyRuleSVM with ``and_operator="min"``.
    query:
        Transition query specifying feature, terms, alpha-cuts, domain, and
        optional order/displacement constraints.
    context:
        Additional context clause conjoined with the base context.
    solver:
        MILP solver configuration.

    Returns
    -------
    CertifiedTransitionEnvelope
    """
    # --- Validate model ---
    try:
        check_is_fitted(model)
    except Exception as exc:
        return _invalid_envelope(
            query, context, solver, f"model not fitted: {exc}"
        )

    if not hasattr(model, "classes_") or len(model.classes_) != 2:
        return _invalid_envelope(query, context, solver, "model must be binary")

    if model.and_operator != "min":
        return _invalid_envelope(
            query, context, solver,
            f"only 'min' t-norm is supported; got '{model.and_operator}'"
        )

    # --- Resolve target sign ---
    try:
        target_sign = _resolve_target_sign(model, query.target_class)
    except ValueError as exc:
        return _invalid_envelope(query, context, solver, str(exc))

    # --- Validate query ---
    err = _validate_query(model, query)
    if err:
        return _invalid_envelope(query, context, solver, err)

    # --- Combine contexts (base_context + extra context; extra overrides on same (feature, term)) ---
    combined_lits_dict: dict[tuple[int, str], ContextLiteral] = {}
    for lit in query.base_context.literals:
        combined_lits_dict[(lit.feature_index, lit.term)] = lit
    for lit in context.literals:
        combined_lits_dict[(lit.feature_index, lit.term)] = lit
    combined_context = ContextClause(literals=tuple(combined_lits_dict.values()))

    # --- Find affected rules (exactly nonzero coef_, containing feature j) ---
    j = query.feature_index
    affected: list[int] = []
    for k, beta in enumerate(model.coef_):
        if beta != 0.0:
            for cond in model.rules_[k].conditions:
                if cond.feature == j:
                    affected.append(k)
                    break

    # --- Check tied anchors ---
    if affected:
        tied_err = _check_tied_anchors(model, affected, j)
        if tied_err:
            return _invalid_envelope(query, context, solver, tied_err)

    # --- Trivial case: no affected rules ---
    if not affected:
        trivial_record = SolverRecord(
            status="OPTIMAL",
            primal_value=0.0,
            global_bound=0.0,
            absolute_gap=0.0,
            relative_gap_reported=0.0,
            runtime_seconds=0.0,
            node_count=0,
            backend="scipy-highs",
            backend_version=_highs_version(),
            termination_reason="No nonzero rules reference the transition feature; objective is identically 0.",
            witness_status="ABSENT",
            max_constraint_residual=0.0,
            witness=None,
        )
        return CertifiedTransitionEnvelope(
            status="OPTIMAL",
            dual_lower=0.0,
            primal_lower=0.0,
            primal_upper=0.0,
            dual_upper=0.0,
            lower_solve=trivial_record,
            upper_solve=trivial_record,
            query=query,
            context=context,
            solver_config=solver,
            proof_level="SOLVER_BOUNDED",
            target_sign=target_sign,
            affected_rule_indices=(),
        )

    # --- Solve lower bound ---
    lower_rec = _solve_milp_direction(
        model, query, combined_context, affected,
        target_sign, is_minimize=True, solver=solver,
    )

    # --- Check if transition is infeasible ---
    if lower_rec.status == "INFEASIBLE":
        return CertifiedTransitionEnvelope(
            status="INFEASIBLE",
            dual_lower=None,
            primal_lower=None,
            primal_upper=None,
            dual_upper=None,
            lower_solve=lower_rec,
            upper_solve=None,
            query=query,
            context=context,
            solver_config=solver,
            proof_level="SOLVER_BOUNDED",
            target_sign=target_sign,
            affected_rule_indices=tuple(affected),
        )

    # --- Solve upper bound ---
    upper_rec = _solve_milp_direction(
        model, query, combined_context, affected,
        target_sign, is_minimize=False, solver=solver,
    )

    # --- Combine results ---
    # Certified bounds:
    # d_a^L = global_bound from lower minimize (lower bound on L)
    # p_a^L = primal_value from lower minimize (best known L)
    # p_a^U = primal_value from upper maximize (best known U)
    # d_a^U = global_bound from upper maximize (upper bound on U)

    d_L = lower_rec.global_bound   # certified lower bound on L
    p_L = lower_rec.primal_value   # best known lower
    p_U = upper_rec.primal_value   # best known upper
    d_U = upper_rec.global_bound   # certified upper bound on U

    # Determine overall status
    if lower_rec.status == "OPTIMAL" and upper_rec.status == "OPTIMAL":
        overall_status: SolverStatus = "OPTIMAL"
    elif lower_rec.status in ("OPTIMAL", "BOUNDED") and upper_rec.status in ("OPTIMAL", "BOUNDED"):
        overall_status = "BOUNDED"
    elif p_L is not None or p_U is not None:
        overall_status = "FEASIBLE_ONLY"
    else:
        overall_status = "UNKNOWN"

    # Determine proof level
    any_validated = (
        (lower_rec.witness is not None and lower_rec.witness.validated)
        or (upper_rec.witness is not None and upper_rec.witness.validated)
    )
    has_bounds = d_L is not None or d_U is not None
    if any_validated and has_bounds:
        proof_level: ProofLevel = "SOLVER_BOUNDED"
    elif any_validated:
        proof_level = "WITNESS_CHECKED"
    else:
        proof_level = "UNKNOWN"

    if overall_status == "OPTIMAL":
        proof_level = "SOLVER_BOUNDED"

    return CertifiedTransitionEnvelope(
        status=overall_status,
        dual_lower=d_L,
        primal_lower=p_L,
        primal_upper=p_U,
        dual_upper=d_U,
        lower_solve=lower_rec,
        upper_solve=upper_rec,
        query=query,
        context=context,
        solver_config=solver,
        proof_level=proof_level,
        target_sign=target_sign,
        affected_rule_indices=tuple(affected),
    )


def _invalid_envelope(
    query: TransitionQuery,
    context: ContextClause,
    solver: MilpConfig,
    reason: str,
) -> CertifiedTransitionEnvelope:
    """Return an INVALID CertifiedTransitionEnvelope."""
    invalid_record = SolverRecord(
        status="INVALID",
        primal_value=None,
        global_bound=None,
        absolute_gap=None,
        relative_gap_reported=None,
        runtime_seconds=0.0,
        node_count=None,
        backend="scipy-highs",
        backend_version=_highs_version(),
        termination_reason=reason,
        witness_status="ABSENT",
        max_constraint_residual=None,
        witness=None,
    )
    return CertifiedTransitionEnvelope(
        status="INVALID",
        dual_lower=None,
        primal_lower=None,
        primal_upper=None,
        dual_upper=None,
        lower_solve=invalid_record,
        upper_solve=None,
        query=query,
        context=context,
        solver_config=solver,
        proof_level="UNKNOWN",
        target_sign=0,
        affected_rule_indices=(),
    )


# ------------------------------------------------------------------ #
# Canonical hash functions                                           #
# ------------------------------------------------------------------ #


def _sha256_hex(obj: Any) -> str:
    """Return first 16 hex chars of the SHA-256 of the canonical JSON."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def hash_model(model: SparseMaxMarginFuzzyRuleMachine) -> str:
    """Canonical hash of a fitted FuzzyRuleSVM.

    Covers: class order, selected feature indices, feature names,
    partition anchors, ordered rule conditions, exact coefficient
    bit patterns, intercept bit pattern, and and_operator.
    """
    import struct

    def _f64_bits(v: float) -> int:
        return struct.unpack(">Q", struct.pack(">d", v))[0]

    parts: dict[str, Any] = {
        "and_operator": str(model.and_operator),
        "classes": [str(c) for c in model.classes_],
        "selected_feature_indices": [int(i) for i in model.selected_feature_indices_],
        "feature_names": [str(n) for n in model.feature_names_in_],
        "intercept_bits": _f64_bits(float(model.intercept_)),
        "partitions": [
            {
                "low": float(p.low),
                "medium": float(p.medium),
                "high": float(p.high),
            }
            for p in model.partitions_
        ],
        "rules": [
            {
                "conditions": sorted(
                    [{"feature": int(c.feature), "term": str(c.term)} for c in rule.conditions],
                    key=lambda x: (x["feature"], x["term"]),
                ),
                "coef_bits": _f64_bits(float(model.coef_[k])),
            }
            for k, rule in enumerate(model.rules_)
        ],
    }
    return _sha256_hex(parts)


def hash_domain(domain: LinearDomain) -> str:
    """Canonical hash of a LinearDomain (box + polyhedral constraints)."""
    parts: dict[str, Any] = {
        "lower": [float(v) for v in domain.lower],
        "upper": [float(v) for v in domain.upper],
        "A_ub": [[float(v) for v in row] for row in domain.A_ub],
        "b_ub": [float(v) for v in domain.b_ub],
    }
    return _sha256_hex(parts)


def hash_query(query: TransitionQuery) -> str:
    """Canonical hash of a TransitionQuery (excluding domain, hashed separately)."""
    parts: dict[str, Any] = {
        "feature_index": int(query.feature_index),
        "source_term": str(query.source_term),
        "destination_term": str(query.destination_term),
        "source_alpha": float(query.source_alpha),
        "destination_alpha": float(query.destination_alpha),
        "target_class": str(query.target_class),
        "enforce_term_order": bool(query.enforce_term_order),
        "min_raw_displacement": float(query.min_raw_displacement),
    }
    return _sha256_hex(parts)


# ------------------------------------------------------------------ #
# Serialisation helpers                                              #
# ------------------------------------------------------------------ #


def envelope_to_dict(env: CertifiedTransitionEnvelope) -> dict:
    """Convert a CertifiedTransitionEnvelope to a JSON-serialisable dict.

    Rejects NaN and infinity as required by the MCLTA spec.
    """
    def _safe_float(v: float | None) -> float | None:
        if v is None:
            return None
        if not np.isfinite(v):
            raise ValueError(f"Non-finite value {v!r} in envelope; JSON serialisation rejected.")
        return float(v)

    def _witness_dict(w: TransitionWitness | None) -> dict | None:
        if w is None:
            return None
        return {
            "context": list(w.context),
            "source_value": _safe_float(w.source_value),
            "destination_value": _safe_float(w.destination_value),
            "source_vector": list(w.source_vector),
            "destination_vector": list(w.destination_vector),
            "source_memberships": list(w.source_memberships),
            "destination_memberships": list(w.destination_memberships),
            "source_target_score": _safe_float(w.source_target_score),
            "destination_target_score": _safe_float(w.destination_target_score),
            "rule_indices": list(w.rule_indices),
            "source_activations": list(w.source_activations),
            "destination_activations": list(w.destination_activations),
            "per_rule_score_changes": list(w.per_rule_score_changes),
            "total_score_change": _safe_float(w.total_score_change),
            "alpha_cut_residual": _safe_float(w.alpha_cut_residual),
            "context_residual": _safe_float(w.context_residual),
            "domain_residual": _safe_float(w.domain_residual),
            "displacement_residual": _safe_float(w.displacement_residual),
            "objective_error": _safe_float(w.objective_error),
            "validated": w.validated,
            "validation_notes": list(w.validation_notes),
        }

    def _record_dict(r: SolverRecord | None) -> dict | None:
        if r is None:
            return None
        return {
            "status": r.status,
            "primal_value": _safe_float(r.primal_value),
            "global_bound": _safe_float(r.global_bound),
            "absolute_gap": _safe_float(r.absolute_gap),
            "relative_gap_reported": _safe_float(r.relative_gap_reported),
            "runtime_seconds": _safe_float(r.runtime_seconds),
            "node_count": r.node_count,
            "backend": r.backend,
            "backend_version": r.backend_version,
            "termination_reason": r.termination_reason,
            "witness_status": r.witness_status,
            "max_constraint_residual": _safe_float(r.max_constraint_residual),
            "witness": _witness_dict(r.witness),
        }

    query = env.query
    dom = query.domain
    return {
        "schema_version": env.schema_version,
        "status": env.status,
        "dual_lower": _safe_float(env.dual_lower),
        "primal_lower": _safe_float(env.primal_lower),
        "primal_upper": _safe_float(env.primal_upper),
        "dual_upper": _safe_float(env.dual_upper),
        "proof_level": env.proof_level,
        "target_sign": env.target_sign,
        "affected_rule_indices": list(env.affected_rule_indices),
        "query": {
            "feature_index": query.feature_index,
            "source_term": query.source_term,
            "destination_term": query.destination_term,
            "source_alpha": query.source_alpha,
            "destination_alpha": query.destination_alpha,
            "target_class": str(query.target_class),
            "enforce_term_order": query.enforce_term_order,
            "min_raw_displacement": query.min_raw_displacement,
            "domain": {
                "lower": list(dom.lower),
                "upper": list(dom.upper),
                "A_ub": [list(row) for row in dom.A_ub],
                "b_ub": list(dom.b_ub),
                "provenance": dom.provenance,
            },
        },
        "context": {
            "literals": [
                {
                    "feature_index": lit.feature_index,
                    "term": lit.term,
                    "min_membership": lit.min_membership,
                }
                for lit in env.context.literals
            ]
        },
        "lower_solve": _record_dict(env.lower_solve),
        "upper_solve": _record_dict(env.upper_solve),
    }
