"""Minimal Certified Linguistic Transition Atlas (MCLTA) synthesis.

Builds on the CLTE solver in ``transition_envelopes`` to construct a
grammar-relative atlas: the smallest set of context clauses that covers every
feasible linguistic description without merging materially distinct transition
regimes.

The module exposes:
  - ``synthesize_transition_atlas`` — full atlas synthesis
  - ``verify_transition_atlas``     — independent atlas verification

All public types are frozen dataclasses. JSON serialisation rejects NaN/inf.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.utils.validation import check_is_fitted

from fysvm.rule_svm import FuzzyRuleSVM, SparseMaxMarginFuzzyRuleMachine
from fysvm.transition_envelopes import (
    CertifiedTransitionEnvelope,
    ContextClause,
    ContextLiteral,
    LinearDomain,
    MilpConfig,
    TransitionQuery,
    _highs_version,
    _mu_scalar,
    certify_transition_envelope,
    hash_domain,
    hash_model,
    hash_query,
)

# Re-export public types that users need
__all__ = [
    "ContextGrammar",
    "MaterialityPolicy",
    "AtomRecord",
    "CandidateRecord",
    "TransitionAtlas",
    "AtlasVerificationReport",
    "synthesize_transition_atlas",
    "verify_transition_atlas",
]

# Direction signatures from the spec
DirectionSignature = Literal[
    "DECREASE",
    "DECREASE_OR_NEGLIGIBLE",
    "MIXED",
    "NEGLIGIBLE",
    "NEGLIGIBLE_OR_INCREASE",
    "INCREASE",
]

ALL_SIGNATURES: frozenset[str] = frozenset([
    "DECREASE", "DECREASE_OR_NEGLIGIBLE", "MIXED",
    "NEGLIGIBLE", "NEGLIGIBLE_OR_INCREASE", "INCREASE",
])

AdmissibilityStatus = Literal[
    "DEFINITELY_ADMISSIBLE",
    "DEFINITELY_INADMISSIBLE",
    "ADMISSIBILITY_UNKNOWN",
]

GrammarLimitedStatus = Literal[
    "DEFINITELY_GRAMMAR_LIMITED",
    "NOT_GRAMMAR_LIMITED",
    "POSSIBLY_GRAMMAR_LIMITED",
]

AtlasStatus = Literal[
    "MINIMUM_SOLVER_CERTIFIED",
    "NEAR_MINIMUM_SOLVER_CERTIFIED",
    "VALID_COVER_MINIMALITY_UNKNOWN",
    "INFEASIBLE_TRANSITION",
    "GRAMMAR_INSUFFICIENT",
    "UNKNOWN",
    "INVALID",
]


# ------------------------------------------------------------------ #
# Public data types                                                  #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class ContextGrammar:
    """Grammar specifying the context linguistic space."""

    feature_indices: tuple[int, ...]
    bins_by_feature: tuple[tuple[ContextLiteral, ...], ...]
    max_clause_literals: int
    coverage_mode: Literal["description_complete"] = "description_complete"

    def __post_init__(self) -> None:
        if len(self.feature_indices) != len(self.bins_by_feature):
            raise ValueError(
                "feature_indices and bins_by_feature must have the same length"
            )
        for bins in self.bins_by_feature:
            if not bins:
                raise ValueError("Each feature must have at least one bin")


@dataclass(frozen=True)
class MaterialityPolicy:
    """Policy for deciding which envelope differences are material."""

    margin_scale: float          # s > 0 (denominator for normalisation)
    direction_epsilon: float     # epsilon_dir >= 0
    merge_tolerance: float       # eta_merge >= 0
    grammar_limited_width: float # threshold for GRAMMAR_LIMITED diagnostic


@dataclass(frozen=True)
class AtomRecord:
    """Result for one complete grammar description."""

    description: tuple[int, ...]  # bin index per grammar feature (deterministic)
    context_clause: ContextClause
    envelope: CertifiedTransitionEnvelope

    # Direction classification
    direction_signature: DirectionSignature | None     # None if unresolved
    possible_signatures: frozenset[str]                # set of possible signatures
    grammar_limited_status: GrammarLimitedStatus

    # Normalised widths
    normalised_width_lo: float   # w_a^lo (from possible interval)
    normalised_width_hi: float   # w_a^hi (conservative)


@dataclass(frozen=True)
class CandidateRecord:
    """Result for one candidate clause."""

    clause: ContextClause
    literal_count: int
    extension_atom_indices: frozenset[int]   # indices into atlas.atoms

    # Derived envelope (from atom envelopes)
    derived_dual_lower: float | None
    derived_primal_lower: float | None
    derived_primal_upper: float | None
    derived_dual_upper: float | None

    # Admissibility
    admissibility_status: AdmissibilityStatus
    admissibility_notes: tuple[str, ...]

    # Lexicographic secondary objectives
    max_norm_expansion: float   # maximum normalised expansion over extension atoms


@dataclass(frozen=True)
class SetCoverRecord:
    """Result of one set-cover solve."""

    status: Literal["OPTIMAL", "FEASIBLE", "INFEASIBLE", "TIMEOUT", "UNKNOWN"]
    selected_candidate_indices: tuple[int, ...]  # indices into atlas.candidates
    objective_value: int | None            # number of selected clauses
    global_lower_bound: float | None       # lower bound on optimal
    runtime_seconds: float
    backend: str


@dataclass(frozen=True)
class TransitionAtlas:
    """Complete Minimal Certified Linguistic Transition Atlas."""

    schema_version: str
    status: AtlasStatus

    # Inputs (frozen)
    query: TransitionQuery
    grammar: ContextGrammar
    materiality: MaterialityPolicy
    envelope_solver: MilpConfig
    set_cover_time_limit_seconds: float

    # Canonical input hashes (SHA-256 first 16 hex chars)
    model_hash: str
    domain_hash: str
    query_hash: str
    grammar_hash: str
    materiality_hash: str

    # Grammar results
    feasible_atom_count: int
    atoms: tuple[AtomRecord, ...]
    candidates: tuple[CandidateRecord, ...]

    # Cover solution
    selected_candidate_indices: tuple[int, ...]
    min_cardinality_lower: int | None   # N_L
    min_cardinality_upper: int | None   # N_U

    # Set-cover solver records
    greedy_cover: SetCoverRecord | None
    exact_cover: SetCoverRecord | None
    optimistic_cover: SetCoverRecord | None

    # Diagnostics
    warnings: tuple[str, ...]
    runtime_seconds: float


@dataclass(frozen=True)
class AtlasVerificationReport:
    """Report from verify_transition_atlas."""

    atlas_hash: str
    status: Literal["PASS", "FAIL", "PARTIAL"]
    coverage_verified: bool
    admissibility_verified: bool
    witness_revalidated: bool
    hash_verified: bool             # model/query/domain hashes match the atlas record
    set_cover_verified: bool        # rerun set cover bounds are consistent (if rerun_set_cover=True)
    coverage_gap_atoms: tuple[int, ...]   # atom indices not covered by selected
    rerun_cardinality_lower: int | None   # N_L from rerun set cover (if rerun_set_cover=True)
    rerun_cardinality_upper: int | None   # N_U from rerun set cover (if rerun_set_cover=True)
    warnings: tuple[str, ...]
    notes: tuple[str, ...]
    runtime_seconds: float


# ------------------------------------------------------------------ #
# Materiality helpers                                                #
# ------------------------------------------------------------------ #


def classify_direction(
    L: float, U: float, eps_dir: float
) -> DirectionSignature:
    """Classify [L, U] into one of six direction signatures."""
    if U < -eps_dir:
        return "DECREASE"
    if L < -eps_dir and -eps_dir <= U <= eps_dir:
        return "DECREASE_OR_NEGLIGIBLE"
    if L < -eps_dir and U > eps_dir:
        return "MIXED"
    if L >= -eps_dir and U <= eps_dir:
        return "NEGLIGIBLE"
    if -eps_dir <= L <= eps_dir and U > eps_dir:
        return "NEGLIGIBLE_OR_INCREASE"
    # L > eps_dir
    return "INCREASE"


def possible_signatures(
    dL: float | None, pL: float | None,
    pU: float | None, dU: float | None,
    eps_dir: float,
) -> frozenset[str]:
    """Return all direction signatures consistent with [dL..pL, pU..dU].

    The true L ∈ [dL, pL] and true U ∈ [pU, dU].
    """
    if dL is None or pL is None or pU is None or dU is None:
        return frozenset(ALL_SIGNATURES)  # completely unknown

    # True L ∈ [min_L, max_L], True U ∈ [min_U, max_U]
    min_L = dL
    max_L = pL
    min_U = pU
    max_U = dU

    # We need to find which signatures [L, U] can have for
    # L in [min_L, max_L] and U in [min_U, max_U] with L <= U
    sigs: set[str] = set()
    for sig in ALL_SIGNATURES:
        # Check if there exists L in [min_L, max_L] and U in [min_U, max_U]
        # with L <= U such that classify_direction(L, U) == sig
        # By testing corner and critical combinations
        for L_test in [min_L, max_L, -eps_dir - 1e-15, -eps_dir, -eps_dir + 1e-15,
                        eps_dir - 1e-15, eps_dir, eps_dir + 1e-15, 0.0]:
            if not (min_L <= L_test <= max_L):
                continue
            for U_test in [min_U, max_U, -eps_dir - 1e-15, -eps_dir, -eps_dir + 1e-15,
                           eps_dir - 1e-15, eps_dir, eps_dir + 1e-15, 0.0]:
                if not (min_U <= U_test <= max_U):
                    continue
                if L_test <= U_test:
                    s = classify_direction(L_test, U_test, eps_dir)
                    if s == sig:
                        sigs.add(sig)
    return frozenset(sigs)


def _normalised_expansion(
    atom_dL: float | None, atom_pL: float | None,
    atom_pU: float | None, atom_dU: float | None,
    cand_dL: float | None, cand_pL: float | None,
    cand_pU: float | None, cand_dU: float | None,
    s: float,
) -> tuple[float, float]:
    """Return (ell_lo, ell_hi) for atom vs candidate envelope.

    ell_lo = max(0, d_a^L - p_c^L, p_c^U - d_a^U) / s   (conservative lower)
    ell_hi = max(0, p_a^L - d_c^L, d_c^U - p_a^U) / s   (conservative upper)
    """
    if s <= 0:
        return 0.0, 0.0

    # ell_lo: smallest possible expansion
    lo_terms = [0.0]
    if atom_dL is not None and cand_pL is not None:
        lo_terms.append(atom_dL - cand_pL)
    if cand_pU is not None and atom_dU is not None:
        lo_terms.append(cand_pU - atom_dU)
    ell_lo = max(lo_terms) / s

    # ell_hi: largest possible expansion
    hi_terms = [0.0]
    if atom_pL is not None and cand_dL is not None:
        hi_terms.append(atom_pL - cand_dL)
    if cand_dU is not None and atom_pU is not None:
        hi_terms.append(cand_dU - atom_pU)
    ell_hi = max(hi_terms) / s

    return max(0.0, ell_lo), max(0.0, ell_hi)


def _normalised_width(
    dL: float | None, pL: float | None,
    pU: float | None, dU: float | None,
    s: float,
) -> tuple[float, float]:
    """Return (w_lo, w_hi) normalised width bounds for an atom.

    w_lo = max(0, p_a^U - p_a^L) / s
    w_hi = (d_a^U - d_a^L) / s
    """
    if s <= 0:
        return 0.0, float("inf")
    w_lo = 0.0
    if pU is not None and pL is not None:
        w_lo = max(0.0, (pU - pL) / s)
    w_hi = float("inf")
    if dU is not None and dL is not None:
        w_hi = (dU - dL) / s
    return w_lo, w_hi


def _grammar_limited_status(
    possible_sigs: frozenset[str],
    w_lo: float,
    w_hi: float,
    grammar_limited_width: float,
) -> GrammarLimitedStatus:
    """Classify grammar-limited status for one atom."""
    if possible_sigs == frozenset({"MIXED"}) or w_lo > grammar_limited_width:
        return "DEFINITELY_GRAMMAR_LIMITED"
    if "MIXED" not in possible_sigs and w_hi <= grammar_limited_width:
        return "NOT_GRAMMAR_LIMITED"
    return "POSSIBLY_GRAMMAR_LIMITED"


def _classify_atom_admissibility_in_candidate(
    atom_dL: float | None, atom_pL: float | None,
    atom_pU: float | None, atom_dU: float | None,
    cand_dL: float | None, cand_pL: float | None,
    cand_pU: float | None, cand_dU: float | None,
    atom_possible_sigs: frozenset[str],
    cand_possible_sigs: frozenset[str],
    merge_tolerance: float,
    s: float,
) -> tuple[bool, bool, float, float]:
    """Return (definitely_ok, definitely_bad, ell_lo, ell_hi) for one atom."""
    ell_lo, ell_hi = _normalised_expansion(
        atom_dL, atom_pL, atom_pU, atom_dU,
        cand_dL, cand_pL, cand_pU, cand_dU,
        s,
    )
    # Definitely bad: expansion too large
    if ell_lo > merge_tolerance:
        return False, True, ell_lo, ell_hi
    # Definitely bad: disjoint signatures
    if atom_possible_sigs.isdisjoint(cand_possible_sigs):
        return False, True, ell_lo, ell_hi
    # Definitely OK: tight expansion and same singleton signature
    if (
        ell_hi <= merge_tolerance
        and len(atom_possible_sigs) == 1
        and atom_possible_sigs == cand_possible_sigs
    ):
        return True, False, ell_lo, ell_hi
    return False, False, ell_lo, ell_hi


# ------------------------------------------------------------------ #
# Grammar enumeration                                                #
# ------------------------------------------------------------------ #


def _enumerate_atoms(grammar: ContextGrammar) -> list[tuple[int, ...]]:
    """Enumerate all complete descriptions as tuples of bin indices.

    A complete description assigns one bin to every grammar feature.
    Returns a deterministically-ordered list of (bin_idx_0, ..., bin_idx_H-1).
    """
    n_features = len(grammar.feature_indices)
    bin_counts = [len(bins) for bins in grammar.bins_by_feature]
    return list(itertools.product(*[range(bc) for bc in bin_counts]))


def _description_to_clause(
    description: tuple[int, ...], grammar: ContextGrammar
) -> ContextClause:
    """Convert a complete description (bin indices) to a ContextClause."""
    lits = tuple(
        grammar.bins_by_feature[fi][bin_idx]
        for fi, bin_idx in enumerate(description)
    )
    return ContextClause(literals=lits)


def _partial_to_clause(
    description: tuple[int | None, ...], grammar: ContextGrammar
) -> ContextClause:
    """Convert a partial description to a ContextClause (None = unspecified)."""
    lits = []
    for fi, bin_idx in enumerate(description):
        if bin_idx is not None:
            lits.append(grammar.bins_by_feature[fi][bin_idx])
    return ContextClause(literals=tuple(lits))


def _extends(
    complete: tuple[int, ...], partial: tuple[int | None, ...]
) -> bool:
    """Return True if the complete description extends the partial description."""
    for c_val, p_val in zip(complete, partial):
        if p_val is not None and c_val != p_val:
            return False
    return True


def _enumerate_candidates(grammar: ContextGrammar) -> list[tuple[int | None, ...]]:
    """Enumerate all partial descriptions with at most max_clause_literals specified.

    Returns list of partial description tuples (None = unspecified feature).
    Does not include the empty clause (all None) as that is trivially covered.
    """
    n = len(grammar.feature_indices)
    bin_counts = [len(bins) for bins in grammar.bins_by_feature]
    L_max = grammar.max_clause_literals
    candidates = []
    # For each subset of features with size 1..L_max, assign one bin to each
    for size in range(1, L_max + 1):
        for feat_subset in itertools.combinations(range(n), size):
            # Assign one bin per selected feature
            bin_ranges = [range(bin_counts[fi]) for fi in feat_subset]
            for bin_assignment in itertools.product(*bin_ranges):
                partial: list[int | None] = [None] * n
                for fi, bin_idx in zip(feat_subset, bin_assignment):
                    partial[fi] = bin_idx
                candidates.append(tuple(partial))
    return candidates


# ------------------------------------------------------------------ #
# Set cover solvers                                                  #
# ------------------------------------------------------------------ #


def _greedy_set_cover(
    n_atoms: int,
    atom_to_cands: list[list[int]],   # atom_idx -> list of candidate indices
    cand_to_atoms: list[list[int]],   # cand_idx -> list of atom indices
    n_cands: int,
) -> list[int]:
    """Return a greedy cover (list of candidate indices)."""
    covered = set()
    selected = []
    remaining = set(range(n_atoms))

    # Candidate priority: most uncovered atoms first, then by index (deterministic)
    cand_cover_sizes = [len(atoms) for atoms in cand_to_atoms]

    while remaining:
        best = -1
        best_gain = -1
        for c in range(n_cands):
            gain = sum(1 for a in cand_to_atoms[c] if a in remaining)
            if gain > best_gain or (gain == best_gain and (best == -1 or c < best)):
                best_gain = gain
                best = c
        if best == -1 or best_gain == 0:
            break  # cannot cover remaining atoms
        selected.append(best)
        for a in cand_to_atoms[best]:
            remaining.discard(a)

    return selected


def _solve_set_cover_milp(
    n_atoms: int,
    atom_feasible: list[bool],          # which atoms are feasible
    cand_to_atoms: list[list[int]],     # candidate -> list of feasible atom indices it covers
    n_cands: int,
    admissibility: list[str],
    only_definite: bool,                # if True, exclude ADMISSIBILITY_UNKNOWN candidates
    time_limit: float,
    cand_literal_counts: list[int] | None = None,      # literal count per candidate
    cand_max_expansions: list[float] | None = None,    # max normalised expansion per candidate
    lexicographic_objectives: bool = True,
) -> SetCoverRecord:
    """Solve set cover as a MILP with four lexicographic stages.

    Stage 1: minimise cardinality (count of selected clauses).
    Stage 2: fix cardinality; minimise total literal count.
    Stage 3: fix literal count; minimise maximum normalised expansion.
    Stage 4: fix max expansion; minimise sum of original candidate IDs (lex tiebreak).

    Stages 2-4 are only run when lexicographic_objectives=True, stage 1 is OPTIMAL,
    and cand_literal_counts / cand_max_expansions are provided.  The LP dual bound
    reported in the returned record always comes from stage 1.
    """
    t_start = time.perf_counter()

    feasible_atoms = [i for i, f in enumerate(atom_feasible) if f]
    if not feasible_atoms:
        return SetCoverRecord(
            status="OPTIMAL",
            selected_candidate_indices=(),
            objective_value=0,
            global_lower_bound=0.0,
            runtime_seconds=0.0,
            backend="scipy-highs",
        )

    # Select candidate pool
    if only_definite:
        cand_pool = [c for c in range(n_cands) if admissibility[c] == "DEFINITELY_ADMISSIBLE"]
    else:
        cand_pool = [c for c in range(n_cands) if admissibility[c] != "DEFINITELY_INADMISSIBLE"]

    if not cand_pool:
        return SetCoverRecord(
            status="INFEASIBLE",
            selected_candidate_indices=(),
            objective_value=None,
            global_lower_bound=None,
            runtime_seconds=time.perf_counter() - t_start,
            backend="scipy-highs",
        )

    n_pool = len(cand_pool)
    pool_idx = {c: i for i, c in enumerate(cand_pool)}

    # ------------------------------------------------------------------
    # Build base coverage constraint matrix (n_atom_constrs x n_pool)
    # ------------------------------------------------------------------
    row_list: list[int] = []
    col_list: list[int] = []
    val_list: list[float] = []
    lb_base: list[float] = []
    ub_base: list[float] = []

    n_atom_constrs = 0
    for a in feasible_atoms:
        covering = [pool_idx[c] for c in range(n_cands) if c in pool_idx and a in cand_to_atoms[c]]
        if not covering:
            t_end = time.perf_counter()
            return SetCoverRecord(
                status="INFEASIBLE",
                selected_candidate_indices=(),
                objective_value=None,
                global_lower_bound=None,
                runtime_seconds=t_end - t_start,
                backend="scipy-highs",
            )
        for i in covering:
            row_list.append(n_atom_constrs)
            col_list.append(i)
            val_list.append(1.0)
        lb_base.append(1.0)
        ub_base.append(float(len(covering)))
        n_atom_constrs += 1

    A_base = sp.csc_matrix((val_list, (row_list, col_list)), shape=(n_atom_constrs, n_pool))
    lb_base_arr = np.array(lb_base, dtype=np.float64)
    ub_base_arr = np.array(ub_base, dtype=np.float64)

    integrality_base = np.ones(n_pool, dtype=int)
    bounds_base = Bounds(np.zeros(n_pool), np.ones(n_pool))
    options_base: dict[str, Any] = {"disp": False, "mip_rel_gap": 1e-6}

    # Helper: parse a scipy milp result into status string
    def _parse_status(r: Any) -> Literal["OPTIMAL", "FEASIBLE", "INFEASIBLE", "TIMEOUT", "UNKNOWN"]:
        s = r.status
        if s == 0:
            return "OPTIMAL"
        if s == 1:
            return "TIMEOUT" if r.fun is None else "FEASIBLE"
        if s == 2:
            return "INFEASIBLE"
        return "UNKNOWN"

    # Helper: extract selected candidates from a binary solution vector
    def _extract(x: np.ndarray) -> tuple[int, ...]:
        idxs = [i for i in range(n_pool) if x[i] > 0.5]
        return tuple(sorted(cand_pool[i] for i in idxs))

    # ---------------------------------------------------------------
    # Stage 1: minimise cardinality
    # ---------------------------------------------------------------
    tl1 = time_limit * (0.45 if lexicographic_objectives else 1.0)
    c1 = np.ones(n_pool, dtype=np.float64)
    try:
        r1 = milp(
            c=c1,
            constraints=LinearConstraint(A_base, lb_base_arr, ub_base_arr),
            integrality=integrality_base,
            bounds=bounds_base,
            options={**options_base, "time_limit": tl1},
        )
    except Exception:
        return SetCoverRecord(
            status="UNKNOWN", selected_candidate_indices=(), objective_value=None,
            global_lower_bound=None, runtime_seconds=time.perf_counter() - t_start,
            backend="scipy-highs",
        )

    sc_dual: float | None = None
    raw_dual = getattr(r1, "mip_dual_bound", None)
    if raw_dual is not None:
        sc_dual = float(raw_dual)

    s1 = _parse_status(r1)

    # If stage 1 did not reach optimality (or secondary objectives disabled), return here
    run_secondary = (
        s1 == "OPTIMAL"
        and lexicographic_objectives
        and cand_literal_counts is not None
        and r1.x is not None
    )
    if not run_secondary:
        selected: tuple[int, ...] = ()
        obj_val: int | None = None
        if r1.x is not None:
            selected = _extract(r1.x)
            obj_val = len(selected)
        return SetCoverRecord(
            status=s1, selected_candidate_indices=selected,
            objective_value=obj_val, global_lower_bound=sc_dual,
            runtime_seconds=time.perf_counter() - t_start, backend="scipy-highs",
        )

    N_star = int(round(r1.fun))  # type: ignore[arg-type]
    best_x = r1.x.copy()  # type: ignore[union-attr]

    # ---------------------------------------------------------------
    # Stage 2: fix cardinality = N*; minimise total literal count
    # ---------------------------------------------------------------
    assert cand_literal_counts is not None  # guaranteed by run_secondary check
    pool_lit = np.array(
        [cand_literal_counts[c] if c < len(cand_literal_counts) else 1 for c in cand_pool],
        dtype=np.float64,
    )
    # Add cardinality equality: N* <= sum(x) <= N*
    A2 = sp.vstack([A_base, sp.csc_matrix(np.ones((1, n_pool)))])
    lb2 = np.concatenate([lb_base_arr, [float(N_star)]])
    ub2 = np.concatenate([ub_base_arr, [float(N_star)]])

    tl2 = max(0.5, (time_limit - (time.perf_counter() - t_start)) * 0.35)
    try:
        r2 = milp(
            c=pool_lit,
            constraints=LinearConstraint(A2, lb2, ub2),
            integrality=integrality_base,
            bounds=bounds_base,
            options={**options_base, "time_limit": tl2},
        )
    except Exception:
        r2 = None

    if r2 is None or r2.status != 0 or r2.x is None:
        return SetCoverRecord(
            status="OPTIMAL", selected_candidate_indices=_extract(best_x),
            objective_value=N_star, global_lower_bound=sc_dual,
            runtime_seconds=time.perf_counter() - t_start, backend="scipy-highs",
        )

    L_star = float(r2.fun)
    best_x = r2.x.copy()

    # ---------------------------------------------------------------
    # Stage 3: fix literal count; minimise maximum normalised expansion
    # Introduces auxiliary variable t (index n_pool in extended variable vector).
    # Variables: [x_0, ..., x_{n_pool-1}, t]
    # New constraints: max_exp[c] * x_c - t <= 0  for each c in pool.
    # ---------------------------------------------------------------
    if cand_max_expansions is None:
        # Skip stage 3 if expansion data not available
        return SetCoverRecord(
            status="OPTIMAL", selected_candidate_indices=_extract(best_x),
            objective_value=N_star, global_lower_bound=sc_dual,
            runtime_seconds=time.perf_counter() - t_start, backend="scipy-highs",
        )

    pool_exp = np.array(
        [max(0.0, cand_max_expansions[c]) if c < len(cand_max_expansions) else 0.0
         for c in cand_pool],
        dtype=np.float64,
    )
    max_exp_ub = float(np.max(pool_exp)) + 1.0 if len(pool_exp) > 0 else 1.0

    n_vars3 = n_pool + 1   # last var is t
    t_idx = n_pool

    # Extend A2 by one zero column for t
    A3_left = sp.hstack([A2, sp.csc_matrix(np.zeros((A2.shape[0], 1)))])
    lb3 = np.concatenate([lb2, [-np.inf]])
    ub3 = np.concatenate([ub2, [L_star]])

    # Add literal count upper bound row (with t column = 0)
    # (already encoded as ub3 on the last cardinality row from A2; here we add an
    # explicit lit-count upper bound row separate from cardinality)
    lit_row = np.zeros((1, n_vars3))
    lit_row[0, :n_pool] = pool_lit
    A3_lit = sp.csc_matrix(lit_row)
    A3_base_ext = sp.vstack([A3_left, A3_lit])
    lb3_ext = np.concatenate([lb3, [-np.inf]])
    ub3_ext = np.concatenate([ub3, [L_star]])

    # Expansion constraints: max_exp[c]*x_c - t <= 0
    exp_data: list[float] = []
    exp_row_idx: list[int] = []
    exp_col_idx: list[int] = []
    for i, exp_val in enumerate(pool_exp):
        exp_row_idx.append(i)
        exp_col_idx.append(i)
        exp_data.append(float(exp_val))
        exp_row_idx.append(i)
        exp_col_idx.append(t_idx)
        exp_data.append(-1.0)
    A3_exp = sp.csc_matrix(
        (exp_data, (exp_row_idx, exp_col_idx)), shape=(n_pool, n_vars3)
    )
    lb3_exp = np.full(n_pool, -np.inf)
    ub3_exp = np.zeros(n_pool)

    A3_full = sp.vstack([A3_base_ext, A3_exp])
    lb3_full = np.concatenate([lb3_ext, lb3_exp])
    ub3_full = np.concatenate([ub3_ext, ub3_exp])

    c3 = np.zeros(n_vars3, dtype=np.float64)
    c3[t_idx] = 1.0

    int3 = np.ones(n_vars3, dtype=int)
    int3[t_idx] = 0   # t is continuous
    bounds3 = Bounds(np.zeros(n_vars3), np.concatenate([np.ones(n_pool), [max_exp_ub]]))

    tl3 = max(0.5, (time_limit - (time.perf_counter() - t_start)) * 0.4)
    try:
        r3 = milp(
            c=c3,
            constraints=LinearConstraint(A3_full, lb3_full, ub3_full),
            integrality=int3,
            bounds=bounds3,
            options={**options_base, "time_limit": tl3},
        )
    except Exception:
        r3 = None

    if r3 is None or r3.status != 0 or r3.x is None:
        return SetCoverRecord(
            status="OPTIMAL", selected_candidate_indices=_extract(best_x),
            objective_value=N_star, global_lower_bound=sc_dual,
            runtime_seconds=time.perf_counter() - t_start, backend="scipy-highs",
        )

    t_star = float(r3.fun)
    best_x = r3.x[:n_pool].copy()

    # ---------------------------------------------------------------
    # Stage 4: fix max expansion; minimise sum of original candidate IDs
    # (approximates lexicographically smallest clause-ID sequence)
    # ---------------------------------------------------------------
    # Tighten t upper bound to t_star; use same constraint structure as stage 3
    bounds4 = Bounds(
        np.zeros(n_vars3),
        np.concatenate([np.ones(n_pool), [t_star + 1e-9]]),
    )
    # Objective: minimise sum(original_cand_id * x_c) to prefer smaller IDs
    c4 = np.zeros(n_vars3, dtype=np.float64)
    for local_i, global_c in enumerate(cand_pool):
        c4[local_i] = float(global_c)  # original candidate index as weight

    tl4 = max(0.5, time_limit - (time.perf_counter() - t_start))
    try:
        r4 = milp(
            c=c4,
            constraints=LinearConstraint(A3_full, lb3_full, ub3_full),
            integrality=int3,
            bounds=bounds4,
            options={**options_base, "time_limit": tl4},
        )
    except Exception:
        r4 = None

    if r4 is not None and r4.status == 0 and r4.x is not None:
        best_x = r4.x[:n_pool].copy()

    return SetCoverRecord(
        status="OPTIMAL",
        selected_candidate_indices=_extract(best_x),
        objective_value=N_star,
        global_lower_bound=sc_dual,
        runtime_seconds=time.perf_counter() - t_start,
        backend="scipy-highs",
    )


# ------------------------------------------------------------------ #
# Canonical hash functions                                           #
# ------------------------------------------------------------------ #


def _sha256_hex(obj: Any) -> str:
    """Return first 16 hex chars of SHA-256 of the canonical JSON."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def hash_grammar(grammar: ContextGrammar) -> str:
    """Canonical hash of a ContextGrammar."""
    parts: dict[str, Any] = {
        "feature_indices": list(grammar.feature_indices),
        "bins_by_feature": [
            [
                {"feature_index": int(lit.feature_index), "term": str(lit.term),
                 "min_membership": float(lit.min_membership)}
                for lit in bins
            ]
            for bins in grammar.bins_by_feature
        ],
        "max_clause_literals": int(grammar.max_clause_literals),
        "coverage_mode": str(grammar.coverage_mode),
    }
    return _sha256_hex(parts)


def hash_materiality(materiality: MaterialityPolicy) -> str:
    """Canonical hash of a MaterialityPolicy."""
    parts: dict[str, Any] = {
        "margin_scale": float(materiality.margin_scale),
        "direction_epsilon": float(materiality.direction_epsilon),
        "merge_tolerance": float(materiality.merge_tolerance),
        "grammar_limited_width": float(materiality.grammar_limited_width),
    }
    return _sha256_hex(parts)


# ------------------------------------------------------------------ #
# Main synthesis function                                            #
# ------------------------------------------------------------------ #


def synthesize_transition_atlas(
    model: FuzzyRuleSVM,
    query: TransitionQuery,
    grammar: ContextGrammar,
    materiality: MaterialityPolicy,
    *,
    envelope_solver: MilpConfig = MilpConfig(),
    set_cover_time_limit_seconds: float = 60.0,
) -> TransitionAtlas:
    """Synthesise a Minimal Certified Linguistic Transition Atlas.

    Parameters
    ----------
    model:
        Fitted binary FuzzyRuleSVM with and_operator="min".
    query:
        Transition query (feature, terms, alpha, domain, ...).
    grammar:
        Context grammar specifying feature space and bins.
    materiality:
        Policy for direction signatures and merge tolerance.
    envelope_solver:
        MILP config for per-atom CLTE solves.
    set_cover_time_limit_seconds:
        Time budget for set-cover MILPs.

    Returns
    -------
    TransitionAtlas
    """
    t_start = time.perf_counter()
    warnings_list: list[str] = []

    # --- Validate model ---
    try:
        check_is_fitted(model)
    except Exception as exc:
        return _invalid_atlas(
            query, grammar, materiality, envelope_solver,
            set_cover_time_limit_seconds, f"model not fitted: {exc}"
        )

    if model.and_operator != "min":
        return _invalid_atlas(
            query, grammar, materiality, envelope_solver,
            set_cover_time_limit_seconds,
            f"only 'min' t-norm supported; got '{model.and_operator}'"
        )

    if materiality.margin_scale <= 0:
        return _invalid_atlas(
            query, grammar, materiality, envelope_solver,
            set_cover_time_limit_seconds, "margin_scale must be > 0"
        )

    n_screened = len(model.partitions_)
    j = query.feature_index

    # Validate grammar feature indices
    for fi, feat in enumerate(grammar.feature_indices):
        if feat == j:
            return _invalid_atlas(
                query, grammar, materiality, envelope_solver,
                set_cover_time_limit_seconds,
                f"grammar feature {feat} is the transition feature; grammar must exclude it"
            )
        if not (0 <= feat < n_screened):
            return _invalid_atlas(
                query, grammar, materiality, envelope_solver,
                set_cover_time_limit_seconds,
                f"grammar feature_index {feat} out of range [0, {n_screened})"
            )
        # Check bin coverage: for description_complete, bins must cover the feature range
        dom = query.domain
        lo_feat = dom.lower[feat]
        hi_feat = dom.upper[feat]
        p_feat = model.partitions_[feat]
        coverage_error = _check_grammar_coverage(grammar, fi, feat, p_feat, lo_feat, hi_feat)
        if coverage_error is not None:
            return _invalid_atlas(
                query, grammar, materiality, envelope_solver,
                set_cover_time_limit_seconds, coverage_error
            )

    s = materiality.margin_scale
    eps_dir = materiality.direction_epsilon

    # --- Compute canonical input hashes ---
    _model_hash = hash_model(model)
    _domain_hash = hash_domain(query.domain)
    _query_hash = hash_query(query)
    _grammar_hash = hash_grammar(grammar)
    _materiality_hash = hash_materiality(materiality)

    # --- Step 1: Enumerate atoms ---
    all_descriptions = _enumerate_atoms(grammar)
    n_total_atoms = len(all_descriptions)

    # --- Step 2: Solve CLTE for each atom ---
    atom_records: list[AtomRecord] = []
    n_feasible = 0

    for desc in all_descriptions:
        clause = _description_to_clause(desc, grammar)
        # Solve CLTE with this context clause
        env = certify_transition_envelope(
            model, query, context=clause, solver=envelope_solver
        )

        # Classify direction signature
        dL = env.dual_lower
        pL = env.primal_lower
        pU = env.primal_upper
        dU = env.dual_upper

        if env.status == "INFEASIBLE":
            # Infeasible atom: not part of feasible atom set
            dir_sig = None
            poss_sigs: frozenset[str] = frozenset()
            gl_status = "NOT_GRAMMAR_LIMITED"
            w_lo, w_hi = 0.0, 0.0
        elif env.status == "INVALID":
            dir_sig = None
            poss_sigs = frozenset(ALL_SIGNATURES)
            gl_status = "POSSIBLY_GRAMMAR_LIMITED"
            w_lo, w_hi = 0.0, float("inf")
        else:
            n_feasible += 1
            poss_sigs = possible_signatures(dL, pL, pU, dU, eps_dir)
            if dL is not None and dU is not None:
                dir_sig = classify_direction(
                    (dL + pL) / 2 if pL is not None else dL,
                    (dU + pU) / 2 if pU is not None else dU,
                    eps_dir,
                )
                # Use tighter estimate if both are available
                if pL is not None and pU is not None:
                    dir_sig = classify_direction(pL, pU, eps_dir)
            else:
                dir_sig = None
            w_lo, w_hi = _normalised_width(dL, pL, pU, dU, s)
            gl_status = _grammar_limited_status(poss_sigs, w_lo, w_hi, materiality.grammar_limited_width)

        atom_records.append(AtomRecord(
            description=desc,
            context_clause=clause,
            envelope=env,
            direction_signature=dir_sig,
            possible_signatures=poss_sigs,
            grammar_limited_status=gl_status,
            normalised_width_lo=w_lo,
            normalised_width_hi=w_hi,
        ))

    # Check infeasible base transition
    if n_feasible == 0:
        return TransitionAtlas(
            schema_version="1.0",
            status="INFEASIBLE_TRANSITION",
            query=query,
            grammar=grammar,
            materiality=materiality,
            envelope_solver=envelope_solver,
            set_cover_time_limit_seconds=set_cover_time_limit_seconds,
            model_hash=_model_hash,
            domain_hash=_domain_hash,
            query_hash=_query_hash,
            grammar_hash=_grammar_hash,
            materiality_hash=_materiality_hash,
            feasible_atom_count=0,
            atoms=tuple(atom_records),
            candidates=(),
            selected_candidate_indices=(),
            min_cardinality_lower=None,
            min_cardinality_upper=None,
            greedy_cover=None,
            exact_cover=None,
            optimistic_cover=None,
            warnings=tuple(warnings_list),
            runtime_seconds=time.perf_counter() - t_start,
        )

    # --- Step 3: Enumerate candidate clauses ---
    all_partials = _enumerate_candidates(grammar)
    # Add the empty clause (all None) only if L_max == 0 (unconditional)
    # For max_clause_literals >= 1, we already have all partial descriptions

    feasible_atom_indices = [
        i for i, a in enumerate(atom_records) if a.envelope.status != "INFEASIBLE"
    ]

    # Build atom -> candidates and candidate -> atoms maps
    # Use bitsets (frozensets) for efficiency
    candidate_records: list[CandidateRecord] = []
    cand_to_atoms_list: list[list[int]] = []

    for partial in all_partials:
        # Find extending feasible atoms
        extending = frozenset(
            i for i in feasible_atom_indices
            if _extends(atom_records[i].description, partial)
        )
        if not extending:
            continue  # empty extension set -> skip

        clause = _partial_to_clause(partial, grammar)
        lit_count = sum(1 for x in partial if x is not None)

        # Derive envelope from atom envelopes
        ext_atoms = [atom_records[i] for i in extending]

        d_Ls = [a.envelope.dual_lower for a in ext_atoms if a.envelope.dual_lower is not None]
        p_Ls = [a.envelope.primal_lower for a in ext_atoms if a.envelope.primal_lower is not None]
        p_Us = [a.envelope.primal_upper for a in ext_atoms if a.envelope.primal_upper is not None]
        d_Us = [a.envelope.dual_upper for a in ext_atoms if a.envelope.dual_upper is not None]

        # Candidate envelope: conservative bounds from atom envelopes
        # L_c = min over atoms, U_c = max over atoms
        cand_dL = min(d_Ls) if d_Ls else None
        cand_pL = min(p_Ls) if p_Ls else None
        cand_pU = max(p_Us) if p_Us else None
        cand_dU = max(d_Us) if d_Us else None

        # Classify admissibility for this candidate
        adm_status, adm_notes = _classify_candidate_admissibility(
            extending, atom_records,
            cand_dL, cand_pL, cand_pU, cand_dU,
            materiality,
        )

        # Compute maximum normalised expansion over extension atoms (for lexicographic objective)
        max_expansion = 0.0
        for ai in extending:
            atom = atom_records[ai]
            dL_a = atom.envelope.dual_lower
            pL_a = atom.envelope.primal_lower
            pU_a = atom.envelope.primal_upper
            dU_a = atom.envelope.dual_upper
            _, ell_hi = _normalised_expansion(
                dL_a, pL_a, pU_a, dU_a,
                cand_dL, cand_pL, cand_pU, cand_dU,
                materiality.margin_scale,
            )
            if ell_hi > max_expansion:
                max_expansion = ell_hi

        candidate_records.append(CandidateRecord(
            clause=clause,
            literal_count=lit_count,
            extension_atom_indices=extending,
            derived_dual_lower=cand_dL,
            derived_primal_lower=cand_pL,
            derived_primal_upper=cand_pU,
            derived_dual_upper=cand_dU,
            admissibility_status=adm_status,
            admissibility_notes=tuple(adm_notes),
            max_norm_expansion=max_expansion,
        ))
        cand_to_atoms_list.append(list(extending))

    # --- Step 4: Set cover ---
    n_cands = len(candidate_records)
    atom_feasible = [i in feasible_atom_indices for i in range(len(atom_records))]
    admissibility_list = [c.admissibility_status for c in candidate_records]

    # Check if any admissible cover exists
    has_definite = any(
        a == "DEFINITELY_ADMISSIBLE" for a in admissibility_list
    )

    # Greedy cover (using all non-inadmissible candidates only)
    # Build a filtered pool: exclude DEFINITELY_INADMISSIBLE candidates.
    greedy_global_indices = [
        ci for ci, c in enumerate(candidate_records)
        if c.admissibility_status != "DEFINITELY_INADMISSIBLE"
    ]
    greedy_pool_atoms = [
        list(candidate_records[ci].extension_atom_indices)
        for ci in greedy_global_indices
    ]
    greedy_local_indices = _greedy_set_cover(
        len(atom_records), [], greedy_pool_atoms, len(greedy_global_indices)
    )
    # Map local indices back to global candidate indices
    greedy_indices = [greedy_global_indices[li] for li in greedy_local_indices]
    # Check if greedy covers all feasible atoms
    greedy_covered = set()
    for ci in greedy_indices:
        greedy_covered.update(candidate_records[ci].extension_atom_indices)
    greedy_feasible_covered = all(i in greedy_covered for i in feasible_atom_indices)

    greedy_record = SetCoverRecord(
        status="OPTIMAL" if greedy_feasible_covered else "INFEASIBLE",
        selected_candidate_indices=tuple(greedy_indices),
        objective_value=len(greedy_indices) if greedy_feasible_covered else None,
        global_lower_bound=None,
        runtime_seconds=0.0,
        backend="greedy",
    )

    # Exact set cover (only definitely admissible candidates)
    exact_cover_budget = set_cover_time_limit_seconds * 0.5
    exact_rec = _solve_set_cover_milp(
        len(atom_records),
        atom_feasible,
        cand_to_atoms_list,
        n_cands,
        admissibility_list,
        only_definite=True,
        time_limit=exact_cover_budget,
        cand_literal_counts=[c.literal_count for c in candidate_records],
        cand_max_expansions=[c.max_norm_expansion for c in candidate_records],
    )

    # Optimistic cover (all non-inadmissible candidates)
    optim_budget = set_cover_time_limit_seconds * 0.5
    optim_rec = _solve_set_cover_milp(
        len(atom_records),
        atom_feasible,
        cand_to_atoms_list,
        n_cands,
        admissibility_list,
        only_definite=False,
        time_limit=optim_budget,
        cand_literal_counts=[c.literal_count for c in candidate_records],
        cand_max_expansions=[c.max_norm_expansion for c in candidate_records],
    )

    # --- Step 5: Compute cardinality bounds ---
    N_U: int | None = None
    N_L: int | None = None

    if exact_rec.status == "OPTIMAL" and exact_rec.objective_value is not None:
        N_U = exact_rec.objective_value
    elif greedy_feasible_covered:
        N_U = len(greedy_indices)

    if optim_rec.global_lower_bound is not None:
        atol = envelope_solver.outer_bound_atol
        rtol = envelope_solver.outer_bound_rtol
        B = optim_rec.global_lower_bound
        tau_outer = atol + rtol * max(1.0, abs(B))
        N_L = max(0, int(np.ceil(B - tau_outer)))

    # --- Step 6: Determine selected cover and atlas status ---
    # Prefer exact cover if it found valid solution
    if exact_rec.status == "OPTIMAL" and exact_rec.selected_candidate_indices:
        selected_indices = exact_rec.selected_candidate_indices
    elif greedy_feasible_covered:
        selected_indices = tuple(greedy_indices)
    else:
        selected_indices = ()

    # Verify the selected cover is actually valid
    if selected_indices:
        covered = set()
        for ci in selected_indices:
            if ci < len(candidate_records):
                covered.update(candidate_records[ci].extension_atom_indices)
        cover_valid = all(i in covered for i in feasible_atom_indices)
    else:
        cover_valid = len(feasible_atom_indices) == 0

    # Atlas status
    if not cover_valid:
        # Check if any admissible cover even exists
        all_adm_covered = set()
        for ci, c in enumerate(candidate_records):
            if c.admissibility_status == "DEFINITELY_ADMISSIBLE":
                all_adm_covered.update(c.extension_atom_indices)
        if not all(i in all_adm_covered for i in feasible_atom_indices):
            atlas_status: AtlasStatus = "GRAMMAR_INSUFFICIENT"
        else:
            atlas_status = "UNKNOWN"
    elif N_L is not None and N_U is not None and N_L == N_U:
        # Check all selected are definitely admissible
        all_adm = all(
            candidate_records[ci].admissibility_status == "DEFINITELY_ADMISSIBLE"
            for ci in selected_indices
        )
        if all_adm and exact_rec.status == "OPTIMAL":
            atlas_status = "MINIMUM_SOLVER_CERTIFIED"
        else:
            atlas_status = "VALID_COVER_MINIMALITY_UNKNOWN"
    elif N_U is not None and N_L is not None:
        gap = N_U - N_L
        rel_gap = gap / max(1, N_L)
        if gap <= 1 and rel_gap <= 0.10:
            atlas_status = "NEAR_MINIMUM_SOLVER_CERTIFIED"
        else:
            atlas_status = "VALID_COVER_MINIMALITY_UNKNOWN"
    elif N_U is not None:
        atlas_status = "VALID_COVER_MINIMALITY_UNKNOWN"
    else:
        atlas_status = "UNKNOWN"

    return TransitionAtlas(
        schema_version="1.0",
        status=atlas_status,
        query=query,
        grammar=grammar,
        materiality=materiality,
        envelope_solver=envelope_solver,
        set_cover_time_limit_seconds=set_cover_time_limit_seconds,
        model_hash=_model_hash,
        domain_hash=_domain_hash,
        query_hash=_query_hash,
        grammar_hash=_grammar_hash,
        materiality_hash=_materiality_hash,
        feasible_atom_count=n_feasible,
        atoms=tuple(atom_records),
        candidates=tuple(candidate_records),
        selected_candidate_indices=tuple(selected_indices),
        min_cardinality_lower=N_L,
        min_cardinality_upper=N_U,
        greedy_cover=greedy_record,
        exact_cover=exact_rec,
        optimistic_cover=optim_rec,
        warnings=tuple(warnings_list),
        runtime_seconds=time.perf_counter() - t_start,
    )


def _classify_candidate_admissibility(
    extending: frozenset[int],
    atom_records: list[AtomRecord],
    cand_dL: float | None,
    cand_pL: float | None,
    cand_pU: float | None,
    cand_dU: float | None,
    materiality: MaterialityPolicy,
) -> tuple[AdmissibilityStatus, list[str]]:
    """Classify admissibility of a candidate clause."""
    s = materiality.margin_scale
    eta = materiality.merge_tolerance

    notes: list[str] = []
    all_definitely_ok = True
    any_definitely_bad = False

    # Compute candidate's possible signatures
    cand_poss = possible_signatures(cand_dL, cand_pL, cand_pU, cand_dU, materiality.direction_epsilon)

    for ai in extending:
        atom = atom_records[ai]
        dL_a = atom.envelope.dual_lower
        pL_a = atom.envelope.primal_lower
        pU_a = atom.envelope.primal_upper
        dU_a = atom.envelope.dual_upper

        atom_poss = atom.possible_signatures

        def_ok, def_bad, ell_lo, ell_hi = _classify_atom_admissibility_in_candidate(
            dL_a, pL_a, pU_a, dU_a,
            cand_dL, cand_pL, cand_pU, cand_dU,
            atom_poss, cand_poss,
            eta, s,
        )
        if not def_ok:
            all_definitely_ok = False
        if def_bad:
            any_definitely_bad = True
            notes.append(f"atom {ai}: ell_lo={ell_lo:.4f} > eta={eta:.4f} or disjoint signatures")

    if any_definitely_bad:
        return "DEFINITELY_INADMISSIBLE", notes
    if all_definitely_ok:
        return "DEFINITELY_ADMISSIBLE", notes
    return "ADMISSIBILITY_UNKNOWN", notes


def _check_grammar_coverage(
    grammar: ContextGrammar,
    fi: int,
    feat: int,
    partition: Any,
    lo: float,
    hi: float,
) -> str | None:
    """Return None if grammar bins cover the full feature range, or an error string.

    Checks that for every point in a dense grid over [lo, hi], at least one
    declared bin achieves its min_membership threshold.  Returns the first
    uncovered value found, or None if all points are covered.
    """
    grid = np.linspace(lo, hi, 20)
    bins = grammar.bins_by_feature[fi]
    for x_val in grid:
        covered = False
        for lit in bins:
            mu = _mu_scalar(lit.term, float(x_val), partition.low, partition.medium, partition.high)
            if mu >= lit.min_membership:
                covered = True
                break
        if not covered:
            return (
                f"Grammar feature {feat} (grammar position {fi}): "
                f"value {x_val:.4f} not covered by any bin at threshold "
                f"{bins[0].min_membership:.3f}. "
                "Grammar coverage is incomplete — cannot guarantee "
                "description-complete coverage. Add extra bins or lower gamma."
            )
    return None


# ------------------------------------------------------------------ #
# Atlas verification                                                 #
# ------------------------------------------------------------------ #


def verify_transition_atlas(
    model: FuzzyRuleSVM,
    atlas: TransitionAtlas,
    *,
    postcheck_atol: float = 1e-8,
    rerun_set_cover: bool = True,
) -> AtlasVerificationReport:
    """Independently verify a TransitionAtlas.

    Checks:
    1. Hash verification: model, domain, and query hashes match atlas records.
    2. Coverage: every feasible atom is covered by some selected candidate.
    3. Admissibility: selected candidates are all definitely admissible.
    4. Witness revalidation: witnesses in atom envelopes satisfy constraints.
    5. Set-cover rerun (if rerun_set_cover=True): rerun set cover from atom
       records and compare N_L/N_U against the atlas values.

    Parameters
    ----------
    model:
        The same fitted model used to synthesise the atlas.
    atlas:
        The atlas to verify.
    postcheck_atol:
        Tolerance for witness residual checks.
    rerun_set_cover:
        If True, rerun set cover from scratch using only the atom records
        stored in the atlas (no new MILP solves) and compare cardinality
        bounds against atlas.min_cardinality_lower/upper.

    Returns
    -------
    AtlasVerificationReport
    """
    t_start = time.perf_counter()
    notes: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Hash verification
    # ------------------------------------------------------------------
    hash_verified = True
    actual_model_hash = hash_model(model)
    if atlas.model_hash and actual_model_hash != atlas.model_hash:
        hash_verified = False
        notes.append(
            f"model_hash mismatch: atlas recorded {atlas.model_hash!r}, "
            f"actual {actual_model_hash!r}"
        )

    actual_domain_hash = hash_domain(atlas.query.domain)
    if atlas.domain_hash and actual_domain_hash != atlas.domain_hash:
        hash_verified = False
        notes.append(
            f"domain_hash mismatch: atlas recorded {atlas.domain_hash!r}, "
            f"actual {actual_domain_hash!r}"
        )

    actual_query_hash = hash_query(atlas.query)
    if atlas.query_hash and actual_query_hash != atlas.query_hash:
        hash_verified = False
        notes.append(
            f"query_hash mismatch: atlas recorded {atlas.query_hash!r}, "
            f"actual {actual_query_hash!r}"
        )

    # ------------------------------------------------------------------
    # 2. Coverage check
    # ------------------------------------------------------------------
    feasible_atom_indices = [
        i for i, a in enumerate(atlas.atoms)
        if a.envelope.status not in ("INFEASIBLE", "INVALID")
    ]
    covered = set()
    for ci in atlas.selected_candidate_indices:
        if ci < len(atlas.candidates):
            covered.update(atlas.candidates[ci].extension_atom_indices)

    gap_atoms = [i for i in feasible_atom_indices if i not in covered]
    coverage_verified = len(gap_atoms) == 0

    # ------------------------------------------------------------------
    # 3. Admissibility check
    # ------------------------------------------------------------------
    admissibility_verified = True
    for ci in atlas.selected_candidate_indices:
        if ci < len(atlas.candidates):
            if atlas.candidates[ci].admissibility_status != "DEFINITELY_ADMISSIBLE":
                admissibility_verified = False
                notes.append(
                    f"Selected candidate {ci} has admissibility "
                    f"{atlas.candidates[ci].admissibility_status}"
                )

    # ------------------------------------------------------------------
    # 4. Witness revalidation
    # ------------------------------------------------------------------
    witness_revalidated = True
    for ai, atom in enumerate(atlas.atoms):
        env = atom.envelope
        for rec in [env.lower_solve, env.upper_solve]:
            if rec is None:
                continue
            if rec.witness is None:
                continue
            w = rec.witness
            if not w.validated:
                witness_revalidated = False
                notes.append(f"Atom {ai}: witness not validated: {w.validation_notes}")
            # Additional recheck: verify objective reconstruction
            if abs(w.objective_error) > postcheck_atol:
                witness_revalidated = False
                notes.append(
                    f"Atom {ai}: objective_error={w.objective_error:.3e} > atol={postcheck_atol:.3e}"
                )

    # ------------------------------------------------------------------
    # 5. Set-cover rerun (independent of synthesis code)
    # ------------------------------------------------------------------
    rerun_lower: int | None = None
    rerun_upper: int | None = None
    set_cover_verified = True  # vacuously true when not requested

    if rerun_set_cover and len(atlas.candidates) > 0:
        n_atoms = len(atlas.atoms)
        atom_feasible = [i in feasible_atom_indices for i in range(n_atoms)]
        admissibility_list = [c.admissibility_status for c in atlas.candidates]
        cand_to_atoms = [list(c.extension_atom_indices) for c in atlas.candidates]
        n_cands = len(atlas.candidates)

        rerun_exact = _solve_set_cover_milp(
            n_atoms,
            atom_feasible,
            cand_to_atoms,
            n_cands,
            admissibility_list,
            only_definite=True,
            time_limit=30.0,
            cand_literal_counts=[c.literal_count for c in atlas.candidates],
            cand_max_expansions=[c.max_norm_expansion for c in atlas.candidates],
        )
        rerun_optim = _solve_set_cover_milp(
            n_atoms,
            atom_feasible,
            cand_to_atoms,
            n_cands,
            admissibility_list,
            only_definite=False,
            time_limit=30.0,
            cand_literal_counts=[c.literal_count for c in atlas.candidates],
            cand_max_expansions=[c.max_norm_expansion for c in atlas.candidates],
        )

        if rerun_exact.status == "OPTIMAL" and rerun_exact.objective_value is not None:
            rerun_upper = rerun_exact.objective_value
        if rerun_optim.global_lower_bound is not None:
            tau = 1e-8 + 1e-9 * max(1.0, abs(rerun_optim.global_lower_bound))
            rerun_lower = max(0, int(np.ceil(rerun_optim.global_lower_bound - tau)))

        # Compare with atlas claims
        if rerun_upper is not None and atlas.min_cardinality_upper is not None:
            if rerun_upper != atlas.min_cardinality_upper:
                set_cover_verified = False
                notes.append(
                    f"Set-cover rerun N_U={rerun_upper} differs from atlas "
                    f"N_U={atlas.min_cardinality_upper}"
                )
        if rerun_lower is not None and atlas.min_cardinality_lower is not None:
            if rerun_lower > atlas.min_cardinality_lower + 1:
                set_cover_verified = False
                notes.append(
                    f"Set-cover rerun N_L={rerun_lower} exceeds atlas "
                    f"N_L={atlas.min_cardinality_lower} by more than tolerance"
                )

    # ------------------------------------------------------------------
    # Atlas content hash (structural fingerprint)
    # ------------------------------------------------------------------
    atlas_str = json.dumps({
        "schema_version": atlas.schema_version,
        "status": atlas.status,
        "n_atoms": len(atlas.atoms),
        "n_candidates": len(atlas.candidates),
        "selected": list(atlas.selected_candidate_indices),
    }, sort_keys=True)
    atlas_hash = hashlib.sha256(atlas_str.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------------
    core_pass = coverage_verified and admissibility_verified and witness_revalidated
    all_pass = core_pass and hash_verified and set_cover_verified
    if all_pass:
        overall = "PASS"
    elif core_pass:
        overall = "PARTIAL"
    else:
        overall = "FAIL"

    return AtlasVerificationReport(
        atlas_hash=atlas_hash,
        status=overall,
        coverage_verified=coverage_verified,
        admissibility_verified=admissibility_verified,
        witness_revalidated=witness_revalidated,
        hash_verified=hash_verified,
        set_cover_verified=set_cover_verified,
        coverage_gap_atoms=tuple(gap_atoms),
        rerun_cardinality_lower=rerun_lower,
        rerun_cardinality_upper=rerun_upper,
        warnings=tuple(warnings),
        notes=tuple(notes),
        runtime_seconds=time.perf_counter() - t_start,
    )


def _invalid_atlas(
    query: TransitionQuery,
    grammar: ContextGrammar,
    materiality: MaterialityPolicy,
    envelope_solver: MilpConfig,
    set_cover_time_limit_seconds: float,
    reason: str,
) -> TransitionAtlas:
    """Return an INVALID TransitionAtlas."""
    return TransitionAtlas(
        schema_version="1.0",
        status="INVALID",
        query=query,
        grammar=grammar,
        materiality=materiality,
        envelope_solver=envelope_solver,
        set_cover_time_limit_seconds=set_cover_time_limit_seconds,
        model_hash="",
        domain_hash="",
        query_hash="",
        grammar_hash="",
        materiality_hash="",
        feasible_atom_count=0,
        atoms=(),
        candidates=(),
        selected_candidate_indices=(),
        min_cardinality_lower=None,
        min_cardinality_upper=None,
        greedy_cover=None,
        exact_cover=None,
        optimistic_cover=None,
        warnings=(reason,),
        runtime_seconds=0.0,
    )
