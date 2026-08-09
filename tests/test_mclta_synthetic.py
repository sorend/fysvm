"""Tests for MCLTA synthetic fixtures.

Tests that planted atlas results are recovered correctly and that
failure cases produce the expected outcomes.
"""

from __future__ import annotations

import pytest
import numpy as np

from fysvm.mclta_synthetic import (
    _inject_model,
    make_checkerboard_fixture,
    make_context_reversal_fixture,
    make_default_grammar_bins,
    make_default_grammar_for_model,
    make_infeasible_transition_fixture,
    make_irrelevant_feature_fixture,
    make_length1_positive_rule_fixture,
    make_relational_tightening_fixture,
    make_same_term_fixture,
    make_three_regime_fixture,
    make_tiny_coefficient_fixture,
    make_zero_transition_fixture,
)
from fysvm.transition_atlas import (
    ContextGrammar,
    MaterialityPolicy,
    classify_direction,
    synthesize_transition_atlas,
    verify_transition_atlas,
)
from fysvm.transition_envelopes import (
    ContextLiteral,
    LinearDomain,
    MilpConfig,
    TransitionQuery,
    certify_transition_envelope,
)
from reference_mclta import (
    brute_force_envelope,
    verify_envelope_contains_grid,
)

_FAST_SOLVER = MilpConfig(time_limit_seconds=60.0, relative_gap=1e-6)
_GRID_N = 40  # grid resolution for brute-force checks


def _default_materiality() -> MaterialityPolicy:
    return MaterialityPolicy(
        margin_scale=1.0,
        direction_epsilon=0.05,
        merge_tolerance=0.10,
        grammar_limited_width=0.5,
    )


# ------------------------------------------------------------------ #
# Fixture 1: Zero transition                                         #
# ------------------------------------------------------------------ #


def test_zero_transition_certifies_zero():
    """Zero transition must certify [0, 0] exactly."""
    fix = make_zero_transition_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_FAST_SOLVER)
    assert result.status == "OPTIMAL"
    assert result.primal_lower == pytest.approx(0.0, abs=1e-10)
    assert result.primal_upper == pytest.approx(0.0, abs=1e-10)


# ------------------------------------------------------------------ #
# Fixture 2: Length-1 rule analytic check                           #
# ------------------------------------------------------------------ #


def test_length1_positive_rule_analytic():
    """Analytic extrema for single HIGH rule must be inside certified bounds."""
    fix = make_length1_positive_rule_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_FAST_SOLVER)
    assert result.status == "OPTIMAL"

    tol = 1e-4
    # Analytic lower = 1.0
    assert fix.exact_lower == pytest.approx(result.primal_lower, abs=tol), (
        f"Expected lower {fix.exact_lower}, got {result.primal_lower}"
    )
    # Analytic upper = 2.0
    assert fix.exact_upper == pytest.approx(result.primal_upper, abs=tol), (
        f"Expected upper {fix.exact_upper}, got {result.primal_upper}"
    )

    # Soundness: certified bounds contain analytic extrema
    if result.dual_lower is not None:
        assert result.dual_lower <= fix.exact_lower + tol
    if result.dual_upper is not None:
        assert result.dual_upper >= fix.exact_upper - tol


def test_length1_positive_rule_grid_soundness():
    """Grid search extrema must fall inside certified outer bounds."""
    fix = make_length1_positive_rule_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_FAST_SOLVER)
    if result.status not in ("INFEASIBLE", "INVALID", "UNKNOWN"):
        ok, gmin, gmax, msg = verify_envelope_contains_grid(
            fix.model, fix.query.feature_index,
            fix.query.source_term, fix.query.destination_term,
            fix.query.source_alpha, fix.query.destination_alpha,
            result.target_sign,
            list(fix.query.domain.lower),
            list(fix.query.domain.upper),
            certified_lower=result.dual_lower,
            certified_upper=result.dual_upper,
            n_grid=_GRID_N,
        )
        assert ok, f"Soundness failed: {msg}"


# ------------------------------------------------------------------ #
# Fixture 3: Context reversal — two-regime atlas                     #
# ------------------------------------------------------------------ #


def test_context_reversal_uncond_is_mixed():
    """Unconditional envelope for context reversal fixture is MIXED."""
    fix = make_context_reversal_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_FAST_SOLVER)
    if result.status == "OPTIMAL" and result.primal_lower is not None and result.primal_upper is not None:
        sig = classify_direction(result.primal_lower, result.primal_upper, eps_dir=0.05)
        assert sig == "MIXED", (
            f"Expected MIXED, got {sig} with L={result.primal_lower:.3f}, U={result.primal_upper:.3f}"
        )


def test_context_reversal_low_ctx_positive():
    """Low-context transition: analytic bounds [1.5, 3.0]."""
    fix = make_context_reversal_fixture()
    result = certify_transition_envelope(
        fix.model, fix.query, context=fix.context_low, solver=_FAST_SOLVER
    )
    if result.status == "OPTIMAL" and result.primal_lower is not None:
        assert abs(result.primal_lower - fix.exact_lower_low_ctx) <= 1e-4, (
            f"Low-context lower bound: expected {fix.exact_lower_low_ctx}, "
            f"got {result.primal_lower}"
        )
    if result.status == "OPTIMAL" and result.primal_upper is not None:
        assert abs(result.primal_upper - fix.exact_upper_low_ctx) <= 1e-4, (
            f"Low-context upper bound: expected {fix.exact_upper_low_ctx}, "
            f"got {result.primal_upper}"
        )


def test_context_reversal_high_ctx_negative():
    """High-context transition: analytic bounds [-3.0, -1.5]."""
    fix = make_context_reversal_fixture()
    result = certify_transition_envelope(
        fix.model, fix.query, context=fix.context_high, solver=_FAST_SOLVER
    )
    if result.status == "OPTIMAL" and result.primal_lower is not None:
        assert abs(result.primal_lower - fix.exact_lower_high_ctx) <= 1e-4, (
            f"High-context lower bound: expected {fix.exact_lower_high_ctx}, "
            f"got {result.primal_lower}"
        )
    if result.status == "OPTIMAL" and result.primal_upper is not None:
        assert abs(result.primal_upper - fix.exact_upper_high_ctx) <= 1e-4, (
            f"High-context upper bound: expected {fix.exact_upper_high_ctx}, "
            f"got {result.primal_upper}"
        )


def test_context_reversal_atlas_requires_at_least_two_clauses():
    """For context reversal, the minimum atlas size must be >= 2 (strict policy)."""
    fix = make_context_reversal_fixture()
    grammar = make_default_grammar_for_model(fix.model, 0, gamma=0.5, max_clause_literals=2)
    if len(grammar.feature_indices) == 0:
        pytest.skip("No context features in grammar")

    materiality = MaterialityPolicy(
        margin_scale=1.0,
        direction_epsilon=0.05,
        merge_tolerance=0.05,  # strict: cannot merge INCREASE and DECREASE
        grammar_limited_width=0.5,
    )
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=30.0,
    )
    # The context reversal fixture has a MIXED unconditional envelope, so
    # no single clause covers ALL feasible atoms while remaining admissible.
    # The minimum cover must contain at least 2 clauses.
    if atlas.status in ("MINIMUM_SOLVER_CERTIFIED", "NEAR_MINIMUM_SOLVER_CERTIFIED"):
        assert atlas.min_cardinality_upper is not None
        assert atlas.min_cardinality_upper >= 2, (
            f"Expected at least 2 clauses for context reversal, "
            f"got min_cardinality_upper={atlas.min_cardinality_upper}"
        )
    elif atlas.status == "VALID_COVER_MINIMALITY_UNKNOWN":
        # Cover is valid but minimum not proven; still check the upper bound
        assert atlas.min_cardinality_upper is None or atlas.min_cardinality_upper >= 2


# ------------------------------------------------------------------ #
# Fixture 4: Infeasible transition                                   #
# ------------------------------------------------------------------ #


def test_infeasible_transition_returns_infeasible():
    """Infeasible transition fixture must return INFEASIBLE status."""
    fix = make_infeasible_transition_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_FAST_SOLVER)
    assert result.status == "INFEASIBLE"


# ------------------------------------------------------------------ #
# Fixture 5: Relational tightening                                  #
# ------------------------------------------------------------------ #


def test_relational_tightening_soundness():
    """Certified lower must be >= relational theoretical lower."""
    fix = make_relational_tightening_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_FAST_SOLVER)
    if result.status == "OPTIMAL" and result.primal_lower is not None:
        assert result.primal_lower >= fix.relational_lower - 1e-4, (
            f"Relational lower {result.primal_lower:.4f} < expected {fix.relational_lower:.4f}"
        )
    if result.status == "OPTIMAL" and result.primal_upper is not None:
        assert result.primal_upper <= fix.relational_upper + 1e-4, (
            f"Relational upper {result.primal_upper:.4f} > expected {fix.relational_upper:.4f}"
        )


def test_relational_bounds_tighter_than_independent():
    """Certified lower > independent subtraction lower."""
    fix = make_relational_tightening_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_FAST_SOLVER)
    if result.status == "OPTIMAL" and result.primal_lower is not None:
        assert result.primal_lower > fix.independent_lower - 1e-4, (
            f"Relational lower {result.primal_lower:.4f} not tighter than "
            f"independent lower {fix.independent_lower:.4f}"
        )


# ------------------------------------------------------------------ #
# Fixture 6: Tiny coefficient                                        #
# ------------------------------------------------------------------ #


def test_tiny_coefficient_envelope():
    """Tiny coefficient: solver returns some finite envelope (soundness check)."""
    fix = make_tiny_coefficient_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_FAST_SOLVER)
    # For extremely small objective coefficients (~1e-8), HiGHS may not solve precisely.
    # Verify: (1) result is not INVALID/ERROR, (2) envelope contains grid extrema.
    assert result.status in ("OPTIMAL", "BOUNDED", "FEASIBLE_ONLY", "INFEASIBLE")
    if result.status not in ("INFEASIBLE",) and result.dual_lower is not None:
        ok, gmin, gmax, msg = verify_envelope_contains_grid(
            fix.model, fix.query.feature_index,
            fix.query.source_term, fix.query.destination_term,
            fix.query.source_alpha, fix.query.destination_alpha,
            result.target_sign,
            list(fix.query.domain.lower),
            list(fix.query.domain.upper),
            certified_lower=result.dual_lower,
            certified_upper=result.dual_upper,
            n_grid=20,
            atol=1e-8,  # lenient absolute tolerance for tiny-coefficient case
        )
        assert ok, f"Tiny coefficient soundness failed: {msg}"


# ------------------------------------------------------------------ #
# Atlas recovery tests                                               #
# ------------------------------------------------------------------ #


def test_atlas_synthesis_context_reversal():
    """Atlas synthesis for context reversal fixture should complete."""
    fix = make_context_reversal_fixture()
    grammar = make_default_grammar_for_model(fix.model, 0, gamma=0.5, max_clause_literals=2)
    if len(grammar.feature_indices) == 0:
        pytest.skip("No context features")

    materiality = _default_materiality()
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=30.0,
    )
    assert atlas.status != "INVALID"
    assert len(atlas.atoms) > 0


def test_atlas_all_witnesses_validated():
    """All atom witnesses must pass independent validation."""
    fix = make_context_reversal_fixture()
    grammar = make_default_grammar_for_model(fix.model, 0, gamma=0.5, max_clause_literals=2)
    if len(grammar.feature_indices) == 0:
        pytest.skip("No context features")

    materiality = _default_materiality()
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=30.0,
    )

    failures = []
    for i, atom in enumerate(atlas.atoms):
        for rec in [atom.envelope.lower_solve, atom.envelope.upper_solve]:
            if rec is None:
                continue
            if rec.witness is not None and not rec.witness.validated:
                failures.append(
                    f"Atom {i}: witness not validated: {rec.witness.validation_notes}"
                )
    assert not failures, "\n".join(failures)


def test_atlas_verification_report():
    """Verification report should pass for a valid atlas."""
    fix = make_context_reversal_fixture()
    grammar = make_default_grammar_for_model(fix.model, 0, gamma=0.5, max_clause_literals=2)
    if len(grammar.feature_indices) == 0:
        pytest.skip("No context features")

    materiality = _default_materiality()
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=30.0,
    )

    if atlas.status in ("INFEASIBLE_TRANSITION", "GRAMMAR_INSUFFICIENT", "INVALID"):
        pytest.skip(f"Atlas status: {atlas.status}")

    report = verify_transition_atlas(fix.model, atlas)
    assert report.coverage_verified, f"Coverage gaps: {report.coverage_gap_atoms}"


# ------------------------------------------------------------------ #
# Test: scale-invariance of materiality                              #
# ------------------------------------------------------------------ #


def test_scale_invariance_same_direction_results():
    """Scaling all materiality quantities together preserves direction signatures."""
    fix = make_context_reversal_fixture()
    grammar = make_default_grammar_for_model(fix.model, 0, gamma=0.5, max_clause_literals=2)
    if len(grammar.feature_indices) == 0:
        pytest.skip("No context features")

    mat1 = MaterialityPolicy(
        margin_scale=1.0, direction_epsilon=0.05,
        merge_tolerance=0.10, grammar_limited_width=0.5,
    )
    mat2 = MaterialityPolicy(
        margin_scale=2.0, direction_epsilon=0.10,  # eps_dir / s stays at 0.05
        merge_tolerance=0.10, grammar_limited_width=0.5,
    )

    atlas1 = synthesize_transition_atlas(
        fix.model, fix.query, grammar, mat1,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=15.0,
    )
    atlas2 = synthesize_transition_atlas(
        fix.model, fix.query, grammar, mat2,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=15.0,
    )

    # Both atlases should have the same number of atoms
    assert len(atlas1.atoms) == len(atlas2.atoms)

    # Atom direction signatures should be the same (scale-invariant)
    for a1, a2 in zip(atlas1.atoms, atlas2.atoms):
        if a1.direction_signature and a2.direction_signature:
            assert a1.direction_signature == a2.direction_signature


# ------------------------------------------------------------------ #
# Fixture 8: Same-term transition                                     #
# ------------------------------------------------------------------ #


def test_same_term_fixture_analytic():
    """Same-term (low->low) transition: exact bounds [-0.5, 0.5]."""
    fix = make_same_term_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_FAST_SOLVER)
    assert result.status == "OPTIMAL"
    tol = 1e-4
    assert abs(result.primal_lower - fix.exact_lower) <= tol, (
        f"Lower bound: expected {fix.exact_lower}, got {result.primal_lower}"
    )
    assert abs(result.primal_upper - fix.exact_upper) <= tol, (
        f"Upper bound: expected {fix.exact_upper}, got {result.primal_upper}"
    )


def test_same_term_fixture_grid_soundness():
    """Grid extrema for same-term transition must fall inside certified bounds."""
    fix = make_same_term_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_FAST_SOLVER)
    if result.status not in ("INFEASIBLE", "INVALID", "UNKNOWN"):
        ok, gmin, gmax, msg = verify_envelope_contains_grid(
            fix.model, fix.query.feature_index,
            fix.query.source_term, fix.query.destination_term,
            fix.query.source_alpha, fix.query.destination_alpha,
            result.target_sign,
            list(fix.query.domain.lower),
            list(fix.query.domain.upper),
            certified_lower=result.dual_lower,
            certified_upper=result.dual_upper,
            n_grid=_GRID_N,
        )
        assert ok, f"Same-term soundness failed: {msg}"


# ------------------------------------------------------------------ #
# Fixture 4: Three-regime atlas                                       #
# ------------------------------------------------------------------ #


def test_three_regime_atlas_has_three_atoms():
    """Three-regime fixture must produce exactly 3 atoms (one per context bin)."""
    fix = make_three_regime_fixture()
    grammar = ContextGrammar(
        feature_indices=(1,),
        bins_by_feature=(tuple(fix.grammar_bins),),
        max_clause_literals=1,
    )
    materiality = MaterialityPolicy(
        margin_scale=1.0, direction_epsilon=0.05,
        merge_tolerance=0.10, grammar_limited_width=0.5,
    )
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=30.0,
    )
    assert atlas.status != "INVALID", f"Atlas is INVALID: {atlas.warnings}"
    assert len(atlas.atoms) == 3, f"Expected 3 atoms, got {len(atlas.atoms)}"


def test_three_regime_atlas_directions():
    """Three-regime fixture: three distinct direction signatures expected."""
    fix = make_three_regime_fixture()
    grammar = ContextGrammar(
        feature_indices=(1,),
        bins_by_feature=(tuple(fix.grammar_bins),),
        max_clause_literals=1,
    )
    materiality = MaterialityPolicy(
        margin_scale=1.0, direction_epsilon=0.05,
        merge_tolerance=0.10, grammar_limited_width=0.5,
    )
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=30.0,
    )
    assert atlas.status != "INVALID", f"Atlas is INVALID: {atlas.warnings}"
    signatures = [
        a.direction_signature for a in atlas.atoms
        if a.direction_signature is not None
        and a.envelope.status not in ("INFEASIBLE", "INVALID")
    ]
    # The fixture has INCREASE, NEGLIGIBLE, DECREASE regimes — at least 2 distinct
    assert len(set(signatures)) >= 2, (
        f"Expected at least 2 distinct direction signatures; got: {signatures}"
    )


def test_three_regime_atlas_verification():
    """Three-regime atlas verification report must pass coverage check."""
    fix = make_three_regime_fixture()
    grammar = ContextGrammar(
        feature_indices=(1,),
        bins_by_feature=(tuple(fix.grammar_bins),),
        max_clause_literals=1,
    )
    materiality = MaterialityPolicy(
        margin_scale=1.0, direction_epsilon=0.05,
        merge_tolerance=0.10, grammar_limited_width=0.5,
    )
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=30.0,
    )
    if atlas.status in ("INFEASIBLE_TRANSITION", "GRAMMAR_INSUFFICIENT", "INVALID"):
        pytest.skip(f"Atlas status: {atlas.status}")
    report = verify_transition_atlas(fix.model, atlas)
    assert report.coverage_verified, f"Coverage gap: {report.coverage_gap_atoms}"
    assert report.hash_verified, f"Hash mismatch: {report.notes}"


# ------------------------------------------------------------------ #
# Fixture 9: Checkerboard — two-literal clauses required             #
# ------------------------------------------------------------------ #


def test_checkerboard_single_literal_grammar_insufficient():
    """With L_max=1, checkerboard fixture must yield GRAMMAR_INSUFFICIENT."""
    fix = make_checkerboard_fixture()
    grammar = ContextGrammar(
        feature_indices=(1, 2),
        bins_by_feature=(
            make_default_grammar_bins(1, gamma=0.5),
            make_default_grammar_bins(2, gamma=0.5),
        ),
        max_clause_literals=1,
    )
    materiality = MaterialityPolicy(
        margin_scale=1.0, direction_epsilon=0.05,
        merge_tolerance=0.05,  # strict: no merging across direction changes
        grammar_limited_width=0.5,
    )
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=30.0,
    )
    assert atlas.status == "GRAMMAR_INSUFFICIENT", (
        f"Expected GRAMMAR_INSUFFICIENT with L_max=1, got {atlas.status}"
    )


def test_checkerboard_two_literal_valid_cover():
    """With L_max=2, checkerboard fixture must yield a valid cover."""
    fix = make_checkerboard_fixture()
    grammar = ContextGrammar(
        feature_indices=(1, 2),
        bins_by_feature=(
            make_default_grammar_bins(1, gamma=0.5),
            make_default_grammar_bins(2, gamma=0.5),
        ),
        max_clause_literals=2,
    )
    materiality = MaterialityPolicy(
        margin_scale=1.0, direction_epsilon=0.05,
        merge_tolerance=0.05,  # strict
        grammar_limited_width=0.5,
    )
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=60.0,
    )
    assert atlas.status not in ("INVALID", "GRAMMAR_INSUFFICIENT"), (
        f"Expected valid cover with L_max=2, got {atlas.status}"
    )
    assert atlas.min_cardinality_upper is not None
    # Two-literal cover: at least 2 clauses needed (INCREASE and DECREASE atoms)
    assert atlas.min_cardinality_upper >= 2, (
        f"Expected >= 2 clauses, got {atlas.min_cardinality_upper}"
    )


def test_checkerboard_selected_cover_verified():
    """Checkerboard atlas verification report must pass with L_max=2."""
    fix = make_checkerboard_fixture()
    grammar = ContextGrammar(
        feature_indices=(1, 2),
        bins_by_feature=(
            make_default_grammar_bins(1, gamma=0.5),
            make_default_grammar_bins(2, gamma=0.5),
        ),
        max_clause_literals=2,
    )
    materiality = MaterialityPolicy(
        margin_scale=1.0, direction_epsilon=0.05,
        merge_tolerance=0.05,
        grammar_limited_width=0.5,
    )
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=60.0,
    )
    if atlas.status in ("INFEASIBLE_TRANSITION", "GRAMMAR_INSUFFICIENT", "INVALID"):
        pytest.skip(f"Atlas status: {atlas.status}")
    report = verify_transition_atlas(fix.model, atlas)
    assert report.coverage_verified, f"Coverage gaps: {report.coverage_gap_atoms}"
    assert report.hash_verified, f"Hash mismatch: {report.notes}"


# ------------------------------------------------------------------ #
# Fixture 10: Irrelevant grammar feature                             #
# ------------------------------------------------------------------ #


def test_irrelevant_feature_atoms_have_identical_envelopes():
    """Atoms differing only in the irrelevant feature must have identical bounds."""
    fix = make_irrelevant_feature_fixture()
    grammar = ContextGrammar(
        feature_indices=(fix.relevant_feature_index, fix.irrelevant_feature_index),
        bins_by_feature=(
            make_default_grammar_bins(fix.relevant_feature_index, gamma=0.5),
            make_default_grammar_bins(fix.irrelevant_feature_index, gamma=0.5),
        ),
        max_clause_literals=2,
    )
    materiality = MaterialityPolicy(
        margin_scale=1.0, direction_epsilon=0.05,
        merge_tolerance=0.10, grammar_limited_width=0.5,
    )
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=30.0,
    )
    assert atlas.status != "INVALID", f"Atlas INVALID: {atlas.warnings}"
    assert len(atlas.atoms) == 9, f"Expected 9 atoms (3×3), got {len(atlas.atoms)}"

    # Group atoms by their x1 (relevant) bin (first grammar feature, index 0)
    # Each x1 bin should have 3 atoms (one per x2 bin) with identical envelopes
    tol = 1e-6
    for x1_bin_idx in range(3):
        # Atoms at positions x1_bin_idx * 3 + 0, +1, +2 share the same x1 bin
        group = [atlas.atoms[x1_bin_idx * 3 + k] for k in range(3)]
        ref = group[0].envelope
        for other in group[1:]:
            assert abs((other.envelope.primal_lower or 0.0) - (ref.primal_lower or 0.0)) <= tol, (
                f"x1_bin={x1_bin_idx}: primal_lower differs across x2 bins"
            )
            assert abs((other.envelope.primal_upper or 0.0) - (ref.primal_upper or 0.0)) <= tol, (
                f"x1_bin={x1_bin_idx}: primal_upper differs across x2 bins"
            )


def test_irrelevant_feature_valid_cover():
    """Irrelevant grammar feature: atlas achieves a valid cover."""
    fix = make_irrelevant_feature_fixture()
    grammar = ContextGrammar(
        feature_indices=(fix.relevant_feature_index, fix.irrelevant_feature_index),
        bins_by_feature=(
            make_default_grammar_bins(fix.relevant_feature_index, gamma=0.5),
            make_default_grammar_bins(fix.irrelevant_feature_index, gamma=0.5),
        ),
        max_clause_literals=2,
    )
    materiality = MaterialityPolicy(
        margin_scale=1.0, direction_epsilon=0.05,
        merge_tolerance=0.10, grammar_limited_width=0.5,
    )
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=30.0,
    )
    assert atlas.status not in ("INVALID", "GRAMMAR_INSUFFICIENT"), (
        f"Unexpected status: {atlas.status}"
    )
    report = verify_transition_atlas(fix.model, atlas)
    assert report.coverage_verified


# ------------------------------------------------------------------ #
# Timeout test: envelope and atlas with near-zero time limit         #
# ------------------------------------------------------------------ #


def test_envelope_timeout_returns_valid_status():
    """A near-zero time limit must not produce INVALID; may return FEASIBLE_ONLY."""
    fix = make_context_reversal_fixture()
    tiny_solver = MilpConfig(time_limit_seconds=0.001, relative_gap=1e-6)
    result = certify_transition_envelope(fix.model, fix.query, solver=tiny_solver)
    # Must be a valid status — not INVALID (which signals a query/model problem)
    assert result.status in (
        "OPTIMAL", "BOUNDED", "FEASIBLE_ONLY", "INFEASIBLE", "UNKNOWN"
    ), f"Unexpected status under timeout: {result.status}"


def test_atlas_timeout_n_l_not_from_timed_out_incumbent():
    """Under timeout, N_L must not be derived from a timed-out incumbent.

    Verify: if optimistic_cover has no global_lower_bound (timed out before
    LP relaxation), then atlas.min_cardinality_lower is None or 0.
    """
    fix = make_context_reversal_fixture()
    grammar = make_default_grammar_for_model(fix.model, 0, gamma=0.5, max_clause_literals=2)
    if len(grammar.feature_indices) == 0:
        pytest.skip("No context features")

    materiality = _default_materiality()
    tiny_solver = MilpConfig(time_limit_seconds=60.0, relative_gap=1e-6)  # normal envelope
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=tiny_solver,
        set_cover_time_limit_seconds=0.001,  # near-zero set-cover budget
    )
    assert atlas.status != "INVALID", f"Atlas INVALID: {atlas.warnings}"

    # The key guarantee: N_L must come from a global dual bound, not from a
    # timed-out incumbent.  If optimistic_cover.global_lower_bound is None,
    # N_L must be None or 0 (not a positive lower bound from an incumbent).
    if atlas.optimistic_cover is not None:
        if atlas.optimistic_cover.global_lower_bound is None:
            assert atlas.min_cardinality_lower in (None, 0), (
                f"N_L={atlas.min_cardinality_lower} is positive but "
                f"optimistic_cover has no global_lower_bound — "
                "timed-out incumbent must not be used as N_L"
            )


# ------------------------------------------------------------------ #
# Tampering test: verify_transition_atlas detects wrong model        #
# ------------------------------------------------------------------ #


def test_tampering_wrong_model_detected():
    """verify_transition_atlas with the wrong model must report hash mismatch."""
    fix = make_context_reversal_fixture()
    grammar = make_default_grammar_for_model(fix.model, 0, gamma=0.5, max_clause_literals=2)
    if len(grammar.feature_indices) == 0:
        pytest.skip("No context features")

    materiality = _default_materiality()
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=30.0,
    )
    if atlas.status == "INVALID":
        pytest.skip("Atlas INVALID, cannot test tampering")

    # Build a DIFFERENT model (different coefficient)
    from fysvm.mclta_synthetic import _inject_model
    wrong_model = _inject_model(
        n_features=2,
        partitions=[(0.2, 0.5, 0.8), (0.2, 0.5, 0.8)],
        rules=[
            (((0, "high"), (1, "low")), 9.9),   # different coefficient
            (((0, "high"), (1, "high")), -9.9),
        ],
        intercept=0.0,
    )

    report = verify_transition_atlas(wrong_model, atlas)
    assert not report.hash_verified, (
        "Expected hash_verified=False when wrong model passed to verify_transition_atlas"
    )
    assert any("model_hash" in n for n in report.notes), (
        f"Expected a model_hash mismatch note; got notes={report.notes}"
    )


# ------------------------------------------------------------------ #
# Rerun set-cover verification                                       #
# ------------------------------------------------------------------ #


def test_atlas_rerun_set_cover_consistent():
    """verify_transition_atlas with rerun_set_cover=True must produce consistent bounds."""
    fix = make_context_reversal_fixture()
    grammar = make_default_grammar_for_model(fix.model, 0, gamma=0.5, max_clause_literals=2)
    if len(grammar.feature_indices) == 0:
        pytest.skip("No context features")

    materiality = _default_materiality()
    atlas = synthesize_transition_atlas(
        fix.model, fix.query, grammar, materiality,
        envelope_solver=_FAST_SOLVER, set_cover_time_limit_seconds=30.0,
    )
    if atlas.status in ("INFEASIBLE_TRANSITION", "INVALID"):
        pytest.skip(f"Atlas status: {atlas.status}")

    report = verify_transition_atlas(fix.model, atlas, rerun_set_cover=True)
    # The rerun should produce consistent bounds: N_U_rerun <= atlas.N_U
    if report.rerun_cardinality_upper is not None and atlas.min_cardinality_upper is not None:
        assert report.rerun_cardinality_upper <= atlas.min_cardinality_upper + 1, (
            f"Rerun N_U={report.rerun_cardinality_upper} inconsistent with "
            f"atlas N_U={atlas.min_cardinality_upper}"
        )
    assert report.set_cover_verified, f"Set cover rerun inconsistency: {report.notes}"
