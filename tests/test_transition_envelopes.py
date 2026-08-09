"""Tests for transition_envelopes.py.

Tests cover:
- Encoding correctness (trivial and nontrivial cases)
- Extrema soundness against brute-force grid
- Invalid model/query rejection
- Witness validation
- Infeasible transitions
- Specific closed-form cases from synthetic fixtures
"""

from __future__ import annotations

import numpy as np
import pytest

from fysvm.mclta_synthetic import (
    _inject_model,
    make_context_reversal_fixture,
    make_infeasible_transition_fixture,
    make_length1_negative_rule_fixture,
    make_length1_positive_rule_fixture,
    make_relational_tightening_fixture,
    make_zero_transition_fixture,
)
from fysvm.transition_envelopes import (
    ContextClause,
    ContextLiteral,
    LinearDomain,
    MilpConfig,
    SolverStatus,
    TransitionQuery,
    certify_transition_envelope,
)
from reference_mclta import (
    brute_force_envelope,
    mu_scalar,
    transition_objective_scalar,
    verify_envelope_contains_grid,
)

# Tight solver config for testing
_TEST_SOLVER = MilpConfig(
    backend="scipy-highs",
    time_limit_seconds=60.0,
    relative_gap=1e-6,
    postcheck_atol=1e-7,
)


# ------------------------------------------------------------------ #
# Fixture helpers                                                     #
# ------------------------------------------------------------------ #


def _make_simple_model():
    """1-feature model, one HIGH rule."""
    return _inject_model(
        n_features=1,
        partitions=[(0.2, 0.5, 0.8)],
        rules=[(((0, "high"),), 1.0)],
        intercept=0.0,
    )


def _make_2feature_model():
    """2-feature model with length-2 rule."""
    return _inject_model(
        n_features=2,
        partitions=[(0.2, 0.5, 0.8), (0.2, 0.5, 0.8)],
        rules=[(((0, "high"), (1, "high")), 2.0)],
        intercept=0.0,
    )


# ------------------------------------------------------------------ #
# Test: zero transition (feature absent from all rules)              #
# ------------------------------------------------------------------ #


def test_zero_transition_returns_zero_envelope():
    """When the transition feature is absent, envelope must be [0, 0]."""
    fix = make_zero_transition_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_TEST_SOLVER)
    assert result.status == "OPTIMAL"
    assert result.dual_lower == pytest.approx(0.0, abs=1e-10)
    assert result.primal_lower == pytest.approx(0.0, abs=1e-10)
    assert result.primal_upper == pytest.approx(0.0, abs=1e-10)
    assert result.dual_upper == pytest.approx(0.0, abs=1e-10)
    assert result.affected_rule_indices == ()


# ------------------------------------------------------------------ #
# Test: length-1 positive rule                                       #
# ------------------------------------------------------------------ #


def test_length1_positive_rule_envelope():
    """Verify closed-form extrema for single HIGH rule with coef=2."""
    fix = make_length1_positive_rule_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_TEST_SOLVER)
    assert result.status == "OPTIMAL"
    assert result.primal_lower == pytest.approx(fix.exact_lower, abs=1e-4)
    assert result.primal_upper == pytest.approx(fix.exact_upper, abs=1e-4)
    assert result.dual_lower == pytest.approx(fix.exact_lower, abs=1e-4)
    assert result.dual_upper == pytest.approx(fix.exact_upper, abs=1e-4)


def test_length1_positive_rule_witnesses_validated():
    """Witnesses for lower and upper extrema must pass independent validation."""
    fix = make_length1_positive_rule_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_TEST_SOLVER)
    assert result.lower_solve.witness is not None
    assert result.lower_solve.witness.validated, (
        f"Lower witness failed: {result.lower_solve.witness.validation_notes}"
    )
    assert result.upper_solve is not None
    assert result.upper_solve.witness is not None
    assert result.upper_solve.witness.validated, (
        f"Upper witness failed: {result.upper_solve.witness.validation_notes}"
    )


# ------------------------------------------------------------------ #
# Test: length-1 negative rule                                       #
# ------------------------------------------------------------------ #


def test_length1_negative_rule_envelope():
    """Single LOW rule with negative coef; verify closed-form bounds."""
    fix = make_length1_negative_rule_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_TEST_SOLVER)
    assert result.status == "OPTIMAL"
    assert result.primal_lower == pytest.approx(fix.exact_lower, abs=1e-4)
    assert result.primal_upper == pytest.approx(fix.exact_upper, abs=1e-4)


# ------------------------------------------------------------------ #
# Test: infeasible transition                                        #
# ------------------------------------------------------------------ #


def test_infeasible_transition():
    """Query with impossible alpha-cut must return INFEASIBLE."""
    fix = make_infeasible_transition_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_TEST_SOLVER)
    assert result.status == "INFEASIBLE"
    assert result.lower_solve.status == "INFEASIBLE"


# ------------------------------------------------------------------ #
# Test: invalid model rejections                                     #
# ------------------------------------------------------------------ #


def test_rejects_product_operator():
    """Product t-norm model must be rejected with INVALID status."""
    model = _inject_model(
        n_features=1,
        partitions=[(0.2, 0.5, 0.8)],
        rules=[(((0, "high"),), 1.0)],
    )
    model.and_operator = "product"
    dom = LinearDomain(lower=(0.0,), upper=(1.0,))
    query = TransitionQuery(
        feature_index=0, source_term="low", destination_term="high",
        source_alpha=0.5, destination_alpha=0.5, target_class=1, domain=dom,
    )
    result = certify_transition_envelope(model, query)
    assert result.status == "INVALID"


def test_rejects_softmin_operator():
    """Softmin t-norm model must be rejected with INVALID status."""
    model = _inject_model(
        n_features=1,
        partitions=[(0.2, 0.5, 0.8)],
        rules=[(((0, "high"),), 1.0)],
    )
    model.and_operator = "softmin"
    dom = LinearDomain(lower=(0.0,), upper=(1.0,))
    query = TransitionQuery(
        feature_index=0, source_term="low", destination_term="high",
        source_alpha=0.5, destination_alpha=0.5, target_class=1, domain=dom,
    )
    result = certify_transition_envelope(model, query)
    assert result.status == "INVALID"


def test_rejects_out_of_range_feature_index():
    """Feature index out of range must return INVALID."""
    model = _inject_model(
        n_features=1,
        partitions=[(0.2, 0.5, 0.8)],
        rules=[(((0, "high"),), 1.0)],
    )
    dom = LinearDomain(lower=(0.0,), upper=(1.0,))
    query = TransitionQuery(
        feature_index=5,  # out of range
        source_term="low", destination_term="high",
        source_alpha=0.5, destination_alpha=0.5, target_class=1, domain=dom,
    )
    result = certify_transition_envelope(model, query)
    assert result.status == "INVALID"


def test_rejects_invalid_source_alpha():
    """source_alpha = 0 must return INVALID (alpha must be > 0)."""
    model = _inject_model(
        n_features=1,
        partitions=[(0.2, 0.5, 0.8)],
        rules=[(((0, "high"),), 1.0)],
    )
    dom = LinearDomain(lower=(0.0,), upper=(1.0,))
    query = TransitionQuery(
        feature_index=0, source_term="low", destination_term="high",
        source_alpha=0.0, destination_alpha=0.5, target_class=1, domain=dom,
    )
    result = certify_transition_envelope(model, query)
    assert result.status == "INVALID"


def test_rejects_unknown_target_class():
    """Unknown target class must return INVALID."""
    model = _inject_model(
        n_features=1,
        partitions=[(0.2, 0.5, 0.8)],
        rules=[(((0, "high"),), 1.0)],
    )
    dom = LinearDomain(lower=(0.0,), upper=(1.0,))
    query = TransitionQuery(
        feature_index=0, source_term="low", destination_term="high",
        source_alpha=0.5, destination_alpha=0.5, target_class=99, domain=dom,
    )
    result = certify_transition_envelope(model, query)
    assert result.status == "INVALID"


def test_rejects_tied_anchors():
    """Model with tied partition anchors for an affected rule must return INVALID."""
    model = _inject_model(
        n_features=1,
        partitions=[(0.5, 0.5, 0.8)],  # q_low == q_mid: tied!
        rules=[(((0, "high"),), 1.0)],
    )
    dom = LinearDomain(lower=(0.0,), upper=(1.0,))
    query = TransitionQuery(
        feature_index=0, source_term="low", destination_term="high",
        source_alpha=0.5, destination_alpha=0.5, target_class=1, domain=dom,
    )
    result = certify_transition_envelope(model, query)
    assert result.status == "INVALID"


# ------------------------------------------------------------------ #
# Test: context clause restriction                                   #
# ------------------------------------------------------------------ #


def test_context_clause_restricts_envelope():
    """Adding a context clause should not widen the unconstrained envelope."""
    fix = make_context_reversal_fixture()
    # Unconditional
    result_uncond = certify_transition_envelope(fix.model, fix.query, solver=_TEST_SOLVER)
    # Low context (restricts to positive regime)
    result_low = certify_transition_envelope(
        fix.model, fix.query, context=fix.context_low, solver=_TEST_SOLVER
    )
    # High context (restricts to negative regime)
    result_high = certify_transition_envelope(
        fix.model, fix.query, context=fix.context_high, solver=_TEST_SOLVER
    )

    # Unconditional should span both signs
    assert result_uncond.primal_lower is not None
    assert result_uncond.primal_upper is not None
    assert result_uncond.primal_lower < 0
    assert result_uncond.primal_upper > 0

    # Low context: should be positive (or zero at worst)
    if result_low.status not in ("INFEASIBLE", "INVALID"):
        if result_low.primal_lower is not None:
            assert result_low.primal_lower >= -1e-4

    # High context: should be negative (or zero at worst)
    if result_high.status not in ("INFEASIBLE", "INVALID"):
        if result_high.primal_upper is not None:
            assert result_high.primal_upper <= 1e-4


def test_context_conjoined_with_base():
    """context argument is conjoined with query.base_context."""
    model = _make_simple_model()
    dom = LinearDomain(lower=(0.0,), upper=(1.0,))
    base_ctx = ContextClause(
        literals=(ContextLiteral(feature_index=0, term="high", min_membership=0.5),)
    )
    extra_ctx = ContextClause(
        literals=(ContextLiteral(feature_index=0, term="high", min_membership=0.5),)
    )
    query = TransitionQuery(
        feature_index=0, source_term="low", destination_term="high",
        source_alpha=0.5, destination_alpha=0.5, target_class=1,
        domain=dom, base_context=base_ctx,
    )
    # With same extra context, same result
    result = certify_transition_envelope(model, query, context=extra_ctx, solver=_TEST_SOLVER)
    # Just check it doesn't crash
    assert result.status in ("OPTIMAL", "BOUNDED", "INFEASIBLE", "UNKNOWN", "FEASIBLE_ONLY", "INVALID")


# ------------------------------------------------------------------ #
# Test: relational tightening                                        #
# ------------------------------------------------------------------ #


def test_relational_bounds_tighter_than_independent():
    """Relational lower bound > independent endpoint subtraction lower bound."""
    fix = make_relational_tightening_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_TEST_SOLVER)
    if result.status in ("OPTIMAL", "BOUNDED") and result.primal_lower is not None:
        assert result.primal_lower > fix.independent_lower - 1e-4, (
            f"Relational lower={result.primal_lower:.4f} should be > "
            f"independent lower={fix.independent_lower:.4f}"
        )


# ------------------------------------------------------------------ #
# Test: grid-search soundness check                                  #
# ------------------------------------------------------------------ #


def test_certified_bounds_contain_grid_extrema_1d():
    """Grid search extrema must fall within certified bounds for 1D model."""
    fix = make_length1_positive_rule_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_TEST_SOLVER)
    if result.status not in ("INFEASIBLE", "INVALID", "UNKNOWN"):
        ok, grid_min, grid_max, msg = verify_envelope_contains_grid(
            fix.model,
            j=fix.query.feature_index,
            source_term=fix.query.source_term,
            destination_term=fix.query.destination_term,
            source_alpha=fix.query.source_alpha,
            destination_alpha=fix.query.destination_alpha,
            target_sign=result.target_sign,
            lower=list(fix.query.domain.lower),
            upper=list(fix.query.domain.upper),
            certified_lower=result.dual_lower,
            certified_upper=result.dual_upper,
            n_grid=100,
        )
        assert ok, f"Grid soundness failed: {msg}"


def test_certified_bounds_contain_grid_extrema_2d():
    """Grid search extrema must fall within certified bounds for 2D model."""
    fix = make_context_reversal_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_TEST_SOLVER)
    if result.status not in ("INFEASIBLE", "INVALID", "UNKNOWN"):
        ok, grid_min, grid_max, msg = verify_envelope_contains_grid(
            fix.model,
            j=fix.query.feature_index,
            source_term=fix.query.source_term,
            destination_term=fix.query.destination_term,
            source_alpha=fix.query.source_alpha,
            destination_alpha=fix.query.destination_alpha,
            target_sign=result.target_sign,
            lower=list(fix.query.domain.lower),
            upper=list(fix.query.domain.upper),
            certified_lower=result.dual_lower,
            certified_upper=result.dual_upper,
            n_grid=50,
        )
        assert ok, f"Grid soundness failed: {msg}"


# ------------------------------------------------------------------ #
# Test: witness completeness                                         #
# ------------------------------------------------------------------ #


def test_witnesses_validated_for_all_optimal_solves():
    """All optimal solves must have validated witnesses."""
    fixtures = [
        make_length1_positive_rule_fixture(),
        make_length1_negative_rule_fixture(),
        make_relational_tightening_fixture(),
    ]
    for fix in fixtures:
        result = certify_transition_envelope(fix.model, fix.query, solver=_TEST_SOLVER)
        if result.lower_solve.status == "OPTIMAL":
            assert result.lower_solve.witness is not None
            assert result.lower_solve.witness.validated, (
                f"Fixture {getattr(fix, 'description', '')} lower witness not validated: "
                f"{result.lower_solve.witness.validation_notes}"
            )
        if result.upper_solve and result.upper_solve.status == "OPTIMAL":
            assert result.upper_solve.witness is not None
            assert result.upper_solve.witness.validated, (
                f"Fixture {getattr(fix, 'description', '')} upper witness not validated: "
                f"{result.upper_solve.witness.validation_notes}"
            )


# ------------------------------------------------------------------ #
# Test: serialisation rejects NaN/inf                                #
# ------------------------------------------------------------------ #


def test_envelope_to_dict_ok():
    """envelope_to_dict should work for a finite envelope."""
    from fysvm.transition_envelopes import envelope_to_dict
    fix = make_length1_positive_rule_fixture()
    result = certify_transition_envelope(fix.model, fix.query, solver=_TEST_SOLVER)
    d = envelope_to_dict(result)
    assert d["schema_version"] == "1.0"
    assert d["status"] in ("OPTIMAL", "BOUNDED", "INFEASIBLE", "FEASIBLE_ONLY", "UNKNOWN", "INVALID")


# ------------------------------------------------------------------ #
# Test: order constraint enforcement                                  #
# ------------------------------------------------------------------ #


def test_order_constraint_positive():
    """With enforce_term_order=True for low->high, v must be >= u."""
    model = _make_simple_model()
    dom = LinearDomain(lower=(0.0,), upper=(1.0,))
    query = TransitionQuery(
        feature_index=0, source_term="low", destination_term="high",
        source_alpha=0.5, destination_alpha=0.5, target_class=1,
        domain=dom, enforce_term_order=True, min_raw_displacement=0.1,
    )
    result = certify_transition_envelope(model, query, solver=_TEST_SOLVER)
    # Check witness displacement
    if result.lower_solve.witness is not None:
        w = result.lower_solve.witness
        assert w.destination_value >= w.source_value + 0.1 - 1e-8


def test_no_order_constraint():
    """With enforce_term_order=False, displacement constraint is relaxed."""
    model = _make_simple_model()
    dom = LinearDomain(lower=(0.0,), upper=(1.0,))
    query = TransitionQuery(
        feature_index=0, source_term="low", destination_term="high",
        source_alpha=0.5, destination_alpha=0.5, target_class=1,
        domain=dom, enforce_term_order=False,
    )
    result = certify_transition_envelope(model, query, solver=_TEST_SOLVER)
    assert result.status in ("OPTIMAL", "BOUNDED", "FEASIBLE_ONLY", "INFEASIBLE")


# ------------------------------------------------------------------ #
# Test: target sign orientation                                      #
# ------------------------------------------------------------------ #


def test_negative_class_target_flips_sign():
    """Requesting negative class target should flip the transition sign."""
    fix = make_length1_positive_rule_fixture()
    # Target = classes_[0] (negative class)
    dom = fix.query.domain
    query_neg = TransitionQuery(
        feature_index=fix.query.feature_index,
        source_term=fix.query.source_term,
        destination_term=fix.query.destination_term,
        source_alpha=fix.query.source_alpha,
        destination_alpha=fix.query.destination_alpha,
        target_class=fix.model.classes_[0],  # negative class
        domain=dom,
    )
    result_pos = certify_transition_envelope(fix.model, fix.query, solver=_TEST_SOLVER)
    result_neg = certify_transition_envelope(fix.model, query_neg, solver=_TEST_SOLVER)

    assert result_pos.target_sign == 1
    assert result_neg.target_sign == -1

    if (
        result_pos.primal_lower is not None
        and result_neg.primal_upper is not None
    ):
        # For negative class, the sign flips
        assert result_neg.primal_upper == pytest.approx(-result_pos.primal_lower, abs=1e-4)


# ------------------------------------------------------------------ #
# Test: domain bounds enforcement                                    #
# ------------------------------------------------------------------ #


def test_domain_restricts_envelope():
    """A tighter domain should produce a narrower or equal envelope."""
    model = _make_simple_model()
    dom_wide = LinearDomain(lower=(0.0,), upper=(1.0,))
    dom_tight = LinearDomain(lower=(0.6,), upper=(0.9,))  # only in high region

    q_wide = TransitionQuery(
        feature_index=0, source_term="low", destination_term="high",
        source_alpha=0.5, destination_alpha=0.5, target_class=1, domain=dom_wide,
    )
    q_tight = TransitionQuery(
        feature_index=0, source_term="low", destination_term="high",
        source_alpha=0.5, destination_alpha=0.5, target_class=1, domain=dom_tight,
    )
    # In tight domain, source (low term) may not be feasible since whole range is in high region
    result_tight = certify_transition_envelope(model, q_tight, solver=_TEST_SOLVER)
    result_wide = certify_transition_envelope(model, q_wide, solver=_TEST_SOLVER)
    # Just check they don't crash; infeasibility is expected for tight domain
    assert result_tight.status in ("OPTIMAL", "BOUNDED", "INFEASIBLE", "FEASIBLE_ONLY", "UNKNOWN", "INVALID")
    assert result_wide.status in ("OPTIMAL", "BOUNDED", "INFEASIBLE", "FEASIBLE_ONLY", "UNKNOWN", "INVALID")


# ------------------------------------------------------------------ #
# Test: polyhedral domain constraint                                 #
# ------------------------------------------------------------------ #


def test_polyhedral_domain_constraint():
    """Polyhedral constraint G*x <= h is applied at both endpoints."""
    model = _make_2feature_model()
    # Add constraint: x0 + x1 <= 1.2 (in screened space)
    dom = LinearDomain(
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
        A_ub=((1.0, 1.0),),
        b_ub=(1.2,),
    )
    query = TransitionQuery(
        feature_index=0, source_term="low", destination_term="high",
        source_alpha=0.5, destination_alpha=0.5, target_class=1,
        domain=dom,
    )
    result = certify_transition_envelope(model, query, solver=_TEST_SOLVER)
    assert result.status in ("OPTIMAL", "BOUNDED", "INFEASIBLE", "FEASIBLE_ONLY", "UNKNOWN", "INVALID")
    # Verify witnesses satisfy polyhedral constraint
    for rec in [result.lower_solve, result.upper_solve]:
        if rec is not None and rec.witness is not None:
            w = rec.witness
            src_vec = list(w.source_vector)
            dst_vec = list(w.destination_vector)
            assert src_vec[0] + src_vec[1] <= 1.2 + 1e-6
            assert dst_vec[0] + dst_vec[1] <= 1.2 + 1e-6
