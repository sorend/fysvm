"""Tests for Specification Fidelity: Independent scalar executable spec + metamorphic conformance."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fysvm import FuzzyRuleSVM
from fysvm.conformance import (
    ConformanceResult,
    MetamorphicResult,
    check_conformance,
    check_metamorphic_relations,
)
import reference_implementations as ref


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_2d_data():
    """Small 2-feature separable dataset."""
    rng = np.random.default_rng(7)
    X_pos = rng.normal(loc=1.0, scale=0.3, size=(30, 2))
    X_neg = rng.normal(loc=-1.0, scale=0.3, size=(30, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 30 + [0] * 30)
    return X, y


@pytest.fixture(scope="module")
def clf_min(small_2d_data):
    X, y = small_2d_data
    clf = FuzzyRuleSVM(
        C=1.0, penalty="l1", and_operator="min",
        max_rule_length=2, max_rules=32, random_state=42,
    )
    clf.fit(X, y)
    return clf


@pytest.fixture(scope="module")
def clf_product(small_2d_data):
    X, y = small_2d_data
    clf = FuzzyRuleSVM(
        C=1.0, penalty="l1", and_operator="product",
        max_rule_length=2, max_rules=32, random_state=42,
    )
    clf.fit(X, y)
    return clf


@pytest.fixture(
    scope="module",
    params=[0.1, 0.35],
    ids=["eta_0.1_default", "eta_0.35_nondefault"],
)
def clf_softmin(small_2d_data, request):
    X, y = small_2d_data
    clf = FuzzyRuleSVM(
        C=1.0, penalty="l1", and_operator="softmin",
        softmin_temperature=request.param,
        max_rule_length=2, max_rules=32, random_state=42,
    )
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# test_reference_membership_boundary_conditions
# ---------------------------------------------------------------------------

class TestReferenceMembershipBoundaryConditions:
    """MR-6 from the design spec: 7 analytically known scalar cases."""

    @pytest.mark.parametrize("q_low,q_mid,q_high", [
        (0.0, 1.0, 2.0),
        (-3.5, 0.0, 3.5),
        (1.0, 5.0, 9.0),
    ])
    def test_at_q_low(self, q_low, q_mid, q_high):
        v = q_low
        assert ref.triangular_low(v, q_low, q_mid, q_high) == pytest.approx(1.0, abs=1e-12)
        assert ref.triangular_medium(v, q_low, q_mid, q_high) == pytest.approx(0.0, abs=1e-12)
        assert ref.triangular_high(v, q_low, q_mid, q_high) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("q_low,q_mid,q_high", [
        (0.0, 1.0, 2.0),
        (-3.5, 0.0, 3.5),
        (1.0, 5.0, 9.0),
    ])
    def test_at_q_mid(self, q_low, q_mid, q_high):
        v = q_mid
        assert ref.triangular_low(v, q_low, q_mid, q_high) == pytest.approx(0.0, abs=1e-12)
        assert ref.triangular_medium(v, q_low, q_mid, q_high) == pytest.approx(1.0, abs=1e-12)
        assert ref.triangular_high(v, q_low, q_mid, q_high) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("q_low,q_mid,q_high", [
        (0.0, 1.0, 2.0),
        (-3.5, 0.0, 3.5),
        (1.0, 5.0, 9.0),
    ])
    def test_at_q_high(self, q_low, q_mid, q_high):
        v = q_high
        assert ref.triangular_low(v, q_low, q_mid, q_high) == pytest.approx(0.0, abs=1e-12)
        assert ref.triangular_medium(v, q_low, q_mid, q_high) == pytest.approx(0.0, abs=1e-12)
        assert ref.triangular_high(v, q_low, q_mid, q_high) == pytest.approx(1.0, abs=1e-12)

    def test_midpoints(self):
        q_low, q_mid, q_high = 0.0, 1.0, 2.0
        # midpoint low–mid: μ_low=0.5, μ_med=0.5, μ_high=0
        v_lm = (q_low + q_mid) / 2
        assert ref.triangular_low(v_lm, q_low, q_mid, q_high) == pytest.approx(0.5, abs=1e-12)
        assert ref.triangular_medium(v_lm, q_low, q_mid, q_high) == pytest.approx(0.5, abs=1e-12)
        assert ref.triangular_high(v_lm, q_low, q_mid, q_high) == pytest.approx(0.0, abs=1e-12)
        # midpoint mid–high: μ_low=0, μ_med=0.5, μ_high=0.5
        v_mh = (q_mid + q_high) / 2
        assert ref.triangular_low(v_mh, q_low, q_mid, q_high) == pytest.approx(0.0, abs=1e-12)
        assert ref.triangular_medium(v_mh, q_low, q_mid, q_high) == pytest.approx(0.5, abs=1e-12)
        assert ref.triangular_high(v_mh, q_low, q_mid, q_high) == pytest.approx(0.5, abs=1e-12)

    def test_far_left_low_is_one(self):
        q_low, q_mid, q_high = 0.0, 1.0, 2.0
        v = q_low - 1000.0
        assert ref.triangular_low(v, q_low, q_mid, q_high) == pytest.approx(1.0, abs=1e-12)
        assert ref.triangular_medium(v, q_low, q_mid, q_high) == pytest.approx(0.0, abs=1e-12)
        assert ref.triangular_high(v, q_low, q_mid, q_high) == pytest.approx(0.0, abs=1e-12)

    def test_far_right_high_is_one(self):
        q_low, q_mid, q_high = 0.0, 1.0, 2.0
        v = q_high + 1000.0
        assert ref.triangular_low(v, q_low, q_mid, q_high) == pytest.approx(0.0, abs=1e-12)
        assert ref.triangular_medium(v, q_low, q_mid, q_high) == pytest.approx(0.0, abs=1e-12)
        assert ref.triangular_high(v, q_low, q_mid, q_high) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# test_reference_min_tnorm_matches_production
# ---------------------------------------------------------------------------

class TestReferenceMinTnormMatchesProduction:
    """Reference membership + activation matches production for min t-norm."""

    def test_membership_tensor_matches(self, clf_min, small_2d_data):
        X, _ = small_2d_data
        X_model = X[:, clf_min.selected_feature_indices_]

        ref_memberships = ref.compute_membership_matrix_ref(X_model, clf_min.partitions_)
        prod_memberships = clf_min._concept_membership_tensor(X_model)

        max_err = float(np.max(np.abs(ref_memberships - prod_memberships)))
        assert max_err < 1e-10, f"Membership tensor mismatch (max_err={max_err:.2e})"

    def test_activation_matrix_matches(self, clf_min, small_2d_data):
        X, _ = small_2d_data
        X_model = X[:, clf_min.selected_feature_indices_]

        ref_memberships = ref.compute_membership_matrix_ref(X_model, clf_min.partitions_)
        ref_Z = ref.compute_activation_matrix_ref(
            ref_memberships, clf_min.rules_, and_operator="min"
        )
        prod_Z = clf_min.transform(X)

        max_err = float(np.max(np.abs(ref_Z - prod_Z)))
        assert max_err < 1e-10, f"Activation matrix mismatch (max_err={max_err:.2e})"

    def test_decision_function_matches(self, clf_min, small_2d_data):
        X, _ = small_2d_data
        X_model = X[:, clf_min.selected_feature_indices_]

        ref_memberships = ref.compute_membership_matrix_ref(X_model, clf_min.partitions_)
        ref_Z = ref.compute_activation_matrix_ref(
            ref_memberships, clf_min.rules_, and_operator="min"
        )
        ref_df = ref.decision_function_ref(ref_Z, clf_min.coef_, clf_min.intercept_)
        prod_df = clf_min.decision_function(X)

        max_err = float(np.max(np.abs(ref_df - prod_df)))
        assert max_err < 1e-10, f"Decision function mismatch (max_err={max_err:.2e})"


# ---------------------------------------------------------------------------
# test_reference_product_tnorm_matches_production
# ---------------------------------------------------------------------------

class TestReferenceProductTnormMatchesProduction:
    """Reference membership + activation matches production for product t-norm."""

    def test_membership_tensor_matches(self, clf_product, small_2d_data):
        X, _ = small_2d_data
        X_model = X[:, clf_product.selected_feature_indices_]

        ref_memberships = ref.compute_membership_matrix_ref(X_model, clf_product.partitions_)
        prod_memberships = clf_product._concept_membership_tensor(X_model)

        max_err = float(np.max(np.abs(ref_memberships - prod_memberships)))
        assert max_err < 1e-10, f"Membership tensor mismatch (max_err={max_err:.2e})"

    def test_activation_matrix_matches(self, clf_product, small_2d_data):
        X, _ = small_2d_data
        X_model = X[:, clf_product.selected_feature_indices_]

        ref_memberships = ref.compute_membership_matrix_ref(X_model, clf_product.partitions_)
        ref_Z = ref.compute_activation_matrix_ref(
            ref_memberships, clf_product.rules_, and_operator="product"
        )
        prod_Z = clf_product.transform(X)

        max_err = float(np.max(np.abs(ref_Z - prod_Z)))
        assert max_err < 1e-10, f"Activation matrix mismatch (max_err={max_err:.2e})"

    def test_decision_function_matches(self, clf_product, small_2d_data):
        X, _ = small_2d_data
        X_model = X[:, clf_product.selected_feature_indices_]

        ref_memberships = ref.compute_membership_matrix_ref(X_model, clf_product.partitions_)
        ref_Z = ref.compute_activation_matrix_ref(
            ref_memberships, clf_product.rules_, and_operator="product"
        )
        ref_df = ref.decision_function_ref(ref_Z, clf_product.coef_, clf_product.intercept_)
        prod_df = clf_product.decision_function(X)

        max_err = float(np.max(np.abs(ref_df - prod_df)))
        assert max_err < 1e-10, f"Decision function mismatch (max_err={max_err:.2e})"


# ---------------------------------------------------------------------------
# test_metamorphic_row_permutation (MR1)
# ---------------------------------------------------------------------------

class TestMetamorphicRowPermutation:
    """MR1: predict(X[π]) == predict(X)[π] for row permutation π."""

    def test_predict_permutation(self, clf_min, small_2d_data):
        X, y = small_2d_data
        rng = np.random.default_rng(123)
        perm = rng.permutation(len(X))

        pred_shuffled = clf_min.predict(X[perm])
        pred_original_shuffled = clf_min.predict(X)[perm]

        assert np.all(pred_shuffled == pred_original_shuffled), (
            "Predictions are not invariant to row permutation."
        )

    def test_decision_function_permutation(self, clf_min, small_2d_data):
        X, y = small_2d_data
        rng = np.random.default_rng(999)
        perm = rng.permutation(len(X))

        df_perm = clf_min.decision_function(X[perm])
        df_orig_perm = clf_min.decision_function(X)[perm]

        max_err = float(np.max(np.abs(df_perm - df_orig_perm)))
        assert max_err <= 1e-12, (
            f"Decision function not invariant to row permutation (max_err={max_err:.2e})."
        )

    def test_mr1_via_check_metamorphic(self, clf_min, small_2d_data):
        X, y = small_2d_data
        results = check_metamorphic_relations(clf_min, X, y, random_state=0)
        mr1 = next(r for r in results if r.relation_name == "MR1_row_permutation")
        assert mr1.passed, (
            f"MR1 row permutation failed: n_violations={mr1.n_violations}, "
            f"max_violation={mr1.max_violation:.2e}"
        )


# ---------------------------------------------------------------------------
# test_metamorphic_membership_boundaries (MR4)
# ---------------------------------------------------------------------------

class TestMetamorphicMembershipBoundaries:
    """MR4: at partition anchors, membership values match exact specification."""

    @pytest.mark.parametrize(
        "anchors,expected",
        [
            ((0.0, 0.0, 2.0), [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            ((0.0, 2.0, 2.0), [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
            ((1.0, 1.0, 1.0), [[1.0, 1.0, 1.0]] * 3),
        ],
        ids=["low_equals_mid", "mid_equals_high", "all_equal"],
    )
    def test_degenerate_anchor_semantics_match_reference_and_production(
        self, clf_min, anchors, expected
    ):
        partition = type(clf_min.partitions_[0])(*anchors)
        X_test = np.asarray(anchors, dtype=np.float64)[:, np.newaxis]

        ref_memberships = ref.compute_membership_matrix_ref(X_test, [partition])[:, 0, :]
        prod_memberships = partition.transform(X_test[:, 0])

        np.testing.assert_allclose(ref_memberships, expected, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(prod_memberships, expected, atol=1e-12, rtol=0.0)

    @pytest.mark.parametrize(
        "anchors",
        [(0.0, 0.0, 2.0), (0.0, 2.0, 2.0), (1.0, 1.0, 1.0)],
        ids=["low_equals_mid", "mid_equals_high", "all_equal"],
    )
    def test_mr4_is_tie_aware(self, clf_min, small_2d_data, anchors):
        X, y = small_2d_data
        clf = deepcopy(clf_min)
        clf.partitions_[0] = type(clf.partitions_[0])(*anchors)

        results = check_metamorphic_relations(clf, X, y, random_state=0)
        mr4 = next(r for r in results if r.relation_name == "MR4_membership_boundaries")

        assert mr4.passed, (
            f"MR4 rejected tied anchors {anchors}: "
            f"n_violations={mr4.n_violations}, max_violation={mr4.max_violation:.2e}"
        )

    def test_production_boundaries_for_min(self, clf_min):
        tol = 1e-12
        expected = {
            0: [1.0, 0.0, 0.0],   # q_low
            1: [0.0, 1.0, 0.0],   # q_mid
            2: [0.0, 0.0, 1.0],   # q_high
        }
        n_screened = len(clf_min.partitions_)
        for j, partition in enumerate(clf_min.partitions_):
            X_test = np.zeros((3, n_screened), dtype=np.float64)
            X_test[0, j] = partition.low
            X_test[1, j] = partition.medium
            X_test[2, j] = partition.high

            memberships = clf_min._concept_membership_tensor(X_test)
            for row_idx in range(3):
                exp_vals = expected[row_idx]
                act_vals = memberships[row_idx, j, :].tolist()
                for term_idx, (exp_v, act_v) in enumerate(zip(exp_vals, act_vals)):
                    assert abs(exp_v - act_v) <= tol, (
                        f"Feature {j}, anchor {['q_low','q_mid','q_high'][row_idx]}, "
                        f"term {['low','med','high'][term_idx]}: "
                        f"expected {exp_v}, got {act_v}"
                    )

    def test_reference_boundaries_match_production(self, clf_min):
        """Reference and production must agree at partition anchors."""
        n_screened = len(clf_min.partitions_)
        for j, partition in enumerate(clf_min.partitions_):
            X_test = np.zeros((3, n_screened), dtype=np.float64)
            X_test[0, j] = partition.low
            X_test[1, j] = partition.medium
            X_test[2, j] = partition.high

            ref_memberships = ref.compute_membership_matrix_ref(X_test, clf_min.partitions_)
            prod_memberships = clf_min._concept_membership_tensor(X_test)

            max_err = float(np.max(np.abs(ref_memberships - prod_memberships)))
            assert max_err <= 1e-12, (
                f"Feature {j}: reference/production boundary disagreement "
                f"(max_err={max_err:.2e})"
            )

    def test_mr4_via_check_metamorphic(self, clf_min, small_2d_data):
        X, y = small_2d_data
        results = check_metamorphic_relations(clf_min, X, y, random_state=0)
        mr4 = next(r for r in results if r.relation_name == "MR4_membership_boundaries")
        assert mr4.passed, (
            f"MR4 membership boundaries failed: n_violations={mr4.n_violations}, "
            f"max_violation={mr4.max_violation:.2e}"
        )


# ---------------------------------------------------------------------------
# test_metamorphic_explanation_additivity (MR5)
# ---------------------------------------------------------------------------

class TestMetamorphicExplanationAdditivity:
    """MR5: contributions sum to margin - intercept (all rules exposed)."""

    def test_additivity_sum_contributions_equals_net(self, clf_min, small_2d_data):
        X, _ = small_2d_data
        explanations = clf_min.explain(X, top_n=clf_min.n_rules_, min_abs_contribution=0.0)
        for i, expl in enumerate(explanations):
            sum_contribs = sum(item["contribution"] for item in expl["top_rules"])
            net = expl["net_rule_contribution"]
            err = abs(sum_contribs - net)
            assert err < 1e-10, (
                f"Sample {i}: sum(contributions)={sum_contribs:.12f} != "
                f"net_rule_contribution={net:.12f} (err={err:.2e})"
            )

    def test_additivity_net_plus_bias_equals_margin(self, clf_min, small_2d_data):
        X, _ = small_2d_data
        explanations = clf_min.explain(X, top_n=clf_min.n_rules_, min_abs_contribution=0.0)
        for i, expl in enumerate(explanations):
            net = expl["net_rule_contribution"]
            bias = expl["bias"]
            margin = expl["margin"]
            err = abs(net + bias - margin)
            assert err < 1e-10, (
                f"Sample {i}: net + bias = {net + bias:.12f} != margin={margin:.12f} "
                f"(err={err:.2e})"
            )

    def test_mr5_via_check_metamorphic(self, clf_min, small_2d_data):
        X, y = small_2d_data
        results = check_metamorphic_relations(clf_min, X, y, random_state=0)
        mr5 = next(r for r in results if r.relation_name == "MR5_explanation_additivity")
        assert mr5.passed, (
            f"MR5 explanation additivity failed: n_violations={mr5.n_violations}, "
            f"max_violation={mr5.max_violation:.2e}"
        )

    def test_additivity_product_torm(self, clf_product, small_2d_data):
        """Additivity also holds for product t-norm."""
        X, _ = small_2d_data
        explanations = clf_product.explain(X, top_n=clf_product.n_rules_, min_abs_contribution=0.0)
        for i, expl in enumerate(explanations):
            net = expl["net_rule_contribution"]
            bias = expl["bias"]
            margin = expl["margin"]
            err = abs(net + bias - margin)
            assert err < 1e-10, (
                f"Sample {i} (product): net + bias = {net + bias:.12f} != "
                f"margin={margin:.12f} (err={err:.2e})"
            )


# ---------------------------------------------------------------------------
# test_conformance_returns_certified_for_min_tnorm
# ---------------------------------------------------------------------------

class TestConformanceCertifiedMinTnorm:
    """check_conformance returns CERTIFIED for and_operator='min'."""

    def test_status_is_certified(self, clf_min, small_2d_data):
        X, _ = small_2d_data
        result = check_conformance(clf_min, X, "test_min")
        assert isinstance(result, ConformanceResult)
        assert result.status == "CERTIFIED", (
            f"Expected CERTIFIED, got {result.status}. "
            f"max_abs_error={result.max_abs_error:.2e}"
        )

    def test_max_abs_error_below_tolerance(self, clf_min, small_2d_data):
        X, _ = small_2d_data
        result = check_conformance(clf_min, X, "test_min", tolerance=1e-10)
        assert result.max_abs_error < 1e-10, (
            f"max_abs_error={result.max_abs_error:.2e} exceeds 1e-10"
        )

    def test_result_fields(self, clf_min, small_2d_data):
        X, _ = small_2d_data
        result = check_conformance(clf_min, X, "test_min")
        assert result.dataset_name == "test_min"
        assert result.and_operator == "min"
        assert result.certificate_eligibility_status == "ELIGIBLE"
        assert result.n_samples == len(X)
        assert result.mean_abs_error >= 0.0
        assert "max_membership_error" in result.details
        assert "max_activation_error" in result.details
        assert "max_decision_function_error" in result.details


# ---------------------------------------------------------------------------
# test_conformance_returns_certified_for_product_tnorm
# ---------------------------------------------------------------------------

class TestConformanceCertifiedProductTnorm:
    """check_conformance returns CERTIFIED for and_operator='product'."""

    def test_status_is_certified(self, clf_product, small_2d_data):
        X, _ = small_2d_data
        result = check_conformance(clf_product, X, "test_product")
        assert result.status == "CERTIFIED", (
            f"Expected CERTIFIED, got {result.status}. "
            f"max_abs_error={result.max_abs_error:.2e}"
        )
        assert result.certificate_eligibility_status == "ELIGIBLE"

    def test_max_abs_error_below_tolerance(self, clf_product, small_2d_data):
        X, _ = small_2d_data
        result = check_conformance(clf_product, X, "test_product", tolerance=1e-10)
        assert result.max_abs_error < 1e-10, (
            f"max_abs_error={result.max_abs_error:.2e} exceeds 1e-10"
        )


# ---------------------------------------------------------------------------
# test_conformance_returns_certified_but_ineligible_for_softmin
# ---------------------------------------------------------------------------

class TestConformanceCertifiedSoftmin:
    """Softmin can conform while remaining outside property-certificate theory."""

    def test_status_is_certified(self, clf_softmin, small_2d_data):
        X, _ = small_2d_data
        result = check_conformance(clf_softmin, X, "test_softmin")
        assert result.status == "CERTIFIED", (
            f"Expected CERTIFIED, got {result.status}; "
            f"max_abs_error={result.max_abs_error:.2e}"
        )

    def test_certificate_eligibility_is_ineligible(self, clf_softmin, small_2d_data):
        X, _ = small_2d_data
        result = check_conformance(clf_softmin, X, "test_softmin")
        assert result.and_operator == "softmin"
        assert result.certificate_eligibility_status == "INELIGIBLE"

    def test_default_and_nondefault_eta_conform(self, clf_softmin, small_2d_data):
        X, _ = small_2d_data
        result = check_conformance(clf_softmin, X, "test_softmin")
        assert clf_softmin.softmin_temperature in {0.1, 0.35}
        assert result.max_abs_error < 1e-10, (
            f"Softmin error unexpectedly large: {result.max_abs_error:.2e}"
        )


class TestConformanceCounterexampleWitness:
    """Counterexamples identify the component with the largest measured error."""

    def test_activation_witness_when_activation_has_max_error(
        self, clf_min, small_2d_data, monkeypatch
    ):
        X, _ = small_2d_data
        clf = deepcopy(clf_min)
        clf.coef_ = np.zeros_like(clf.coef_)
        original_compute_activations = ref.compute_activation_matrix_ref

        def mismatching_activations(*args, **kwargs):
            activations = original_compute_activations(*args, **kwargs).copy()
            activations[0, 0] += 0.25
            return activations

        monkeypatch.setattr(ref, "compute_activation_matrix_ref", mismatching_activations)

        result = check_conformance(clf, X, "activation_counterexample")
        witness = result.details["counterexample"]

        assert result.status == "COUNTEREXAMPLE"
        assert result.certificate_eligibility_status == "ELIGIBLE"
        assert witness["component"] == "activation"
        assert witness["sample_index"] == 0
        assert witness["rule_index"] == 0
        assert witness["absolute_error"] == pytest.approx(0.25)
        assert witness["reference_value"] - witness["production_value"] == pytest.approx(0.25)
