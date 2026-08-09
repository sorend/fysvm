"""Synthetic fixtures and planted transition regimes for MCLTA testing.

Provides utilities to construct FuzzyRuleSVM models with known exact
transition envelopes, making it possible to test the MILP solver against
closed-form ground truth.

All fixtures use and_operator='min' and nondegenerate triangular partitions
unless explicitly testing edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone

from fysvm.rule_svm import (
    FuzzyRule,
    FuzzyRuleSVM,
    RuleCondition,
    SparseMaxMarginFuzzyRuleMachine,
    _FuzzyPartition,
)
from fysvm.transition_envelopes import (
    ContextClause,
    ContextLiteral,
    LinearDomain,
    MilpConfig,
    TransitionQuery,
)


# ------------------------------------------------------------------ #
# Model injection helper                                             #
# ------------------------------------------------------------------ #


def _inject_model(
    n_features: int,
    partitions: list[tuple[float, float, float]],
    rules: list[tuple[tuple[tuple[int, str], ...], float]],
    intercept: float = 0.0,
    positive_class: Any = 1,
    negative_class: Any = 0,
) -> SparseMaxMarginFuzzyRuleMachine:
    """Build a FuzzyRuleSVM with prescribed partitions, rules, and coefficients.

    Parameters
    ----------
    n_features:
        Number of screened features.
    partitions:
        List of (low, medium, high) quantile anchors, one per feature.
    rules:
        List of ((conditions...), coef) where each condition is (feature, term).
    intercept:
        Decision function intercept b.
    positive_class, negative_class:
        Class labels (classes_ = [negative_class, positive_class]).

    Returns
    -------
    Fitted SparseMaxMarginFuzzyRuleMachine with injected parameters.
    """
    # Fit on dummy data to initialise sklearn attributes
    X_dummy = np.zeros((4, n_features), dtype=np.float64)
    y_dummy = np.array([negative_class, negative_class, positive_class, positive_class])
    clf = SparseMaxMarginFuzzyRuleMachine(
        and_operator="min", C=1.0, penalty="l1",
        max_rule_length=2, max_rules=256,
        random_state=0,
    )
    clf.fit(X_dummy, y_dummy)

    # Override partitions
    clf.partitions_ = [
        _FuzzyPartition(low=p[0], medium=p[1], high=p[2])
        for p in partitions
    ]

    # Override rules and coefficients
    fuzzy_rules = []
    coefs = []
    for conditions, coef in rules:
        rule_conditions = tuple(RuleCondition(feat, term) for feat, term in conditions)
        fuzzy_rules.append(FuzzyRule(rule_conditions))
        coefs.append(float(coef))

    clf.rules_ = fuzzy_rules
    clf.coef_ = np.array(coefs, dtype=np.float64)
    clf.intercept_ = float(intercept)
    clf.n_rules_ = len(fuzzy_rules)
    clf.active_rule_indices_ = np.flatnonzero(np.abs(clf.coef_) > 1e-12)
    clf.selected_feature_indices_ = np.arange(n_features, dtype=int)
    clf.n_screened_features_ = n_features
    clf.n_features_in_ = n_features
    clf.feature_names_in_ = np.array([f"x{i}" for i in range(n_features)], dtype=object)
    clf.classes_ = np.array([negative_class, positive_class])

    return clf


# ------------------------------------------------------------------ #
# Fixture 1: Constant (zero) transition                              #
# ------------------------------------------------------------------ #


@dataclass
class ZeroTransitionFixture:
    """Model with zero transition score: feature j absent from all rules."""

    model: SparseMaxMarginFuzzyRuleMachine
    query: TransitionQuery
    exact_lower: float = 0.0
    exact_upper: float = 0.0
    description: str = "Feature j absent from all nonzero rules; envelope = [0, 0]"


def make_zero_transition_fixture() -> ZeroTransitionFixture:
    """Feature j (index 0) not referenced by any nonzero rule."""
    # Only feature 1 appears in rules
    model = _inject_model(
        n_features=2,
        partitions=[(0.2, 0.5, 0.8), (0.2, 0.5, 0.8)],
        rules=[
            (((1, "high"),), 2.0),
            (((1, "low"),), -1.0),
        ],
        intercept=0.0,
    )
    dom = LinearDomain(lower=(0.0, 0.0), upper=(1.0, 1.0))
    query = TransitionQuery(
        feature_index=0,
        source_term="low",
        destination_term="high",
        source_alpha=0.5,
        destination_alpha=0.5,
        target_class=1,
        domain=dom,
    )
    return ZeroTransitionFixture(model=model, query=query)


# ------------------------------------------------------------------ #
# Fixture 2: Length-1 rule with closed-form extrema                  #
# ------------------------------------------------------------------ #


@dataclass
class OneDimTransitionFixture:
    """Model with one length-1 rule; exact extrema computable in closed form."""

    model: SparseMaxMarginFuzzyRuleMachine
    query: TransitionQuery
    exact_lower: float
    exact_upper: float
    description: str


def make_length1_positive_rule_fixture() -> OneDimTransitionFixture:
    """Single rule: IF x0 is HIGH THEN +, coef=2.0.

    Transition: low -> high on feature 0.
    Domain: [0, 1].
    Partitions: q=(0.2, 0.5, 0.8).

    f_t(z, v) - f_t(z, u) = 2 * (mu_high(v) - mu_high(u))

    Source (low term, alpha=0.5): mu_low(u) >= 0.5
      => u <= 0.2 + (1-0.5)*(0.5-0.2) = 0.35  [from linear_down going from q_low to q_mid]
      Actually: mu_low(u) = (0.5 - u) / (0.5 - 0.2) >= 0.5
               => u <= 0.5 - 0.5*(0.5-0.2) = 0.5 - 0.15 = 0.35
      So u in [0, 0.35]

    Destination (high term, alpha=0.5): mu_high(v) >= 0.5
      mu_high(v) = (v - 0.5) / (0.8 - 0.5) >= 0.5
               => v >= 0.5 + 0.5*0.3 = 0.65
      So v in [0.65, 1.0]

    max transition = 2*(mu_high(1.0) - mu_high(0)) = 2*(1 - 0) = 2.0
    min transition = 2*(mu_high(0.65) - mu_high(0.35))
                   = 2*(0.5 - 0) = 1.0  [mu_high(0.35) = 0 since 0.35 <= q_mid=0.5]
                   = 2 * (0.5 - 0) = 1.0
    """
    model = _inject_model(
        n_features=1,
        partitions=[(0.2, 0.5, 0.8)],
        rules=[(((0, "high"),), 2.0)],
        intercept=0.0,
    )
    dom = LinearDomain(lower=(0.0,), upper=(1.0,))
    query = TransitionQuery(
        feature_index=0,
        source_term="low",
        destination_term="high",
        source_alpha=0.5,
        destination_alpha=0.5,
        target_class=1,
        domain=dom,
    )
    # Exact: min = 2*(0.5 - 0) = 1.0, max = 2*(1.0 - 0) = 2.0
    return OneDimTransitionFixture(
        model=model,
        query=query,
        exact_lower=1.0,
        exact_upper=2.0,
        description="Length-1 positive rule; exact lower=1.0, upper=2.0",
    )


def make_length1_negative_rule_fixture() -> OneDimTransitionFixture:
    """Single rule: IF x0 is LOW THEN +, coef=-1.5.

    Transition: low -> high on feature 0.

    f_t(z, v) - f_t(z, u) = -1.5 * (mu_low(v) - mu_low(u))

    Source (low, alpha=0.5): mu_low(u) >= 0.5 => u in [0, 0.35]
    Dest (high, alpha=0.5): mu_high(v) >= 0.5 => v in [0.65, 1.0]

    delta_low = mu_low(v) - mu_low(u)
    mu_low(v) = 0 when v >= q_mid=0.5, so mu_low(v in [0.65,1.0]) = 0
    delta_low in [0 - 1.0, 0 - 0.5] = [-1.0, -0.5]

    f_t delta = -1.5 * delta_low in [-1.5 * (-0.5), -1.5 * (-1.0)] = [0.75, 1.5]
    """
    model = _inject_model(
        n_features=1,
        partitions=[(0.2, 0.5, 0.8)],
        rules=[(((0, "low"),), -1.5)],
        intercept=0.0,
    )
    dom = LinearDomain(lower=(0.0,), upper=(1.0,))
    query = TransitionQuery(
        feature_index=0,
        source_term="low",
        destination_term="high",
        source_alpha=0.5,
        destination_alpha=0.5,
        target_class=1,
        domain=dom,
    )
    return OneDimTransitionFixture(
        model=model,
        query=query,
        exact_lower=0.75,
        exact_upper=1.5,
        description="Length-1 rule on low term, negative coef; exact lower=0.75, upper=1.5",
    )


# ------------------------------------------------------------------ #
# Fixture 3: Context reversal requiring two clauses                  #
# ------------------------------------------------------------------ #


@dataclass
class ContextReversalFixture:
    """Model with sign reversal across context values of feature 1."""

    model: SparseMaxMarginFuzzyRuleMachine
    query: TransitionQuery
    context_low: ContextClause    # context where transition is positive
    context_high: ContextClause   # context where transition is negative
    exact_lower_unconditional: float  # unconditional (mixed)
    exact_upper_unconditional: float
    exact_lower_low_ctx: float
    exact_upper_low_ctx: float
    exact_lower_high_ctx: float
    exact_upper_high_ctx: float
    description: str


def make_context_reversal_fixture() -> ContextReversalFixture:
    """Two-feature model with interaction causing context sign reversal.

    Rules:
      R1: IF x0 is HIGH AND x1 is LOW THEN +, coef = +3.0
      R2: IF x0 is HIGH AND x1 is HIGH THEN +, coef = -3.0

    Feature 0 transition: low -> high
    Feature 1 is context.
    Partitions: q=(0.2, 0.5, 0.8) for both features.

    In low context (x1 is LOW, mu_low(x1) >= 0.5):
      Only R1 is significantly active at destination.
      Transition score change is dominated by +3 * (activation_change_R1)
      -> Positive transition

    In high context (x1 is HIGH, mu_high(x1) >= 0.5):
      Only R2 is significantly active at destination.
      Transition score change is dominated by -3 * (activation_change_R2)
      -> Negative transition
    """
    model = _inject_model(
        n_features=2,
        partitions=[(0.2, 0.5, 0.8), (0.2, 0.5, 0.8)],
        rules=[
            (((0, "high"), (1, "low")), 3.0),
            (((0, "high"), (1, "high")), -3.0),
        ],
        intercept=0.0,
    )
    dom = LinearDomain(lower=(0.0, 0.0), upper=(1.0, 1.0))
    query = TransitionQuery(
        feature_index=0,
        source_term="low",
        destination_term="high",
        source_alpha=0.5,
        destination_alpha=0.5,
        target_class=1,
        domain=dom,
    )
    ctx_low = ContextClause(
        literals=(ContextLiteral(feature_index=1, term="low", min_membership=0.5),)
    )
    ctx_high = ContextClause(
        literals=(ContextLiteral(feature_index=1, term="high", min_membership=0.5),)
    )
    return ContextReversalFixture(
        model=model,
        query=query,
        context_low=ctx_low,
        context_high=ctx_high,
        exact_lower_unconditional=-3.0,
        exact_upper_unconditional=3.0,
        exact_lower_low_ctx=1.5,    # min: phi_R1(z,v) >= min(0.5,0.5)=0.5, score = 3*0.5=1.5
        exact_upper_low_ctx=3.0,
        exact_lower_high_ctx=-3.0,
        exact_upper_high_ctx=-1.5,  # max: phi_R2(z,v) >= min(0.5,0.5)=0.5, score = -3*0.5=-1.5
        description="Sign reversal across context: low ctx -> positive, high ctx -> negative",
    )


# ------------------------------------------------------------------ #
# Fixture 4: Three ordered direction regimes                         #
# ------------------------------------------------------------------ #


@dataclass
class ThreeRegimeFixture:
    """Model with three distinct direction regimes."""

    model: SparseMaxMarginFuzzyRuleMachine
    query: TransitionQuery
    grammar_bins: list[ContextLiteral]  # three bins for feature 1
    description: str


def make_three_regime_fixture() -> ThreeRegimeFixture:
    """Model with INCREASE, NEGLIGIBLE, DECREASE regimes by context.

    Uses a combination of length-2 rules that create distinct regimes.
    """
    model = _inject_model(
        n_features=2,
        partitions=[(0.2, 0.5, 0.8), (0.2, 0.5, 0.8)],
        rules=[
            (((0, "high"), (1, "low")), 2.0),     # positive in low context
            (((0, "high"), (1, "medium")), 0.0),   # zero in medium context
            (((0, "high"), (1, "high")), -2.0),    # negative in high context
            (((0, "low"), (1, "low")), -1.0),      # negative in low context (at source)
            (((0, "low"), (1, "high")), 1.0),      # positive in high context (at source)
        ],
        intercept=0.0,
    )
    dom = LinearDomain(lower=(0.0, 0.0), upper=(1.0, 1.0))
    query = TransitionQuery(
        feature_index=0,
        source_term="low",
        destination_term="high",
        source_alpha=0.5,
        destination_alpha=0.5,
        target_class=1,
        domain=dom,
    )
    bins = [
        ContextLiteral(feature_index=1, term="low", min_membership=0.5),
        ContextLiteral(feature_index=1, term="medium", min_membership=0.5),
        ContextLiteral(feature_index=1, term="high", min_membership=0.5),
    ]
    return ThreeRegimeFixture(
        model=model,
        query=query,
        grammar_bins=bins,
        description="Three direction regimes: INCREASE/NEGLIGIBLE/DECREASE by context",
    )


# ------------------------------------------------------------------ #
# Fixture 5: Infeasible transition                                   #
# ------------------------------------------------------------------ #


@dataclass
class InfeasibleTransitionFixture:
    """Query that is infeasible due to domain/alpha constraints."""

    model: SparseMaxMarginFuzzyRuleMachine
    query: TransitionQuery
    description: str


def make_infeasible_transition_fixture() -> InfeasibleTransitionFixture:
    """Infeasible: source alpha = 1.0 at a level requiring the full range,
    but the domain upper bound cuts off the required feature values."""
    model = _inject_model(
        n_features=1,
        partitions=[(0.2, 0.5, 0.8)],
        rules=[(((0, "high"),), 1.0)],
        intercept=0.0,
    )
    # Domain only allows x0 in [0.0, 0.4], but mu_high requires x0 >= q_high=0.8 for full membership
    # alpha_dest = 1.0 requires mu_high(v) >= 1.0 => v >= 0.8, but upper bound is 0.4
    dom = LinearDomain(lower=(0.0,), upper=(0.4,))
    query = TransitionQuery(
        feature_index=0,
        source_term="low",
        destination_term="high",
        source_alpha=0.5,
        destination_alpha=1.0,   # impossible in restricted domain
        target_class=1,
        domain=dom,
    )
    return InfeasibleTransitionFixture(
        model=model,
        query=query,
        description="Destination alpha=1.0 infeasible when domain upper bound < q_high",
    )


# ------------------------------------------------------------------ #
# Fixture 6: Relational tightening (interaction)                     #
# ------------------------------------------------------------------ #


@dataclass
class RelationalTighteningFixture:
    """Model where shared context makes bounds tighter than independent."""

    model: SparseMaxMarginFuzzyRuleMachine
    query: TransitionQuery
    description: str
    independent_lower: float  # independent endpoint subtraction lower bound
    independent_upper: float  # independent endpoint subtraction upper bound
    relational_lower: float   # true relational lower bound (tighter)
    relational_upper: float   # true relational upper bound (tighter)


def make_relational_tightening_fixture() -> RelationalTighteningFixture:
    """Two-antecedent rule where sharing context reduces the envelope.

    Rule: IF x0 is HIGH AND x1 is HIGH THEN +, coef=2.0

    Transition: x0 from low to high.

    Independent bound: treats source and dest contexts independently
    => destination: mu_high(x0_v) * mu_high(x1_v_dest) ranges from 0 to 1
    => source: mu_high(x0_u) * mu_high(x1_u_src) ranges from 0 to 0 (x0_u is in low region)
    The independent upper bound is 2*(1 - 0) = 2.0

    Relational bound: x1 must be the SAME at source and destination
    => x1_v = x1_u = z1
    The delta is: 2*(min(mu_high(x0_v), mu_high(z1)) - min(mu_high(x0_u), mu_high(z1)))

    When z1 is high (e.g., z1=1.0), delta = 2*(mu_high(x0_v) - mu_high(x0_u))
    With x0_v=1.0, x0_u in low region: delta = 2*(1.0 - 0.0) = 2.0  (same)

    But when z1 is low (e.g., z1=0.5), mu_high(z1)=0, so both activations are 0
    => delta = 0.0

    Actually the tightening happens when the rule has a negative sign:
    Consider: Rule: IF x0 is HIGH AND x1 is HIGH THEN -, coef=-2.0
    + another rule: IF x0 is HIGH, coef=+3.0

    Then transition is: 3*(mu_high(v)-mu_high(u)) - 2*(min(mu_high(v),mu_high(z1)) - min(mu_high(u),mu_high(z1)))

    Independent: would allow mu_high(z1_v) and mu_high(z1_u) to be different.

    For simplicity, use the context reversal fixture's structure:
    The exact relational lower/upper differ from independent subtraction.
    """
    model = _inject_model(
        n_features=2,
        partitions=[(0.2, 0.5, 0.8), (0.2, 0.5, 0.8)],
        rules=[
            (((0, "high"),), 3.0),
            (((0, "high"), (1, "high")), -2.0),
        ],
        intercept=0.0,
    )
    dom = LinearDomain(lower=(0.0, 0.0), upper=(1.0, 1.0))
    query = TransitionQuery(
        feature_index=0,
        source_term="low",
        destination_term="high",
        source_alpha=0.5,
        destination_alpha=0.5,
        target_class=1,
        domain=dom,
    )
    # Relational analysis:
    # delta = 3*(mu_high(v)-mu_high(u)) - 2*(min(mu_high(v),mu_high(z1)) - min(mu_high(u),mu_high(z1)))
    # mu_high(u) = 0 since u is in low region (mu_low(u) >= 0.5)
    # delta = 3*mu_high(v) - 2*(min(mu_high(v), mu_high(z1)) - 0)
    #       = 3*mu_high(v) - 2*min(mu_high(v), mu_high(z1))
    #
    # Case 1: mu_high(z1) >= mu_high(v):
    #   delta = 3*mu_high(v) - 2*mu_high(v) = mu_high(v) >= 0
    # Case 2: mu_high(z1) < mu_high(v):
    #   delta = 3*mu_high(v) - 2*mu_high(z1)
    #   Max when mu_high(v)=1, mu_high(z1)=0: delta = 3*(1) - 0 = 3.0
    #
    # Min relational: mu_high(v) as small as possible (= 0.5 with alpha=0.5):
    #   delta = 3*0.5 - 2*min(0.5, mu_high(z1)) = 1.5 - 2*0.5 = 0.5 (when z1 is high)
    #         or = 1.5 - 2*0 = 1.5 (when z1 is low)
    #   Min = 0.5 when mu_high(z1) >= mu_high(v)=0.5
    #
    # Independent lower bound would be:
    # Independent: min_v(3*mu_high(v)) - max_u_src(2*mu_high(u_src_for_rule))
    # (treating source and dest contexts independently)
    # This would be: 3*0.5 + 0 = 1.5 ... hmm
    # Actually independent subtraction: treats -2*phi(z,v) and +2*phi(z,u) independently
    # worst case: phi(z,v) = 1, phi(z,u) = 0 => -2*1 = -2 term
    # overall min: 3*0.5 + (-2)*1 = -0.5 < 0 (which is wrong since relational min > 0)
    return RelationalTighteningFixture(
        model=model,
        query=query,
        description="Interaction rule: relational lower > independent subtraction lower",
        independent_lower=-0.5,   # independent endpoint subtraction minimum
        independent_upper=3.0,
        relational_lower=0.5,     # true relational minimum
        relational_upper=3.0,
    )


# ------------------------------------------------------------------ #
# Fixture 7: Tiny nonzero coefficient                                #
# ------------------------------------------------------------------ #


def make_tiny_coefficient_fixture() -> OneDimTransitionFixture:
    """Rule with tiny coefficient (1e-8); tests numerical precision."""
    model = _inject_model(
        n_features=1,
        partitions=[(0.2, 0.5, 0.8)],
        rules=[(((0, "high"),), 1e-8)],
        intercept=0.0,
    )
    dom = LinearDomain(lower=(0.0,), upper=(1.0,))
    query = TransitionQuery(
        feature_index=0,
        source_term="low",
        destination_term="high",
        source_alpha=0.5,
        destination_alpha=0.5,
        target_class=1,
        domain=dom,
    )
    return OneDimTransitionFixture(
        model=model,
        query=query,
        exact_lower=5e-9,   # 1e-8 * 0.5
        exact_upper=1e-8,   # 1e-8 * 1.0
        description="Tiny coefficient: envelope is proportionally small",
    )


# ------------------------------------------------------------------ #
# Fixture 8: Same-term transition (low -> low)                        #
# ------------------------------------------------------------------ #


def make_same_term_fixture() -> OneDimTransitionFixture:
    """Low -> low transition; source and dest are the same term.

    Envelope depends on whether same-value is possible and displacement.
    Without displacement constraint, min = 0 (same point), max varies.
    """
    model = _inject_model(
        n_features=1,
        partitions=[(0.2, 0.5, 0.8)],
        rules=[(((0, "low"),), 1.0)],
        intercept=0.0,
    )
    dom = LinearDomain(lower=(0.0,), upper=(1.0,))
    query = TransitionQuery(
        feature_index=0,
        source_term="low",
        destination_term="low",
        source_alpha=0.5,
        destination_alpha=0.5,
        target_class=1,
        domain=dom,
        enforce_term_order=False,  # same term: no order
        min_raw_displacement=0.0,
    )
    # Exact analysis:
    # f_t(z, v) - f_t(z, u) = 1.0 * (mu_low(v) - mu_low(u))
    # Both u and v must satisfy mu_low >= 0.5, so mu_low(u), mu_low(v) in [0.5, 1.0].
    # (mu_low=1.0 at x<=q_low=0.2; mu_low=0.5 at x=0.35; enforce_term_order=False,
    #  min_raw_displacement=0.0 so u and v can be any points in [0, 0.35])
    # delta = mu_low(v) - mu_low(u) ranges over [0.5-1.0, 1.0-0.5] = [-0.5, 0.5].
    return OneDimTransitionFixture(
        model=model,
        query=query,
        exact_lower=-0.5,
        exact_upper=0.5,
        description="Same-term transition with no displacement; symmetric range [-0.5, 0.5]",
    )


# ------------------------------------------------------------------ #
# Fixture 9: Two-literal (checkerboard) interaction                  #
# ------------------------------------------------------------------ #


@dataclass
class CheckerboardFixture:
    """Model where single-literal clauses are always MIXED; two-literal required."""

    model: SparseMaxMarginFuzzyRuleMachine
    query: TransitionQuery
    description: str


def make_checkerboard_fixture() -> CheckerboardFixture:
    """Two-context-feature model requiring two-literal clauses.

    Features: x0 = transition (LOW -> HIGH), x1 = context, x2 = context.
    Partitions: q=(0.2, 0.5, 0.8) for all three.

    Rules (all length-2; x0 HIGH as one antecedent):
      R1: x0 HIGH, x1 LOW -> +3   (INCREASE contribution when x1 is LOW)
      R2: x0 HIGH, x2 LOW -> -3   (DECREASE contribution when x2 is LOW)

    Transition delta = 3*delta_R1 - 3*delta_R2, where:
      delta_Ri = min(mu_HIGH(v), mu_term(zi)) - min(mu_HIGH(u), mu_term(zi))

    Source (u) is in LOW region: mu_HIGH(u) ~ 0, so delta_Ri ~ min(mu_HIGH(v), mu_term(zi)).

    In each (x1, x2) grammar cell:
      (LOW, LOW):   R1 fully active + R2 fully active -> cancel exactly -> [0, 0] NEGLIGIBLE
      (LOW, HIGH):  R1 active, R2 inactive (mu_LOW(x2)=0 when x2 is HIGH) -> INCREASE
      (HIGH, LOW):  R1 inactive (mu_LOW(x1)=0 when x1 is HIGH), R2 active -> DECREASE
      (HIGH, HIGH): neither active -> [0, 0] NEGLIGIBLE

    Single-literal grammar candidates:
      x1=LOW:  covers (LOW,LOW)=NEGLIGIBLE + (LOW,HIGH)=INCREASE -> MIXED, INADMISSIBLE
      x2=LOW:  covers (LOW,LOW)=NEGLIGIBLE + (HIGH,LOW)=DECREASE -> MIXED, INADMISSIBLE
      (all other single literals also inadmissible for analogous reasons)

    => With L_max=1: GRAMMAR_INSUFFICIENT
    => With L_max=2: valid cover achievable (two-literal clauses cover exactly one atom each)
    """
    model = _inject_model(
        n_features=3,
        partitions=[(0.2, 0.5, 0.8), (0.2, 0.5, 0.8), (0.2, 0.5, 0.8)],
        rules=[
            (((0, "high"), (1, "low")), 3.0),
            (((0, "high"), (2, "low")), -3.0),
        ],
        intercept=0.0,
    )
    dom = LinearDomain(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0))
    query = TransitionQuery(
        feature_index=0,
        source_term="low",
        destination_term="high",
        source_alpha=0.5,
        destination_alpha=0.5,
        target_class=1,
        domain=dom,
    )
    return CheckerboardFixture(
        model=model,
        query=query,
        description=(
            "Two-context-feature checkerboard: single literals are always MIXED; "
            "two-literal clauses required for admissible cover"
        ),
    )


# ------------------------------------------------------------------ #
# Fixture 10: Irrelevant grammar feature                             #
# ------------------------------------------------------------------ #


@dataclass
class IrrelevantFeatureFixture:
    """Grammar includes a feature that appears in no affected rule."""

    model: SparseMaxMarginFuzzyRuleMachine
    query: TransitionQuery
    relevant_feature_index: int    # x1: appears in R1
    irrelevant_feature_index: int  # x2: absent from all nonzero rules
    description: str


def make_irrelevant_feature_fixture() -> IrrelevantFeatureFixture:
    """Three-feature model: x0 (transition), x1 (context, in rule), x2 (irrelevant).

    Rule: IF x0 is HIGH AND x1 is LOW THEN +, coef = +2.0

    Feature x2 appears in no rule, so the transition envelope is independent
    of x2. All atoms that share the same x1 bin but differ in x2 must have
    identical certified envelopes.

    Grammar: both x1 and x2 (x2 is irrelevant but declared).
    Expected: single-literal x1 clauses ARE admissible (x2 doesn't alter
    direction), minimum cover is the same as if x2 were omitted.
    """
    model = _inject_model(
        n_features=3,
        partitions=[(0.2, 0.5, 0.8), (0.2, 0.5, 0.8), (0.2, 0.5, 0.8)],
        rules=[
            (((0, "high"), (1, "low")), 2.0),
        ],
        intercept=0.0,
    )
    dom = LinearDomain(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0))
    query = TransitionQuery(
        feature_index=0,
        source_term="low",
        destination_term="high",
        source_alpha=0.5,
        destination_alpha=0.5,
        target_class=1,
        domain=dom,
    )
    return IrrelevantFeatureFixture(
        model=model,
        query=query,
        relevant_feature_index=1,
        irrelevant_feature_index=2,
        description=(
            "x2 appears in no affected rule; all (x1_bin, x2=any) atoms have "
            "identical envelopes; single-literal x1 clauses are admissible"
        ),
    )


# ------------------------------------------------------------------ #
# Helper: default grammar bins for three-term partition              #
# ------------------------------------------------------------------ #


def make_default_grammar_bins(
    feature_index: int, gamma: float = 0.5
) -> tuple[ContextLiteral, ...]:
    """Default three-bin grammar for a feature: low/medium/high at threshold gamma."""
    return (
        ContextLiteral(feature_index=feature_index, term="low", min_membership=gamma),
        ContextLiteral(feature_index=feature_index, term="medium", min_membership=gamma),
        ContextLiteral(feature_index=feature_index, term="high", min_membership=gamma),
    )


def make_default_grammar_for_model(
    model: SparseMaxMarginFuzzyRuleMachine,
    transition_feature: int,
    gamma: float = 0.5,
    max_clause_literals: int = 2,
) -> "ContextGrammar":  # type: ignore[name-defined]  # imported below
    """Build a default grammar covering all context features in affected rules."""
    from fysvm.transition_atlas import ContextGrammar

    # Context features: those in nonzero rules other than the transition feature
    ctx_feats: set[int] = set()
    for k, beta in enumerate(model.coef_):
        if beta != 0.0:
            for cond in model.rules_[k].conditions:
                if cond.feature != transition_feature:
                    ctx_feats.add(cond.feature)

    sorted_feats = sorted(ctx_feats)
    bins_by_feature = tuple(
        make_default_grammar_bins(feat, gamma=gamma) for feat in sorted_feats
    )
    return ContextGrammar(
        feature_indices=tuple(sorted_feats),
        bins_by_feature=bins_by_feature,
        max_clause_literals=max_clause_literals,
    )
