"""Directional reasoning in the fitted unary/pairwise product grammar."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from math import isfinite

import numpy as np
from sklearn.utils.validation import check_is_fitted

from fysvm.rule_svm import FuzzyRuleSVM, _TERM_NAMES, _TERM_TO_INDEX

DEFAULT_TOLERANCE = Fraction(1, 10**9)


def _rational(value: float) -> Fraction:
    value = float(value)
    if not isfinite(value):
        raise ValueError("Inputs, anchors and coefficients must be finite.")
    return Fraction(value)


@dataclass(frozen=True)
class DirectionalCondition:
    """A fitted half-cut in input units. None denotes an unbounded side."""

    feature: int
    name: str
    term: str
    lower: Fraction | None
    upper: Fraction | None


@dataclass(frozen=True)
class DirectionalCertificate:
    """Result of a directional query, with status certified, zero or unresolved.

    Direction is +1 for increasing margin, -1 for decreasing margin, and 0
    for a case effect within tolerance. An empty certified context applies to
    all backgrounds. The bound is a positive magnitude for either direction.
    For an unresolved or zero result, bound is None. available_bound is the
    tight bound with all available conditions applied when direction is nonzero.
    """

    feature: int
    feature_name: str
    transition: int
    segment: tuple[Fraction, Fraction]
    direction: int
    point_effect: Fraction
    status: str
    conditions: tuple[DirectionalCondition, ...]
    bound: Fraction | None
    available_bound: Fraction | None


@dataclass(frozen=True)
class _Partition:
    anchors: tuple[Fraction, Fraction, Fraction]

    def memberships(self, value):
        low, medium, high = self.anchors
        if value <= low:
            return (1, 0, 0)
        if value < medium:
            t = (value - low) / (medium - low)
            return (1 - t, t, 0)
        if value == medium:
            return (0, 1, 0)
        if value < high:
            t = (value - medium) / (high - medium)
            return (0, 1 - t, t)
        return (0, 0, 1)

    def interval(self, term):
        low, medium, high = self.anchors
        left = (low + medium) / 2
        right = (medium + high) / 2
        return ((None, left), (left, right), (right, None))[term]


@dataclass(frozen=True)
class _Effect:
    constant: Fraction
    neighbors: tuple[tuple[int, tuple[Fraction, Fraction, Fraction]], ...]

    def point(self, memberships):
        return self.constant + sum(
            sum(weight * degree for weight, degree in zip(weights, memberships[k]) if degree)
            for k, weights in self.neighbors
        )


def _signed_minima(weights, sign):
    low, medium, high = (sign * w for w in weights)
    left, right = (low + medium) / 2, (medium + high) / 2
    return min(low, medium, high), (min(low, left), min(left, medium, right), min(right, high))


class DirectionalExplainer:
    """Snapshot a fitted binary FuzzyRuleSVM and query its score, in model input units.

    Certifies the mathematical piecewise-linear interpolant defined by the
    stored binary64 anchors and weights. Supply cases in the units used to fit
    the model. The direction is the sign of the exact endpoint difference at
    the case, with magnitudes at or below tolerance classified as zero.
    """

    def __init__(
        self,
        model: FuzzyRuleSVM,
        tolerance: Fraction | float = DEFAULT_TOLERANCE,
    ) -> None:
        check_is_fitted(model, ("coef_", "rules_", "partitions_", "selected_feature_indices_"))
        if model.and_operator != "product":
            raise ValueError("Directional certificates require product conjunctions.")
        self.tolerance = Fraction(tolerance)
        if self.tolerance < 0:
            raise ValueError("Tolerance must be nonnegative.")
        self.n_features = model.n_features_in_
        self.features = tuple(int(i) for i in model.selected_feature_indices_)
        self.names = dict(zip(self.features, model.feature_names_in_, strict=True))
        self._partitions = {
            feature: _Partition(tuple(_rational(a) for a in (p.low, p.medium, p.high)))
            for feature, p in zip(self.features, model.partitions_, strict=True)
        }
        if any(not a < b < c for a, b, c in (p.anchors for p in self._partitions.values())):
            raise ValueError("Directional certificates require three strictly ordered anchors.")

        unary = {feature: [Fraction(0)] * 3 for feature in self.features}
        pairs = {feature: {} for feature in self.features}
        for rule, coefficient in zip(model.rules_, model.coef_, strict=True):
            conditions = tuple(
                (self.features[condition.feature], _TERM_TO_INDEX[condition.term])
                for condition in rule.conditions
            )
            if len(conditions) not in (1, 2) or len({k for k, _ in conditions}) != len(conditions):
                raise ValueError("Every rule must contain one or two distinct features.")
            weight = _rational(coefficient)
            if not weight:
                continue
            feature, term = conditions[0]
            if len(conditions) == 1:
                unary[feature][term] += weight
            else:
                neighbor, other_term = conditions[1]
                # Each feature can be queried, so store both orders of the pair.
                for j, t, k, u in (
                    (feature, term, neighbor, other_term),
                    (neighbor, other_term, feature, term),
                ):
                    table = pairs[j].setdefault(k, [[Fraction(0)] * 3 for _ in _TERM_NAMES])
                    table[t][u] += weight

        self._effects = {}
        self._minima = {}
        for feature in self.features:
            for step in (0, 1):
                constant = unary[feature][step + 1] - unary[feature][step]
                neighbors = []
                for neighbor, table in sorted(pairs[feature].items()):
                    delta = tuple(b - a for a, b in zip(table[step], table[step + 1]))
                    if delta[0] == delta[1] == delta[2]:
                        constant += delta[0]
                    else:
                        neighbors.append((neighbor, delta))
                key = (feature, step)
                self._effects[key] = _Effect(constant, tuple(neighbors))
                for sign in (-1, 1):
                    self._minima[key, sign] = tuple(
                        (k, *_signed_minima(weights, sign)) for k, weights in neighbors
                    )

    def involved_features(self, feature: int, transition: int) -> tuple[int, ...]:
        """Indices whose values can change the adjacent-anchor difference."""
        return tuple(k for k, _ in self._effects[feature, transition].neighbors)

    def explain(
        self, case: Sequence[float] | np.ndarray, feature: int, transition: int
    ) -> DirectionalCertificate:
        """Explain a transition using an original input column index.

        Transition 0 is low to medium, and transition 1 is medium to high.
        """
        return self.explain_case(case, [(feature, transition)])[0]

    def explain_case(
        self,
        case: Sequence[float] | np.ndarray,
        queries: Iterable[tuple[int, int]] | None = None,
    ) -> list[DirectionalCertificate]:
        """Explain requested transitions, or both transitions of every fitted feature."""
        case = np.asarray(case, dtype=float)
        if case.shape != (self.n_features,) or not np.all(np.isfinite(case)):
            raise ValueError("Supply one finite case in the original model input column order.")
        if queries is None:
            queries = self._effects
        queries = tuple(queries)
        if any(query not in self._effects for query in queries):
            raise ValueError(
                "Each query must target a fitted feature and an adjacent transition (0 or 1)."
            )
        needed = set(chain.from_iterable(self.involved_features(*q) for q in queries))
        memberships = {k: self._partitions[k].memberships(_rational(case[k])) for k in needed}
        winning = {k: max(range(3), key=m.__getitem__) for k, m in memberships.items()}
        return [self._explain(query, memberships, winning) for query in queries]

    def _explain(self, query, memberships, winning):
        feature, step = query
        effect = self._effects[query]
        anchors = self._partitions[feature].anchors
        base = (feature, str(self.names[feature]), step, (anchors[step], anchors[step + 1]))
        point = effect.point(memberships)
        if abs(point) <= self.tolerance:
            return DirectionalCertificate(*base, 0, point, "zero", (), None, None)

        sign = 1 if point > 0 else -1
        parts = self._minima[query, sign]
        bound = sign * effect.constant + sum(unrestricted for _, unrestricted, _ in parts)
        gains = sorted(
            ((restricted[winning[k]] - unrestricted, k) for k, unrestricted, restricted in parts),
            key=lambda item: (-item[0], item[1]),
        )
        available = bound + sum(gain for gain, _ in gains)
        if available <= self.tolerance:
            return DirectionalCertificate(*base, sign, point, "unresolved", (), None, available)

        conditions = []
        for gain, k in gains:
            if bound > self.tolerance:
                break
            term = winning[k]
            conditions.append(
                DirectionalCondition(
                    k, str(self.names[k]), _TERM_NAMES[term], *self._partitions[k].interval(term)
                )
            )
            bound += gain
        return DirectionalCertificate(*base, sign, point, "certified", tuple(conditions), bound, available)
