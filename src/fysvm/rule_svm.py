"""Sparse max-margin classifier over fuzzy rule activations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


AndOperator = Literal["min", "product", "softmin"]
Penalty = Literal["l1", "l2"]


@dataclass(frozen=True)
class RuleCondition:
    """One linguistic antecedent in a fuzzy rule."""

    feature: int
    term: str


@dataclass(frozen=True)
class FuzzyRule:
    """A fuzzy rule antecedent represented as feature-term conditions."""

    conditions: tuple[RuleCondition, ...]

    @property
    def length(self) -> int:
        """Number of antecedents in the rule."""

        return len(self.conditions)


@dataclass(frozen=True)
class _FuzzyPartition:
    """Low/medium/high membership partition for one numeric feature."""

    low: float
    medium: float
    high: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        low = _linear_down(values, self.low, self.medium)
        medium = np.minimum(
            _linear_up(values, self.low, self.medium),
            _linear_down(values, self.medium, self.high),
        )
        high = _linear_up(values, self.medium, self.high)
        return np.column_stack((low, medium, high))


class SparseMaxMarginFuzzyRuleMachine(ClassifierMixin, BaseEstimator):
    """Linear max-margin classifier in fuzzy-rule activation space.

    The estimator builds linguistic feature concepts (`low`, `medium`, `high`),
    generates candidate conjunction rules, maps each sample into fuzzy rule
    activations, and trains a sparse linear SVM on that activation matrix. The
    resulting decision function is directly explainable:

    ``f(x) = sum_k beta_k * phi_k(x) + b``

    where every ``phi_k`` is a readable fuzzy rule firing strength.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        penalty: Penalty = "l1",
        max_rule_length: int = 2,
        max_rules: int | None = 256,
        min_rule_coverage: float = 0.02,
        and_operator: AndOperator = "min",
        softmin_temperature: float = 0.1,
        partition_quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95),
        rule_length_penalty: float = 0.25,
        feature_names: Sequence[str] | None = None,
        class_weight: dict[Any, float] | Literal["balanced"] | None = None,
        random_state: int | None = None,
        max_iter: int = 10000,
        tol: float = 1e-4,
    ) -> None:
        self.C = C
        self.penalty = penalty
        self.max_rule_length = max_rule_length
        self.max_rules = max_rules
        self.min_rule_coverage = min_rule_coverage
        self.and_operator = and_operator
        self.softmin_temperature = softmin_temperature
        self.partition_quantiles = partition_quantiles
        self.rule_length_penalty = rule_length_penalty
        self.feature_names = feature_names
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "SparseMaxMarginFuzzyRuleMachine":
        """Fit the fuzzy rule-space max-margin classifier.

        Parameters
        ----------
        X:
            Numeric input samples with shape ``(n_samples, n_features)``.
        y:
            Binary class labels.
        sample_weight:
            Optional fuzzy sample reliability scores. Larger values make margin
            violations more costly for the corresponding samples.
        """

        self._validate_parameters()
        raw_feature_names = self._resolve_feature_names(X)
        X_checked, y_checked = check_X_y(X, y, dtype=np.float64)
        classes = unique_labels(y_checked)
        if len(classes) != 2:
            raise ValueError(
                "SparseMaxMarginFuzzyRuleMachine currently supports binary "
                f"classification; got {len(classes)} classes."
            )

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape != (X_checked.shape[0],):
                raise ValueError(
                    "sample_weight must have shape (n_samples,), got "
                    f"{sample_weight.shape}."
                )
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight values must be non-negative.")

        self.classes_ = classes
        self.n_features_in_ = X_checked.shape[1]
        self.feature_names_in_ = self._finalize_feature_names(
            raw_feature_names, self.n_features_in_
        )

        self.partitions_ = self._fit_partitions(X_checked)
        memberships = self._concept_membership_tensor(X_checked)
        y_signed = self._signed_labels(y_checked)
        rule_weights = sample_weight if sample_weight is not None else np.ones_like(
            y_signed, dtype=np.float64
        )

        self.rules_ = self._generate_rules(memberships, y_signed, rule_weights)
        Z = self._rule_activation_matrix_from_memberships(memberships, self.rules_)
        self.rule_penalties_ = np.array(
            [1.0 + self.rule_length_penalty * (rule.length - 1) for rule in self.rules_],
            dtype=np.float64,
        )
        Z_scaled = Z / self.rule_penalties_

        model = LinearSVC(
            C=self.C,
            penalty=self.penalty,
            loss="squared_hinge",
            dual=False,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        model.fit(Z_scaled, y_checked, sample_weight=sample_weight)

        self.linear_svm_ = model
        scaled_coef = model.coef_.reshape(-1)
        self.coef_ = scaled_coef / self.rule_penalties_
        self.intercept_ = float(model.intercept_[0])
        self.n_rules_ = len(self.rules_)
        self.active_rule_indices_ = np.flatnonzero(np.abs(self.coef_) > 1e-12)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the fuzzy rule activation matrix ``Phi_F(X)``."""

        check_is_fitted(self)
        X_checked = self._check_X_for_prediction(X)
        memberships = self._concept_membership_tensor(X_checked)
        return self._rule_activation_matrix_from_memberships(memberships, self.rules_)

    def concept_memberships(self, X: np.ndarray) -> list[dict[str, dict[str, float]]]:
        """Return low/medium/high memberships for each feature and sample."""

        check_is_fitted(self)
        X_checked = self._check_X_for_prediction(X)
        memberships = self._concept_membership_tensor(X_checked)
        concepts: list[dict[str, dict[str, float]]] = []
        for sample_memberships in memberships:
            sample: dict[str, dict[str, float]] = {}
            for feature_index, feature_name in enumerate(self.feature_names_in_):
                sample[feature_name] = {
                    term: float(sample_memberships[feature_index, term_index])
                    for term_index, term in enumerate(_TERM_NAMES)
                }
            concepts.append(sample)
        return concepts

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return signed fuzzy margins for samples in ``X``."""

        check_is_fitted(self)
        Z = self.transform(X)
        return Z @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels from fuzzy rule-space margins."""

        margins = self.decision_function(X)
        return np.where(margins >= 0.0, self.classes_[1], self.classes_[0])

    def explain(
        self,
        X: np.ndarray,
        *,
        top_n: int = 5,
        min_abs_contribution: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Explain predictions as native fuzzy-rule contributions.

        Each explanation contains the bias, net margin, predicted label, and the
        largest positive and negative rule contributions. The contribution values
        sum with the bias to the reported margin.
        """

        check_is_fitted(self)
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        if min_abs_contribution < 0:
            raise ValueError("min_abs_contribution must be non-negative.")

        Z = self.transform(X)
        margins = Z @ self.coef_ + self.intercept_
        predictions = np.where(margins >= 0.0, self.classes_[1], self.classes_[0])

        explanations: list[dict[str, Any]] = []
        for row, margin, prediction in zip(Z, margins, predictions, strict=True):
            contributions = row * self.coef_
            positive_indices = np.argsort(contributions)[::-1]
            negative_indices = np.argsort(contributions)

            top_positive = self._contribution_items(
                row,
                contributions,
                positive_indices,
                top_n,
                min_abs_contribution,
                positive=True,
            )
            top_negative = self._contribution_items(
                row,
                contributions,
                negative_indices,
                top_n,
                min_abs_contribution,
                positive=False,
            )
            strongest = self._contribution_items(
                row,
                contributions,
                np.argsort(np.abs(contributions))[::-1],
                top_n,
                min_abs_contribution,
                positive=None,
            )

            explanations.append(
                {
                    "prediction": prediction,
                    "margin": float(margin),
                    "bias": self.intercept_,
                    "net_rule_contribution": float(np.sum(contributions)),
                    "top_positive_rules": top_positive,
                    "top_negative_rules": top_negative,
                    "top_rules": strongest,
                }
            )
        return explanations

    def support_rules(self, *, min_abs_weight: float = 1e-8) -> list[dict[str, Any]]:
        """Return learned support rules sorted by absolute coefficient size."""

        check_is_fitted(self)
        if min_abs_weight < 0:
            raise ValueError("min_abs_weight must be non-negative.")
        indices = np.flatnonzero(np.abs(self.coef_) >= min_abs_weight)
        ordered = indices[np.argsort(np.abs(self.coef_[indices]))[::-1]]
        return [self._rule_weight_item(int(index)) for index in ordered]

    def fuzzy_violations(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Return slack values and linguistic violation memberships.

        The numeric slack is ``xi_i = max(0, 1 - y_i f(x_i))``. It is exposed as
        fuzzy categories: `cleanly_classified`, `borderline`, and
        `strong_violation`.
        """

        check_is_fitted(self)
        y_array = np.asarray(y)
        margins = self.decision_function(X)
        if y_array.shape != margins.shape:
            raise ValueError(
                f"y must have shape {margins.shape} for the provided X, got {y_array.shape}."
            )
        signed = self._signed_labels(y_array)
        functional_margins = signed * margins
        slack = np.maximum(0.0, 1.0 - functional_margins)

        results: list[dict[str, Any]] = []
        for margin, functional_margin, xi in zip(
            margins, functional_margins, slack, strict=True
        ):
            results.append(
                {
                    "margin": float(margin),
                    "functional_margin": float(functional_margin),
                    "slack": float(xi),
                    "memberships": {
                        "cleanly_classified": float(_linear_down(np.array([xi]), 0.0, 0.5)[0]),
                        "borderline": float(
                            min(
                                _linear_up(np.array([xi]), 0.0, 1.0)[0],
                                _linear_down(np.array([xi]), 1.0, 2.0)[0],
                            )
                        ),
                        "strong_violation": float(_linear_up(np.array([xi]), 1.0, 2.0)[0]),
                    },
                }
            )
        return results

    def rule_to_string(
        self,
        rule: FuzzyRule,
        *,
        consequent: Any | None = None,
    ) -> str:
        """Render a fuzzy rule as a human-readable string."""

        check_is_fitted(self)
        antecedent = " AND ".join(
            f"{self.feature_names_in_[condition.feature]} is {condition.term}"
            for condition in rule.conditions
        )
        if consequent is None:
            return f"IF {antecedent}"
        return f"IF {antecedent} THEN {consequent}"

    def _validate_parameters(self) -> None:
        if self.C <= 0:
            raise ValueError("C must be positive.")
        if self.penalty not in {"l1", "l2"}:
            raise ValueError("penalty must be either 'l1' or 'l2'.")
        if self.max_rule_length < 1:
            raise ValueError("max_rule_length must be at least 1.")
        if self.max_rules is not None and self.max_rules < 1:
            raise ValueError("max_rules must be None or at least 1.")
        if not 0.0 <= self.min_rule_coverage <= 1.0:
            raise ValueError("min_rule_coverage must be between 0 and 1.")
        if self.and_operator not in {"min", "product", "softmin"}:
            raise ValueError("and_operator must be 'min', 'product', or 'softmin'.")
        if self.softmin_temperature <= 0:
            raise ValueError("softmin_temperature must be positive.")
        if len(self.partition_quantiles) != 3:
            raise ValueError("partition_quantiles must contain three quantiles.")
        q_low, q_mid, q_high = self.partition_quantiles
        if not (0.0 <= q_low <= q_mid <= q_high <= 1.0):
            raise ValueError(
                "partition_quantiles must satisfy 0 <= low <= medium <= high <= 1."
            )
        if self.rule_length_penalty < 0:
            raise ValueError("rule_length_penalty must be non-negative.")

    def _fit_partitions(self, X: np.ndarray) -> list[_FuzzyPartition]:
        quantiles = np.quantile(X, self.partition_quantiles, axis=0)
        return [
            _FuzzyPartition(
                low=float(quantiles[0, feature]),
                medium=float(quantiles[1, feature]),
                high=float(quantiles[2, feature]),
            )
            for feature in range(X.shape[1])
        ]

    def _concept_membership_tensor(self, X: np.ndarray) -> np.ndarray:
        tensor = np.empty((X.shape[0], X.shape[1], len(_TERM_NAMES)), dtype=np.float64)
        for feature, partition in enumerate(self.partitions_):
            tensor[:, feature, :] = partition.transform(X[:, feature])
        return np.clip(tensor, 0.0, 1.0)

    def _generate_rules(
        self,
        memberships: np.ndarray,
        y_signed: np.ndarray,
        sample_weight: np.ndarray,
    ) -> list[FuzzyRule]:
        candidates: list[tuple[float, float, int, FuzzyRule]] = []
        n_features = memberships.shape[1]
        max_length = min(self.max_rule_length, n_features)
        for length in range(1, max_length + 1):
            for feature_indices in combinations(range(n_features), length):
                for term_indices in product(range(len(_TERM_NAMES)), repeat=length):
                    rule = FuzzyRule(
                        tuple(
                            RuleCondition(feature, _TERM_NAMES[term_index])
                            for feature, term_index in zip(
                                feature_indices, term_indices, strict=True
                            )
                        )
                    )
                    activation = self._combine_memberships(
                        memberships, feature_indices, term_indices
                    )
                    fuzzy_coverage = float(np.average(activation, weights=sample_weight))
                    if fuzzy_coverage < self.min_rule_coverage:
                        continue
                    score = self._rule_association_score(
                        activation, y_signed, sample_weight, length
                    )
                    candidates.append((score, fuzzy_coverage, -length, rule))

        if not candidates:
            candidates = self._fallback_single_feature_rules(memberships, y_signed, sample_weight)

        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        if self.max_rules is not None:
            candidates = candidates[: self.max_rules]
        return [rule for _, _, _, rule in candidates]

    def _fallback_single_feature_rules(
        self,
        memberships: np.ndarray,
        y_signed: np.ndarray,
        sample_weight: np.ndarray,
    ) -> list[tuple[float, float, int, FuzzyRule]]:
        candidates: list[tuple[float, float, int, FuzzyRule]] = []
        for feature in range(memberships.shape[1]):
            for term_index, term in enumerate(_TERM_NAMES):
                rule = FuzzyRule((RuleCondition(feature, term),))
                activation = memberships[:, feature, term_index]
                fuzzy_coverage = float(np.average(activation, weights=sample_weight))
                score = self._rule_association_score(activation, y_signed, sample_weight, 1)
                candidates.append((score, fuzzy_coverage, -1, rule))
        return candidates

    def _rule_association_score(
        self,
        activation: np.ndarray,
        y_signed: np.ndarray,
        sample_weight: np.ndarray,
        length: int,
    ) -> float:
        positive = y_signed > 0
        negative = ~positive
        positive_strength = _safe_weighted_average(
            activation[positive], sample_weight[positive]
        )
        negative_strength = _safe_weighted_average(
            activation[negative], sample_weight[negative]
        )
        coverage = _safe_weighted_average(activation, sample_weight)
        return float(abs(positive_strength - negative_strength) * coverage / length)

    def _rule_activation_matrix_from_memberships(
        self,
        memberships: np.ndarray,
        rules: Sequence[FuzzyRule],
    ) -> np.ndarray:
        Z = np.empty((memberships.shape[0], len(rules)), dtype=np.float64)
        for index, rule in enumerate(rules):
            feature_indices = tuple(condition.feature for condition in rule.conditions)
            term_indices = tuple(_TERM_TO_INDEX[condition.term] for condition in rule.conditions)
            Z[:, index] = self._combine_memberships(memberships, feature_indices, term_indices)
        return Z

    def _combine_memberships(
        self,
        memberships: np.ndarray,
        feature_indices: Iterable[int],
        term_indices: Iterable[int],
    ) -> np.ndarray:
        feature_indices = tuple(feature_indices)
        term_indices = tuple(term_indices)
        values = np.column_stack(
            [memberships[:, feature, term] for feature, term in zip(feature_indices, term_indices, strict=True)]
        )
        if values.shape[1] == 1:
            return values[:, 0]
        if self.and_operator == "min":
            return np.min(values, axis=1)
        if self.and_operator == "product":
            return np.prod(values, axis=1)
        return _softmin(values, self.softmin_temperature)

    def _resolve_feature_names(self, X: Any) -> Sequence[str] | None:
        if self.feature_names is not None:
            return self.feature_names
        columns = getattr(X, "columns", None)
        if columns is not None:
            return [str(column) for column in columns]
        return None

    def _finalize_feature_names(
        self,
        raw_feature_names: Sequence[str] | None,
        n_features: int,
    ) -> np.ndarray:
        if raw_feature_names is None:
            return np.array([f"x{index}" for index in range(n_features)], dtype=object)
        names = np.asarray(list(raw_feature_names), dtype=object)
        if names.shape != (n_features,):
            raise ValueError(
                "feature_names must have one entry per feature; got "
                f"{names.shape[0]} names for {n_features} features."
            )
        return names

    def _check_X_for_prediction(self, X: np.ndarray) -> np.ndarray:
        X_checked = check_array(X, dtype=np.float64)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but this estimator was fit "
                f"with {self.n_features_in_} features."
            )
        return X_checked

    def _signed_labels(self, y: np.ndarray) -> np.ndarray:
        y_array = np.asarray(y)
        signed = np.empty(y_array.shape[0], dtype=np.float64)
        positive = y_array == self.classes_[1]
        negative = y_array == self.classes_[0]
        unknown = ~(positive | negative)
        if np.any(unknown):
            unknown_labels = np.unique(y_array[unknown])
            raise ValueError(f"Unknown labels for fitted classes: {unknown_labels!r}")
        signed[positive] = 1.0
        signed[negative] = -1.0
        return signed

    def _contribution_items(
        self,
        firing: np.ndarray,
        contributions: np.ndarray,
        ordered_indices: np.ndarray,
        top_n: int,
        min_abs_contribution: float,
        *,
        positive: bool | None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for index in ordered_indices:
            contribution = contributions[index]
            if abs(contribution) < min_abs_contribution:
                continue
            if positive is True and contribution <= 0:
                continue
            if positive is False and contribution >= 0:
                continue
            items.append(self._rule_contribution_item(int(index), firing[index], contribution))
            if len(items) == top_n:
                break
        return items

    def _rule_contribution_item(
        self,
        index: int,
        firing: float,
        contribution: float,
    ) -> dict[str, Any]:
        weight = float(self.coef_[index])
        support_class = self.classes_[1] if weight >= 0 else self.classes_[0]
        return {
            "rule_index": index,
            "rule": self.rule_to_string(self.rules_[index], consequent=support_class),
            "conditions": [
                {
                    "feature": self.feature_names_in_[condition.feature],
                    "term": condition.term,
                }
                for condition in self.rules_[index].conditions
            ],
            "firing": float(firing),
            "weight": weight,
            "contribution": float(contribution),
            "supports": support_class,
        }

    def _rule_weight_item(self, index: int) -> dict[str, Any]:
        weight = float(self.coef_[index])
        support_class = self.classes_[1] if weight >= 0 else self.classes_[0]
        return {
            "rule_index": index,
            "rule": self.rule_to_string(self.rules_[index], consequent=support_class),
            "weight": weight,
            "length": self.rules_[index].length,
            "supports": support_class,
        }


FuzzyRuleSVM = SparseMaxMarginFuzzyRuleMachine

_TERM_NAMES = ("low", "medium", "high")
_TERM_TO_INDEX = {term: index for index, term in enumerate(_TERM_NAMES)}


def _linear_down(values: np.ndarray, start: float, end: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if end <= start:
        return (values <= start).astype(np.float64)
    result = (end - values) / (end - start)
    result = np.where(values <= start, 1.0, result)
    result = np.where(values >= end, 0.0, result)
    return np.clip(result, 0.0, 1.0)


def _linear_up(values: np.ndarray, start: float, end: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if end <= start:
        return (values >= end).astype(np.float64)
    result = (values - start) / (end - start)
    result = np.where(values <= start, 0.0, result)
    result = np.where(values >= end, 1.0, result)
    return np.clip(result, 0.0, 1.0)


def _safe_weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return 0.0
    return float(np.sum(values * weights) / weight_sum)


def _softmin(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = -values / temperature
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    log_mean_exp = np.log(np.mean(np.exp(scaled - max_scaled), axis=1))
    result = -temperature * (log_mean_exp + max_scaled[:, 0])
    return np.clip(result, 0.0, 1.0)
