"""Membership-space sparse classifiers used as FuzzyRuleSVM ablations."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from fysvm.rule_svm import _FuzzyPartition


class MembershipSVM(ClassifierMixin, BaseEstimator):
    """Sparse linear SVM over raw low/medium/high membership features."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        penalty: str = "l1",
        partition_quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95),
        class_weight: Any = None,
        random_state: int | None = None,
        max_iter: int = 10000,
        tol: float = 1e-4,
    ) -> None:
        self.C = C
        self.penalty = penalty
        self.partition_quantiles = partition_quantiles
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MembershipSVM":
        X, y = check_X_y(X, y, dtype=np.float64)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self._fit_partitions(X)
        memberships = self._memberships(X)
        self.estimator_ = LinearSVC(
            C=self.C,
            penalty=self.penalty,
            loss="squared_hinge",
            dual=False,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        ).fit(memberships, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self.estimator_.decision_function(self._memberships(check_array(X, dtype=np.float64)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self.estimator_.predict(self._memberships(check_array(X, dtype=np.float64)))

    def nonzero_coef_count(self) -> int:
        check_is_fitted(self)
        coef = self.estimator_.coef_
        if coef.ndim == 2:
            return int(np.any(coef != 0, axis=0).sum())
        return int((coef != 0).sum())

    def _fit_partitions(self, X: np.ndarray) -> None:
        quantiles = np.quantile(X, self.partition_quantiles, axis=0)
        self.partitions_ = [
            _FuzzyPartition(
                low=float(quantiles[0, j]),
                medium=float(quantiles[1, j]),
                high=float(quantiles[2, j]),
            )
            for j in range(X.shape[1])
        ]

    def _memberships(self, X: np.ndarray) -> np.ndarray:
        return np.clip(
            np.hstack([partition.transform(X[:, j]) for j, partition in enumerate(self.partitions_)]),
            0.0,
            1.0,
        )


class MembershipLogisticL1(MembershipSVM):
    """L1-logistic regression over the same raw membership features."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        partition_quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95),
        class_weight: Any = None,
        random_state: int | None = None,
        max_iter: int = 5000,
        tol: float = 1e-3,
    ) -> None:
        super().__init__(
            C=C,
            penalty="l1",
            partition_quantiles=partition_quantiles,
            class_weight=class_weight,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MembershipLogisticL1":
        X, y = check_X_y(X, y, dtype=np.float64)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self._fit_partitions(X)
        memberships = self._memberships(X)
        self.estimator_ = LogisticRegression(
            C=self.C,
            penalty="l1",
            solver="saga",
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        ).fit(memberships, y)
        return self


def membership_nonzero_coef_count(model: Any) -> float:
    """Count active membership coefficients for binary or OvR wrappers."""

    if hasattr(model, "estimators_"):
        masks = []
        for estimator in model.estimators_:
            inner = getattr(estimator, "estimator_", None)
            if inner is None or not hasattr(inner, "coef_"):
                continue
            masks.append(np.any(inner.coef_ != 0, axis=0))
        return float(np.any(np.vstack(masks), axis=0).sum()) if masks else 0.0
    if hasattr(model, "nonzero_coef_count"):
        return float(model.nonzero_coef_count())
    return 0.0
