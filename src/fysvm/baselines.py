"""Sklearn-compatible wrappers for FURIA, FARC-HD, and IVTURS baselines.

Sources
-------
- FURIA: simplified Python reimplementation from asieriko/ex_fuzzy_farchd
  (Hühn & Hüllermeier 2009 - original paper)
- FARC-HD: asieriko/ex_fuzzy_farchd with ex-fuzzy T1 partitions
  (Alcalá-Fdez, Alcalá & Herrera 2011 - original paper)
- IVTURS: FARC-HD framework with interval-valued type-2 (T2) ex-fuzzy partitions
  (Sanz, Bustince & Herrera 2013 - original paper; Java reference: JoseanSanz/IVTURS)

All wrappers:
  - Are sklearn-compatible (BaseEstimator, ClassifierMixin)
  - Handle label encoding/decoding for string class labels
  - Are safe to clone and use in GridSearchCV / StratifiedKFold
  - Create fuzzy partitions from training data inside fit()
"""

from __future__ import annotations

import importlib
import sys
import types
import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


# ---------------------------------------------------------------------------
# FURIA
# ---------------------------------------------------------------------------

class FURIAClassifier(BaseEstimator, ClassifierMixin):
    """FURIA – Fuzzy Unordered Rule Induction Algorithm.

    A Python implementation from `asieriko/ex_fuzzy_farchd`, based on:
    Hühn J., Hüllermeier E. (2009) FURIA: An Algorithm for Unordered Fuzzy
    Rule Induction. *Data Mining and Knowledge Discovery*, 19(3), 293–319.

    Parameters
    ----------
    n_folds : int, default=3
        Number of folds for the growing/pruning split inside each class.
    min_no : float, default=2.0
        Minimum total weight of instances a rule must cover.
    n_optimizations : int, default=2
        Number of simplification/optimization passes.
    check_error_rate : bool, default=True
        Stop growing a rule if its error rate is ≥ 0.5.
    use_rule_stretching : bool, default=True
        Apply rule stretching for uncovered test instances.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_folds: int = 3,
        min_no: float = 2.0,
        n_optimizations: int = 2,
        check_error_rate: bool = True,
        use_rule_stretching: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.n_folds = n_folds
        self.min_no = min_no
        self.n_optimizations = n_optimizations
        self.check_error_rate = check_error_rate
        self.use_rule_stretching = use_rule_stretching
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FURIAClassifier":
        """Fit FURIA to training data."""
        X, y = check_X_y(X, y)
        from ex_fuzzy_farchd.furia_classifier import FURIA

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]

        self._clf = FURIA(
            n_folds=self.n_folds,
            min_no=self.min_no,
            n_optimizations=self.n_optimizations,
            check_error_rate=self.check_error_rate,
            use_rule_stretching=self.use_rule_stretching,
            random_state=self.random_state,
        )
        self._clf.fit(X, y_enc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        y_enc_pred = self._clf.predict(X)
        return self.le_.inverse_transform(y_enc_pred.astype(int))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        return self._clf.predict_proba(X)

    @property
    def ruleset_(self):
        """Convenience access to the internal rule set after fitting."""
        check_is_fitted(self)
        return self._clf.ruleset_


# ---------------------------------------------------------------------------
# FARC-HD (type-1 fuzzy partitions)
# ---------------------------------------------------------------------------

class FARCHDClassifier(BaseEstimator, ClassifierMixin):
    """FARC-HD – Fuzzy Association Rule-Based Classification for High-Dimensional problems.

    Python port from `asieriko/ex_fuzzy_farchd`, based on:
    Alcalá-Fdez J., Alcalá R., Herrera F. (2011). A Fuzzy Association
    Rule-Based Classification Model for High-Dimensional Problems with Genetic
    Rule Selection and Lateral Tuning. *IEEE Transactions on Fuzzy Systems*,
    19(5), 857–872.

    Parameters
    ----------
    n_labels : int, default=5
        Number of fuzzy linguistic labels per feature.
    max_depth : int, default=2
        Maximum number of antecedents per rule (≤ 3 for feasibility).
    min_support : float, default=0.05
        Minimum fuzzy support for Apriori rule extraction.
    maxconf : float, default=0.8
        Maximum confidence threshold; rules above this are promoted directly.
    kt : int, default=2
        Minimum pattern coverage count for Stage 2 prescreening.
    pop : int, default=50
        Genetic algorithm population size (Stage 3).
    evaluations : int, default=200
        Genetic algorithm evaluation budget (Stage 3).
    bitsgene : int, default=30
        Bits per gene in Gray-coded lateral tuning.
    delta : float, default=0.2
        Complexity penalty weight in the fitness function.
    fuzzy_type : {'t1', 't2'}, default='t1'
        Fuzzy partition type. Use 't2' for interval-valued type-2 (≈ IVTURS).
    random_state : int or None, default=None
        Random seed (currently passed to numpy before Stage 3; FARCHD itself
        does not expose a full random-state interface).
    """

    def __init__(
        self,
        n_labels: int = 5,
        max_depth: int = 2,
        min_support: float = 0.05,
        maxconf: float = 0.8,
        kt: int = 2,
        pop: int = 50,
        evaluations: int = 200,
        bitsgene: int = 30,
        delta: float = 0.2,
        fuzzy_type: str = "t1",
        random_state: int | None = None,
    ) -> None:
        self.n_labels = n_labels
        self.max_depth = max_depth
        self.min_support = min_support
        self.maxconf = maxconf
        self.kt = kt
        self.pop = pop
        self.evaluations = evaluations
        self.bitsgene = bitsgene
        self.delta = delta
        self.fuzzy_type = fuzzy_type
        self.random_state = random_state

    def _make_partitions(self, X: np.ndarray) -> list:
        """Build ex-fuzzy fuzzy partitions from training data."""
        import ex_fuzzy.utils as utils

        if self.fuzzy_type == "t2":
            partitions = utils.t2_fuzzy_partitions_dataset(X, n_partition=self.n_labels)
        else:
            partitions = utils.t1_fuzzy_partitions_dataset(
                X, n_partition=self.n_labels, shape="trapezoid"
            )
        return partitions

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FARCHDClassifier":
        """Fit FARC-HD to training data."""
        _install_ex_fuzzy_farchd_src_alias()
        from ex_fuzzy_farchd.FARCHD import FARCHD

        X, y = check_X_y(X, y)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]

        classes_dict = {int(c): int(i) for i, c in enumerate(np.unique(y_enc))}
        partitions = self._make_partitions(X)

        self._clf = FARCHD(
            partitions=partitions,
            classes_dict=classes_dict,
            n_labels=self.n_labels,
            max_depth=self.max_depth,
            min_support=self.min_support,
            maxconf=self.maxconf,
            kt=self.kt,
            pop=self.pop,
            evaluations=self.evaluations,
            BITSGENE=self.bitsgene,
            delta=self.delta,
        )
        import io
        import sys
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                self._clf.fit(X, y_enc)
            finally:
                sys.stdout = _old_stdout

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        if self._clf.rules is None:
            # Fallback if no rules were learned
            return np.full(X.shape[0], self.classes_[0])
        raw = self._clf.predict(X)
        # raw contains integer indices (possibly as float); convert to int labels
        idx = np.round(np.asarray(raw)).astype(int)
        # map back through classes_
        idx = np.clip(idx, 0, len(self.classes_) - 1)
        return self.classes_[idx]


# ---------------------------------------------------------------------------
# IVTURS (FARC-HD with interval-valued type-2 fuzzy sets)
# ---------------------------------------------------------------------------

class IVTURSClassifier(FARCHDClassifier):
    """IVTURS – Interval-Valued Type-2 Unordered Rule Sets.

    This is FARC-HD run with interval-valued type-2 (IT2) fuzzy partitions,
    which is the defining characteristic of IVTURS:
    Sanz J.A., Bustince H., Herrera F. (2013). IVTURS: A Linguistic Fuzzy
    Rule-Based Classification System Based on a New Interval-Valued Fuzzy
    Reasoning Method with Tuning and Rule Selection.
    *IEEE Transactions on Fuzzy Systems*, 21(3), 399–411.

    The reference implementation is JoseanSanz/IVTURSfast (Java/KEEL).
    This Python approximation replicates the IT2 partition structure using
    ex-fuzzy's interval-valued type-2 fuzzy sets inside the FARC-HD framework.

    Parameters are identical to FARCHDClassifier except fuzzy_type is fixed to 't2'.
    """

    def __init__(
        self,
        n_labels: int = 5,
        max_depth: int = 2,
        min_support: float = 0.05,
        maxconf: float = 0.8,
        kt: int = 2,
        pop: int = 50,
        evaluations: int = 200,
        bitsgene: int = 30,
        delta: float = 0.2,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_labels=n_labels,
            max_depth=max_depth,
            min_support=min_support,
            maxconf=maxconf,
            kt=kt,
            pop=pop,
            evaluations=evaluations,
            bitsgene=bitsgene,
            delta=delta,
            fuzzy_type="t2",
            random_state=random_state,
        )


def _install_ex_fuzzy_farchd_src_alias() -> None:
    """Support upstream imports that still reference ``src.ex_fuzzy_farchd``."""

    package = importlib.import_module("ex_fuzzy_farchd")
    package_path = list(getattr(package, "__path__", []))

    src_package = sys.modules.get("src")
    if src_package is None:
        src_package = types.ModuleType("src")
        src_package.__path__ = []  # type: ignore[attr-defined]
        sys.modules["src"] = src_package

    alias = sys.modules.get("src.ex_fuzzy_farchd")
    if alias is None:
        alias = types.ModuleType("src.ex_fuzzy_farchd")
        alias.__path__ = package_path  # type: ignore[attr-defined]
        sys.modules["src.ex_fuzzy_farchd"] = alias

    setattr(src_package, "ex_fuzzy_farchd", alias)
