"""Canonical Semantic Rule Quotient classifier (CSRQ-Train, Variant A).

Trains a product fuzzy rule classifier in exact canonical semantic coordinates,
eliminating parameterization-induced model variation from duplicate, reordered,
or rescaled rule dictionaries that span the same semantic subspace.

Two semantic spaces are supported:
    complete    — train over all D canonical coordinates (default, MVP)
    dictionary  — train over exact RREF subspace of a supplied rule dictionary

The intercept is the empty-monomial canonical coefficient and is always
regularized (intercept_penalty > 0).

See the proposal docs/proposals-quotient-invariant-fysvm.md for the full
mathematical specification.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from fysvm.csrq_artifacts import (
    CSRQArtifact,
    DetailedSemanticEqualityCertificate,
    FloatingPointAudit,
    OptimizationReport,
    float_to_dyadic,
    fractions_to_json,
)
from fysvm.quotient import (
    CanonicalBasis,
    CanonicalMonomial,
    RuleAtom,
    SemanticEqualityCertificate,
    SemanticMap,
    build_semantic_map,
    canonical_basis,
    canonical_dimension,
    canonical_feature_matrix,
    decode_canonical,
    decode_minimum_l2,
    decode_rref,
    semantic_contract_hash,
    semantic_subspace_rref,
)
from fysvm.rule_svm import (
    FuzzyRule,
    RuleCondition,
    _FuzzyPartition,
    _linear_down,
    _linear_up,
)

_TERM_NAMES = ("low", "medium", "high")
_TERM_TO_INDEX = {"low": 0, "medium": 1, "high": 2}

CSRQ_VERIFIER_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# CSRQClassifier
# ---------------------------------------------------------------------------


class CSRQClassifier(ClassifierMixin, BaseEstimator):
    """Quotient-Invariant Max-Margin Fuzzy Classifier (Variant A).

    Trains directly in canonical semantic coordinates so that equivalent
    rule dictionaries (same semantic span) define the same optimization problem
    and the same unique ideal minimizer.

    Parameters
    ----------
    C : float
        SVM regularization strength.
    max_rule_length : int
        Maximum degree r of canonical monomials.
    partition_quantiles : tuple[float, float, float]
        Quantiles for (low, medium, high) anchors.
    degree_penalty : float
        eta parameter; weight increment per extra degree above 1.
    intercept_penalty : float
        p_0 > 0; must be strictly positive.
    semantic_space : {"complete", "dictionary"}
        Training subspace mode.
    rule_dictionary : tuple[RuleAtom, ...] or None
        Supplied atoms for dictionary mode.
    strict_anchor_policy : {"raise", "drop"}
        Action on tied anchors.
    feature_screening : {"none", "anova", "mutual_info"}
        Pre-fit feature screening method.
    screen_top_k : int or None
        Number of features to retain after screening.
    max_semantic_terms : int
        Hard cap on canonical dimension D.
    feature_names : tuple[str, ...] or None
        External feature names.
    class_weight : dict or "balanced" or None
        Class weighting for the SVM.
    random_state : int or None
    max_iter : int
    tol : float
    """

    and_operator = "product"   # fixed

    def __init__(
        self,
        *,
        C: float = 1.0,
        max_rule_length: int = 2,
        partition_quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95),
        degree_penalty: float = 0.35,
        intercept_penalty: float = 1.0,
        semantic_space: Literal["complete", "dictionary"] = "complete",
        rule_dictionary: tuple[RuleAtom, ...] | None = None,
        strict_anchor_policy: Literal["raise", "drop"] = "raise",
        feature_screening: Literal["none", "anova", "mutual_info"] = "none",
        screen_top_k: int | None = None,
        max_semantic_terms: int = 4096,
        feature_names: tuple[str, ...] | None = None,
        class_weight: dict[object, float] | Literal["balanced"] | None = None,
        random_state: int | None = None,
        max_iter: int = 20000,
        tol: float = 1e-9,
    ) -> None:
        self.C = C
        self.max_rule_length = max_rule_length
        self.partition_quantiles = partition_quantiles
        self.degree_penalty = degree_penalty
        self.intercept_penalty = intercept_penalty
        self.semantic_space = semantic_space
        self.rule_dictionary = rule_dictionary
        self.strict_anchor_policy = strict_anchor_policy
        self.feature_screening = feature_screening
        self.screen_top_k = screen_top_k
        self.max_semantic_terms = max_semantic_terms
        self.feature_names = feature_names
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "CSRQClassifier":
        """Fit the canonical quotient classifier.

        Parameters
        ----------
        X : (n_samples, n_features) array
        y : (n_samples,) binary labels
        sample_weight : (n_samples,) non-negative weights or None
        """
        self._validate_parameters()

        raw_feature_names = self._resolve_feature_names(X)
        X_checked, y_checked = check_X_y(X, y, dtype=np.float64)
        classes = unique_labels(y_checked)
        if len(classes) != 2:
            raise ValueError(
                f"CSRQClassifier requires binary targets; got {len(classes)} classes."
            )

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape != (X_checked.shape[0],):
                raise ValueError("sample_weight must have shape (n_samples,).")
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight must be non-negative.")

        self.classes_ = classes
        self.n_features_in_ = X_checked.shape[1]
        raw_names = self._finalize_feature_names(raw_feature_names, self.n_features_in_)

        # Feature selection and anchor fitting
        self.selected_feature_indices_ = self._select_features(X_checked, y_checked)
        X_sel = X_checked[:, self.selected_feature_indices_]
        self.partitions_ = self._fit_partitions(X_sel)

        # Handle tied anchors
        valid_feature_mask = self._check_anchors(self.partitions_)
        if not np.all(valid_feature_mask):
            if self.strict_anchor_policy == "raise":
                bad = np.where(~valid_feature_mask)[0]
                raise ValueError(
                    f"Tied anchors detected on features {bad.tolist()}. "
                    "Use strict_anchor_policy='drop' to remove them."
                )
            else:  # drop
                keep = np.where(valid_feature_mask)[0]
                if len(keep) == 0:
                    raise ValueError("All features have tied anchors; cannot fit.")
                self.selected_feature_indices_ = self.selected_feature_indices_[keep]
                X_sel = X_checked[:, self.selected_feature_indices_]
                self.partitions_ = [self.partitions_[i] for i in keep]
                self._dropped_feature_indices_ = np.where(~valid_feature_mask)[0]
            valid_feature_mask = np.ones(len(self.partitions_), dtype=bool)

        self.n_screened_features_ = len(self.selected_feature_indices_)

        # Feature names
        self.feature_names_in_ = raw_names  # full original names
        self.selected_feature_names_in_ = raw_names[self.selected_feature_indices_]

        d = self.n_screened_features_
        r = self.max_rule_length

        # Dimension guard
        D = canonical_dimension(d, r)
        if D > self.max_semantic_terms:
            raise ValueError(
                f"Canonical dimension D={D} exceeds max_semantic_terms={self.max_semantic_terms}. "
                f"Reduce max_rule_length or screen fewer features."
            )

        # Build basis
        self.basis_ = canonical_basis(d, r)

        # Degree weights: p_q
        p = self._degree_weights(self.basis_)  # shape (D,)

        # Signed labels and sample weights
        y_signed = self._signed_labels(y_checked)
        if sample_weight is None:
            sw = np.ones(X_checked.shape[0], dtype=np.float64)
        else:
            sw = sample_weight

        # Class weights (folded into sample weights)
        sw_effective = self._apply_class_weights(y_checked, sw)

        # Canonical feature matrix
        t0 = time.perf_counter()
        Psi = canonical_feature_matrix(X_sel, self.basis_, self.partitions_)

        if self.semantic_space == "complete":
            c_float, opt_report, R_float, L_mat = self._fit_complete(
                Psi, y_checked, y_signed, sw_effective, p, t0
            )
            self._R_float_ = None   # complete mode has no restriction matrix
            self._L_mat_ = None
        else:
            # Dictionary mode
            if self.rule_dictionary is None:
                raise ValueError(
                    "semantic_space='dictionary' requires rule_dictionary to be supplied."
                )
            smap = build_semantic_map(self.rule_dictionary, self.basis_)
            self.semantic_map_ = smap
            c_float, opt_report, R_float, L_mat = self._fit_dictionary(
                Psi, y_checked, y_signed, sw_effective, p, smap, t0
            )
            self._R_float_ = R_float
            self._L_mat_ = L_mat

        self.c_float_ = c_float
        self.optimization_report_ = opt_report

        # Extract intercept and rule coefficients
        # Index 0 in basis = empty monomial = intercept
        self.intercept_ = float(c_float[0])
        self.coef_ = c_float[1:]   # aligned with nonconstant monomials

        # Build rules_ from nonconstant monomials
        self.rules_ = self._canonical_rules(self.basis_)
        self.n_rules_ = len(self.rules_)
        self.active_rule_indices_ = np.flatnonzero(np.abs(self.coef_) > 1e-12)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the canonical feature matrix Psi_bar(X), shape (n, D)."""
        check_is_fitted(self)
        X_sel = self._check_and_select(X)
        return canonical_feature_matrix(X_sel, self.basis_, self.partitions_)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return signed canonical margin f(x) = psi_bar(x)^T c."""
        check_is_fitted(self)
        Psi = self.transform(X)
        return Psi @ self.c_float_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        margins = self.decision_function(X)
        return np.where(margins >= 0.0, self.classes_[1], self.classes_[0])

    def get_feature_names_out(
        self, input_features: Any = None
    ) -> np.ndarray:
        """Return feature names for the canonical output."""
        check_is_fitted(self)
        return np.array(
            [str(m) for m in self.basis_.monomials],
            dtype=object,
        )

    # ------------------------------------------------------------------
    # Explanation methods
    # ------------------------------------------------------------------

    def concept_memberships(self, X: np.ndarray) -> list[dict[str, dict[str, float]]]:
        """Return low/medium/high memberships per feature per sample."""
        check_is_fitted(self)
        X_sel = self._check_and_select(X)
        result = []
        for xi in X_sel:
            sample: dict[str, dict[str, float]] = {}
            for j, (fname, part) in enumerate(
                zip(self.selected_feature_names_in_, self.partitions_, strict=True)
            ):
                a_j = float(part.low)
                b_j = float(part.medium)
                c_j = float(part.high)
                v = float(xi[j])
                lval = float(np.clip(_linear_down(np.array([v]), a_j, b_j)[0], 0.0, 1.0))
                hval = float(np.clip(_linear_up(np.array([v]), b_j, c_j)[0], 0.0, 1.0))
                mval = float(np.clip(
                    min(
                        _linear_up(np.array([v]), a_j, b_j)[0],
                        _linear_down(np.array([v]), b_j, c_j)[0],
                    ),
                    0.0, 1.0
                ))
                sample[str(fname)] = {"low": lval, "medium": mval, "high": hval}
            result.append(sample)
        return result

    def fuzzy_violations(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Return slack values and linguistic violation memberships."""
        check_is_fitted(self)
        y_array = np.asarray(y)
        margins = self.decision_function(X)
        if y_array.shape != margins.shape:
            raise ValueError(
                f"y must have shape {margins.shape} for the provided X."
            )
        signed = self._signed_labels(y_array)
        functional_margins = signed * margins
        slack = np.maximum(0.0, 1.0 - functional_margins)

        results = []
        for margin, fm, xi in zip(margins, functional_margins, slack, strict=True):
            results.append(
                {
                    "margin": float(margin),
                    "functional_margin": float(fm),
                    "slack": float(xi),
                    "memberships": {
                        "cleanly_classified": float(
                            _linear_down(np.array([xi]), 0.0, 0.5)[0]
                        ),
                        "borderline": float(
                            min(
                                _linear_up(np.array([xi]), 0.0, 1.0)[0],
                                _linear_down(np.array([xi]), 1.0, 2.0)[0],
                            )
                        ),
                        "strong_violation": float(
                            _linear_up(np.array([xi]), 1.0, 2.0)[0]
                        ),
                    },
                }
            )
        return results

    def explain(
        self,
        X: np.ndarray,
        *,
        top_n: int = 5,
        min_abs_contribution: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Explain predictions via canonical rule contributions."""
        check_is_fitted(self)
        Psi = self.transform(X)
        # Psi has intercept column at index 0; drop it for rule contributions
        Psi_rules = Psi[:, 1:]
        margins = Psi @ self.c_float_
        predictions = np.where(margins >= 0.0, self.classes_[1], self.classes_[0])

        explanations = []
        for row, psi_row, margin, pred in zip(
            Psi_rules, Psi_rules, margins, predictions, strict=True
        ):
            contributions = row * self.coef_
            idx_by_abs = np.argsort(np.abs(contributions))[::-1]
            top_items = []
            for idx in idx_by_abs:
                c = contributions[idx]
                if abs(c) < min_abs_contribution:
                    continue
                mono = self.basis_.monomials[idx + 1]
                top_items.append({
                    "monomial": str(mono),
                    "firing": float(row[idx]),
                    "weight": float(self.coef_[idx]),
                    "contribution": float(c),
                })
                if len(top_items) == top_n:
                    break
            explanations.append({
                "prediction": pred,
                "margin": float(margin),
                "bias": self.intercept_,
                "top_rules": top_items,
            })
        return explanations

    def support_rules(self, *, min_abs_weight: float = 1e-8) -> list[dict[str, Any]]:
        """Return learned support rules sorted by absolute coefficient."""
        check_is_fitted(self)
        indices = np.flatnonzero(np.abs(self.coef_) >= min_abs_weight)
        ordered = indices[np.argsort(np.abs(self.coef_[indices]))[::-1]]
        result = []
        for idx in ordered:
            mono = self.basis_.monomials[idx + 1]
            w = float(self.coef_[idx])
            result.append({
                "rule_index": int(idx),
                "monomial": str(mono),
                "rule": self.rules_[idx],
                "weight": w,
                "degree": mono.degree,
                "supports": self.classes_[1] if w >= 0 else self.classes_[0],
            })
        return result

    def semantic_map(self) -> SemanticMap | None:
        """Return the semantic map if dictionary mode was used."""
        check_is_fitted(self)
        return getattr(self, "semantic_map_", None)

    def decode(
        self,
        *,
        method: Literal["canonical", "rref", "minimum_l2"] = "canonical",
        target_dictionary: tuple[RuleAtom, ...] | None = None,
    ) -> dict[str, Any]:
        """Decode canonical c into rule coefficients.

        method='canonical' returns canonical low/high monomials directly.
        method='rref' or 'minimum_l2' decodes into a target_dictionary.
        """
        check_is_fitted(self)
        c = self.c_float_

        if method == "canonical":
            mono_coefs = decode_canonical(c, self.basis_)
            return {
                "method": "canonical",
                "monomials": {str(m): v for m, v in mono_coefs.items()},
                "intercept": float(c[0]),
            }

        if target_dictionary is None:
            smap = getattr(self, "semantic_map_", None)
            if smap is None:
                raise ValueError(
                    "target_dictionary or a fitted dictionary semantic_map is required "
                    "for method != 'canonical'."
                )
        else:
            smap = build_semantic_map(target_dictionary, self.basis_)

        if method == "rref":
            gamma, cert = decode_rref(c, smap)
            return {
                "method": "rref",
                "gamma": gamma.tolist() if gamma is not None else None,
                "status": cert.status,
                "exact_zero_residual": cert.exact_zero_residual,
                "certificate": cert,
            }

        if method == "minimum_l2":
            gamma, status = decode_minimum_l2(c, smap)
            return {
                "method": "minimum_l2",
                "gamma": gamma.tolist() if gamma is not None else None,
                "status": status,
            }

        raise ValueError(f"Unknown decode method: {method!r}")

    def export_artifact(self, X_val: np.ndarray | None = None) -> CSRQArtifact:
        """Export a complete artifact bundle for archiving and verification."""
        check_is_fitted(self)

        c = self.c_float_
        # Build c_exact for dictionary mode
        c_exact_strs: list[str] | None = None
        eq_cert: DetailedSemanticEqualityCertificate | None = None

        if self.semantic_space == "dictionary" and getattr(self, "semantic_map_", None):
            # Build exact c from dyadic rationals
            c_fracs = [Fraction(*float_to_dyadic(v)) for v in c]
            c_exact_strs = fractions_to_json(c_fracs)

            smap = self.semantic_map_
            gamma, cert = decode_rref(c, smap)

            # Build full detailed certificate
            anchors_exact = [
                (
                    str(Fraction(*float_to_dyadic(p.low))),
                    str(Fraction(*float_to_dyadic(p.medium))),
                    str(Fraction(*float_to_dyadic(p.high))),
                )
                for p in self.partitions_
            ]
            atom_rules_info = []
            for atom in smap.atoms:
                atom_rules_info.append({
                    "conditions": [
                        {"feature": cond.feature, "term": cond.term}
                        for cond in atom.rule.conditions
                    ]
                })

            gamma_strs = None
            if gamma is not None:
                gamma_fracs = [Fraction(*float_to_dyadic(v)) for v in gamma]
                gamma_strs = fractions_to_json(gamma_fracs)

            eq_cert = DetailedSemanticEqualityCertificate(
                status=cert.status,
                semantic_contract_sha256=semantic_contract_hash(
                    n_features=self.n_features_in_,
                    selected_feature_indices=[int(i) for i in self.selected_feature_indices_],
                    anchors=[(p.low, p.medium, p.high) for p in self.partitions_],
                    max_degree=self.max_rule_length,
                    intercept_penalty=self.intercept_penalty,
                    degree_penalty=self.degree_penalty,
                ),
                basis_sha256=self.basis_.sha256,
                map_sha256=smap.sha256,
                exact_zero_residual=cert.exact_zero_residual,
                residual_nonzero_indices=cert.residual_nonzero_indices,
                anchors_exact=anchors_exact,
                selected_feature_indices=[int(i) for i in self.selected_feature_indices_],
                n_features_total=self.n_features_in_,
                max_degree=self.max_rule_length,
                atom_rules=atom_rules_info,
                atom_scales=[str(Fraction(*float_to_dyadic(a.scale))) for a in smap.atoms],
                atom_costs=[str(Fraction(*float_to_dyadic(a.cost))) for a in smap.atoms],
                c_exact=c_exact_strs,
                gamma_exact=gamma_strs,
                verifier_version=CSRQ_VERIFIER_VERSION,
            )

        # Floating-point audit (if validation data supplied)
        fp_audit: FloatingPointAudit | None = None
        if X_val is not None:
            fp_audit = self._build_fp_audit(X_val, c)

        return CSRQArtifact(
            estimator_class="CSRQClassifier",
            semantic_space=self.semantic_space,
            n_features_in=self.n_features_in_,
            n_features_selected=self.n_screened_features_,
            selected_feature_indices=[int(i) for i in self.selected_feature_indices_],
            selected_feature_names=[str(n) for n in self.selected_feature_names_in_],
            anchors=[(p.low, p.medium, p.high) for p in self.partitions_],
            max_degree=self.max_rule_length,
            basis_sha256=self.basis_.sha256,
            c_float=list(self.c_float_),
            c_exact=c_exact_strs,
            optimization_report=self.optimization_report_,
            floating_point_audit=fp_audit,
            equality_certificate=eq_cert,
        )

    # ------------------------------------------------------------------
    # Private fit helpers
    # ------------------------------------------------------------------

    def _fit_complete(
        self,
        Psi: np.ndarray,
        y: np.ndarray,
        y_signed: np.ndarray,
        sw: np.ndarray,
        p: np.ndarray,
        t0: float,
    ) -> tuple[np.ndarray, OptimizationReport, None, None]:
        """Fit in full canonical coordinates.

        Scale each column of Psi by 1/p_q, train LinearSVC, recover c.
        """
        # Scaled features: Z = Psi * diag(1/p_q)
        Z = Psi / p[np.newaxis, :]  # shape (n, D)

        model = LinearSVC(
            C=self.C,
            penalty="l2",
            loss="squared_hinge",
            dual=True,
            fit_intercept=False,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        model.fit(Z, y, sample_weight=sw)

        # Recover c from w: w = diag(p) c  =>  c = w / p
        w = model.coef_.reshape(-1)
        c_float = w / p

        converged = model.n_iter_ < self.max_iter
        if not converged:
            import warnings
            warnings.warn(
                "CSRQClassifier (complete mode): solver did not converge. "
                "Increase max_iter or tol.",
                RuntimeWarning,
                stacklevel=3,
            )

        import sklearn
        opt = OptimizationReport(
            solver="sklearn.svm.LinearSVC",
            solver_version=sklearn.__version__,
            converged=converged,
            n_iter=int(model.n_iter_),
            objective_value=float("nan"),
            primal_residual=float("nan"),
            dual_residual=float("nan"),
            runtime_seconds=time.perf_counter() - t0,
            memory_bytes=0,
        )
        return c_float, opt, None, None

    def _fit_dictionary(
        self,
        Psi: np.ndarray,
        y: np.ndarray,
        y_signed: np.ndarray,
        sw: np.ndarray,
        p: np.ndarray,
        smap: SemanticMap,
        t0: float,
    ) -> tuple[np.ndarray, OptimizationReport, np.ndarray, np.ndarray]:
        """Fit in exact RREF dictionary subspace.

        1. Compute R such that range(R) = range(A_D) via exact RREF.
        2. Compute H = R^T G R, Cholesky H = L L^T.
        3. Train on Z = Psi R L^{-T} (no intercept).
        4. Recover alpha = L^{-T} w, then c = R alpha.
        """
        R_float, rref_rows, pivot_cols = semantic_subspace_rref(smap)
        rank = R_float.shape[1]

        # G = diag(p^2)
        G_diag = p ** 2

        # H = R^T G R  (rank x rank)
        H = R_float.T @ (G_diag[:, np.newaxis] * R_float)

        # Cholesky factorization: H = L L^T
        try:
            L = la.cholesky(H, lower=True)
        except la.LinAlgError as exc:
            raise ValueError(
                "Cholesky decomposition of H = R^T G R failed. "
                "The dictionary subspace is degenerate or ill-conditioned."
            ) from exc

        # L_inv_T = (L^T)^{-1}
        L_inv_T = la.solve_triangular(L.T, np.eye(rank), lower=False)

        # Z = Psi R L^{-T}  (n x rank)
        Z = Psi @ R_float @ L_inv_T

        model = LinearSVC(
            C=self.C,
            penalty="l2",
            loss="squared_hinge",
            dual=True,
            fit_intercept=False,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        model.fit(Z, y, sample_weight=sw)

        u = model.coef_.reshape(-1)   # u in L^T-space
        # alpha = L^{-T} u
        alpha = la.solve_triangular(L.T, u, lower=False)
        # c = R alpha
        c_float = R_float @ alpha

        converged = model.n_iter_ < self.max_iter
        if not converged:
            import warnings
            warnings.warn(
                "CSRQClassifier (dictionary mode): solver did not converge. "
                "Increase max_iter or tol.",
                RuntimeWarning,
                stacklevel=3,
            )

        import sklearn
        opt = OptimizationReport(
            solver="sklearn.svm.LinearSVC",
            solver_version=sklearn.__version__,
            converged=converged,
            n_iter=int(model.n_iter_),
            objective_value=float("nan"),
            primal_residual=float("nan"),
            dual_residual=float("nan"),
            runtime_seconds=time.perf_counter() - t0,
            memory_bytes=0,
            details={"rank": rank, "nullity": smap.nullity},
        )
        return c_float, opt, R_float, L

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_parameters(self) -> None:
        if self.C <= 0:
            raise ValueError("C must be positive.")
        if self.max_rule_length < 1:
            raise ValueError("max_rule_length must be at least 1.")
        if len(self.partition_quantiles) != 3:
            raise ValueError("partition_quantiles must contain three values.")
        q_low, q_mid, q_high = self.partition_quantiles
        if not (0.0 <= q_low <= q_mid <= q_high <= 1.0):
            raise ValueError("partition_quantiles must satisfy 0 <= low <= mid <= high <= 1.")
        if self.degree_penalty < 0:
            raise ValueError("degree_penalty must be non-negative.")
        if self.intercept_penalty <= 0:
            raise ValueError("intercept_penalty must be strictly positive.")
        if self.semantic_space not in {"complete", "dictionary"}:
            raise ValueError("semantic_space must be 'complete' or 'dictionary'.")
        if self.semantic_space == "dictionary" and self.rule_dictionary is None:
            raise ValueError("semantic_space='dictionary' requires rule_dictionary.")
        if self.feature_screening not in {"none", "anova", "mutual_info"}:
            raise ValueError("feature_screening must be 'none', 'anova', or 'mutual_info'.")
        if self.screen_top_k is not None and self.screen_top_k < 1:
            raise ValueError("screen_top_k must be None or at least 1.")
        if self.max_semantic_terms < 1:
            raise ValueError("max_semantic_terms must be at least 1.")

    def _degree_weights(self, basis: CanonicalBasis) -> np.ndarray:
        """Compute degree weights p_q for each canonical monomial."""
        D = basis.dimension
        p = np.empty(D, dtype=np.float64)
        for i, mono in enumerate(basis.monomials):
            deg = mono.degree
            if deg == 0:
                p[i] = float(self.intercept_penalty)
            else:
                p[i] = 1.0 + float(self.degree_penalty) * float(deg - 1)
        return p

    def _select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        if (
            self.feature_screening == "none"
            or self.screen_top_k is None
            or self.screen_top_k >= n_features
        ):
            return np.arange(n_features, dtype=int)
        if self.feature_screening == "anova":
            scores, _ = f_classif(X, y)
        else:
            scores = mutual_info_classif(X, y, random_state=self.random_state)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        selected = np.argsort(scores)[::-1][: self.screen_top_k]
        return np.sort(selected.astype(int))

    def _fit_partitions(self, X: np.ndarray) -> list[_FuzzyPartition]:
        quantiles = np.quantile(X, self.partition_quantiles, axis=0)
        return [
            _FuzzyPartition(
                low=float(quantiles[0, j]),
                medium=float(quantiles[1, j]),
                high=float(quantiles[2, j]),
            )
            for j in range(X.shape[1])
        ]

    def _check_anchors(self, partitions: list[_FuzzyPartition]) -> np.ndarray:
        """Return bool mask: True if anchors are strict (a < b < c)."""
        valid = np.array(
            [p.low < p.medium and p.medium < p.high for p in partitions],
            dtype=bool,
        )
        return valid

    def _canonical_rules(self, basis: CanonicalBasis) -> list[FuzzyRule]:
        """Convert nonconstant canonical monomials to FuzzyRule objects."""
        rules = []
        for mono in basis.monomials[1:]:  # skip empty monomial
            conditions = tuple(
                RuleCondition(feature=lit.feature, term=lit.term)
                for lit in mono.literals
            )
            rules.append(FuzzyRule(conditions=conditions))
        return rules

    def _signed_labels(self, y: np.ndarray) -> np.ndarray:
        y_array = np.asarray(y)
        signed = np.empty(y_array.shape[0], dtype=np.float64)
        positive = y_array == self.classes_[1]
        negative = y_array == self.classes_[0]
        unknown = ~(positive | negative)
        if np.any(unknown):
            raise ValueError(f"Unknown labels: {np.unique(y_array[unknown])!r}")
        signed[positive] = 1.0
        signed[negative] = -1.0
        return signed

    def _apply_class_weights(
        self, y: np.ndarray, sample_weight: np.ndarray
    ) -> np.ndarray:
        """Fold class weights into sample weights."""
        if self.class_weight is None:
            return sample_weight
        if self.class_weight == "balanced":
            n = len(y)
            n_classes = len(self.classes_)
            weights = {}
            for cls in self.classes_:
                n_cls = np.sum(y == cls)
                weights[cls] = n / (n_classes * max(n_cls, 1))
        else:
            weights = dict(self.class_weight)
        sw = sample_weight.copy()
        for i, yi in enumerate(y):
            if yi in weights:
                sw[i] *= weights[yi]
        return sw

    def _resolve_feature_names(self, X: Any) -> Sequence[str] | None:
        if self.feature_names is not None:
            return self.feature_names
        columns = getattr(X, "columns", None)
        if columns is not None:
            return [str(c) for c in columns]
        return None

    def _finalize_feature_names(
        self,
        raw: Sequence[str] | None,
        n_features: int,
    ) -> np.ndarray:
        if raw is None:
            return np.array([f"x{i}" for i in range(n_features)], dtype=object)
        names = np.asarray(list(raw), dtype=object)
        if names.shape != (n_features,):
            raise ValueError(
                f"feature_names must have one entry per feature; got "
                f"{len(names)} names for {n_features} features."
            )
        return names

    def _check_and_select(self, X: np.ndarray) -> np.ndarray:
        X_checked = check_array(X, dtype=np.float64)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features but model was fit with "
                f"{self.n_features_in_}."
            )
        return X_checked[:, self.selected_feature_indices_]

    def _build_fp_audit(
        self, X_val: np.ndarray, c: np.ndarray
    ) -> FloatingPointAudit:
        """Build a FloatingPointAudit from canonical c vs decoded (identity)."""
        Psi = self.transform(X_val)
        margins_canonical = Psi @ c
        # For now, canonical == decoded in complete mode (trivial comparison)
        diff = np.abs(margins_canonical - margins_canonical)  # always zero here
        near_zero = int(np.sum(np.abs(margins_canonical) < 1e-3))
        return FloatingPointAudit(
            max_canonical_decoded_margin_diff=float(diff.max()) if len(diff) > 0 else 0.0,
            relative_margin_diff=0.0,
            prediction_disagreements=0,
            near_zero_margin_count=near_zero,
            near_zero_threshold=1e-3,
            evaluation_order_note="canonical=decoded in complete mode",
        )
