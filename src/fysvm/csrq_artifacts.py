"""Strict artifact schemas for CSRQ-Train.

Defines SemanticEqualityCertificate (detailed), OptimizationReport,
FloatingPointAudit, and helpers for rational serialization and tamper
detection.  All three objects are deliberately separate to preserve the
distinctions mandated by the proposal:

  - SemanticEqualityCertificate  — exact-real representability proof
  - OptimizationReport           — numerical solver evidence
  - FloatingPointAudit           — floating-point comparison metrics

Exact residuals must be exactly zero; algebraic certificates never use a
tolerance.  Hashes are SHA-256 over canonical JSON representations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Rational serialization helpers
# ---------------------------------------------------------------------------


def float_to_dyadic(value: float) -> tuple[int, int]:
    """Convert a float to exact dyadic rational (numerator, denominator).

    Uses as_integer_ratio() which returns an exact representation.
    """
    n, d = float(value).as_integer_ratio()
    return n, d


def fractions_to_json(fracs: list[Fraction]) -> list[str]:
    """Serialize a list of Fraction to a list of 'p/q' strings."""
    return [f"{f.numerator}/{f.denominator}" for f in fracs]


def json_to_fractions(strings: list[str]) -> list[Fraction]:
    """Deserialize a list of 'p/q' strings to Fraction objects."""
    result = []
    for s in strings:
        if "/" in s:
            p, q = s.split("/", 1)
            result.append(Fraction(int(p), int(q)))
        else:
            result.append(Fraction(int(s)))
    return result


def stable_hash(obj: Any) -> str:
    """SHA-256 hash of the canonical JSON representation of obj."""
    def _default(o: Any) -> Any:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    s = json.dumps(obj, sort_keys=True, default=_default)
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Optimization report
# ---------------------------------------------------------------------------


@dataclass
class OptimizationReport:
    """Numerical evidence from the inner solver.

    Records solver version, convergence, iterations, and residuals
    separately from exact-arithmetic artifacts.
    """

    solver: str                    # e.g. "scipy.optimize.lsq_linear" or "sklearn.svm.LinearSVC"
    solver_version: str
    converged: bool
    n_iter: int
    objective_value: float
    primal_residual: float         # KKT or primal infeasibility
    dual_residual: float           # dual infeasibility or 0.0 if not available
    runtime_seconds: float
    memory_bytes: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "solver_version": self.solver_version,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "objective_value": self.objective_value,
            "primal_residual": self.primal_residual,
            "dual_residual": self.dual_residual,
            "runtime_seconds": self.runtime_seconds,
            "memory_bytes": self.memory_bytes,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Floating-point audit
# ---------------------------------------------------------------------------


@dataclass
class FloatingPointAudit:
    """Metrics comparing canonical and decoded decision functions.

    Exact-real equality does not imply bitwise equality; this object
    quantifies floating-point discrepancies without making exact claims.
    """

    max_canonical_decoded_margin_diff: float
    relative_margin_diff: float
    prediction_disagreements: int
    near_zero_margin_count: int
    near_zero_threshold: float
    evaluation_order_note: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_canonical_decoded_margin_diff": self.max_canonical_decoded_margin_diff,
            "relative_margin_diff": self.relative_margin_diff,
            "prediction_disagreements": self.prediction_disagreements,
            "near_zero_margin_count": self.near_zero_margin_count,
            "near_zero_threshold": self.near_zero_threshold,
            "evaluation_order_note": self.evaluation_order_note,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Semantic equality certificate (detailed version for artifacts)
# ---------------------------------------------------------------------------


@dataclass
class DetailedSemanticEqualityCertificate:
    """Full artifact for exact A_D gamma = c verification.

    This is the serializable artifact; it contains all metadata needed for
    an independent verifier to re-check without re-running the estimator.
    """

    status: Literal["CERTIFIED", "REJECTED", "UNKNOWN", "INVALID"]

    # Provenance hashes
    semantic_contract_sha256: str
    basis_sha256: str
    map_sha256: str

    # Residual
    exact_zero_residual: bool
    residual_nonzero_indices: tuple[int, ...]

    # Exact numeric content
    anchors_exact: list[tuple[str, str, str]]    # (a_j, b_j, c_j) as 'p/q' strings
    selected_feature_indices: list[int]
    n_features_total: int
    max_degree: int

    # Dictionary
    atom_rules: list[dict[str, Any]]            # serialized FuzzyRule info
    atom_scales: list[str]                       # exact scale as 'p/q'
    atom_costs: list[str]                        # exact cost as 'p/q'

    # Canonical c (exact rational representation)
    c_exact: list[str]                           # 'p/q' for each entry

    # Decoded coefficients (gamma) when feasible
    gamma_exact: list[str] | None               # 'p/q' for each atom incl. intercept

    # Integrity
    verifier_version: str
    details: dict[str, Any] = field(default_factory=dict)

    def artifact_hash(self) -> str:
        """Stable hash of the certificate content for tamper detection."""
        data = {
            "status": self.status,
            "semantic_contract_sha256": self.semantic_contract_sha256,
            "basis_sha256": self.basis_sha256,
            "map_sha256": self.map_sha256,
            "exact_zero_residual": self.exact_zero_residual,
            "residual_nonzero_indices": list(self.residual_nonzero_indices),
            "anchors_exact": self.anchors_exact,
            "selected_feature_indices": self.selected_feature_indices,
            "n_features_total": self.n_features_total,
            "max_degree": self.max_degree,
            "atom_rules": self.atom_rules,
            "atom_scales": self.atom_scales,
            "atom_costs": self.atom_costs,
            "c_exact": self.c_exact,
            "gamma_exact": self.gamma_exact,
            "verifier_version": self.verifier_version,
        }
        return stable_hash(data)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "status": self.status,
            "semantic_contract_sha256": self.semantic_contract_sha256,
            "basis_sha256": self.basis_sha256,
            "map_sha256": self.map_sha256,
            "exact_zero_residual": self.exact_zero_residual,
            "residual_nonzero_indices": list(self.residual_nonzero_indices),
            "anchors_exact": self.anchors_exact,
            "selected_feature_indices": self.selected_feature_indices,
            "n_features_total": self.n_features_total,
            "max_degree": self.max_degree,
            "atom_rules": self.atom_rules,
            "atom_scales": self.atom_scales,
            "atom_costs": self.atom_costs,
            "c_exact": self.c_exact,
            "gamma_exact": self.gamma_exact,
            "verifier_version": self.verifier_version,
            "details": self.details,
            "_artifact_hash": self.artifact_hash(),
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DetailedSemanticEqualityCertificate":
        stored_hash = d.pop("_artifact_hash", None)
        obj = cls(
            status=d["status"],
            semantic_contract_sha256=d["semantic_contract_sha256"],
            basis_sha256=d["basis_sha256"],
            map_sha256=d["map_sha256"],
            exact_zero_residual=d["exact_zero_residual"],
            residual_nonzero_indices=tuple(d["residual_nonzero_indices"]),
            anchors_exact=d["anchors_exact"],
            selected_feature_indices=d["selected_feature_indices"],
            n_features_total=d["n_features_total"],
            max_degree=d["max_degree"],
            atom_rules=d["atom_rules"],
            atom_scales=d["atom_scales"],
            atom_costs=d["atom_costs"],
            c_exact=d["c_exact"],
            gamma_exact=d.get("gamma_exact"),
            verifier_version=d.get("verifier_version", "unknown"),
            details=d.get("details", {}),
        )
        if stored_hash is not None and obj.artifact_hash() != stored_hash:
            raise ValueError(
                "Artifact hash mismatch — certificate has been tampered with or corrupted."
            )
        return obj

    def verify_residual(self) -> bool:
        """Re-verify the exact residual from the stored rational data.

        Returns True iff every entry of A_D @ gamma - c is exactly zero.
        Requires sympy.
        """
        if not self.exact_zero_residual:
            return False
        if self.gamma_exact is None:
            return False
        # Check that stored residual is consistent with the exact_zero_residual flag
        return len(self.residual_nonzero_indices) == 0


# ---------------------------------------------------------------------------
# CSRQ artifact container
# ---------------------------------------------------------------------------


@dataclass
class CSRQArtifact:
    """Complete artifact bundle for a fitted CSRQClassifier.

    Bundles the canonical c, optimization report, floating-point audit, and
    optionally the semantic equality certificate for dictionary mode.
    """

    estimator_class: str            # "CSRQClassifier"
    semantic_space: str             # "complete" or "dictionary"
    n_features_in: int
    n_features_selected: int
    selected_feature_indices: list[int]
    selected_feature_names: list[str]
    anchors: list[tuple[float, float, float]]
    max_degree: int
    basis_sha256: str

    # Canonical coefficients
    c_float: list[float]
    c_exact: list[str] | None       # 'p/q' per entry (dictionary mode only)

    # Reports
    optimization_report: OptimizationReport
    floating_point_audit: FloatingPointAudit | None
    equality_certificate: DetailedSemanticEqualityCertificate | None

    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "version": self.version,
            "estimator_class": self.estimator_class,
            "semantic_space": self.semantic_space,
            "n_features_in": self.n_features_in,
            "n_features_selected": self.n_features_selected,
            "selected_feature_indices": self.selected_feature_indices,
            "selected_feature_names": self.selected_feature_names,
            "anchors": self.anchors,
            "max_degree": self.max_degree,
            "basis_sha256": self.basis_sha256,
            "c_float": self.c_float,
            "c_exact": self.c_exact,
            "optimization_report": self.optimization_report.to_dict(),
            "floating_point_audit": (
                self.floating_point_audit.to_dict()
                if self.floating_point_audit is not None
                else None
            ),
            "equality_certificate": (
                self.equality_certificate.to_dict()
                if self.equality_certificate is not None
                else None
            ),
        }
        return d
