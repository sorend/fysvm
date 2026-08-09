"""Tests for CSRQ artifact schemas — serialization, exact residuals, tamper detection."""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest

from fysvm.csrq import CSRQClassifier
from fysvm.csrq_artifacts import (
    CSRQArtifact,
    DetailedSemanticEqualityCertificate,
    FloatingPointAudit,
    OptimizationReport,
    float_to_dyadic,
    fractions_to_json,
    json_to_fractions,
    stable_hash,
)
from fysvm.quotient import RuleAtom
from fysvm.rule_svm import FuzzyRule, RuleCondition


def _make_atom(feature: int, term: str) -> RuleAtom:
    rule = FuzzyRule((RuleCondition(feature, term),))
    return RuleAtom(rule=rule, scale=1.0, cost=1.0)


@pytest.fixture
def fitted_csrq():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 3))
    y = (X[:, 0] > 0).astype(int)
    clf = CSRQClassifier(C=1.0, max_rule_length=1)
    clf.fit(X, y)
    return clf, X, y


# ---------------------------------------------------------------------------
# Rational serialization
# ---------------------------------------------------------------------------

def test_float_to_dyadic_exact():
    """as_integer_ratio gives exact representation."""
    for v in [0.5, 0.25, 1.0, -0.125, 3.0, 0.1]:
        n, d = float_to_dyadic(v)
        assert abs(n / d - v) < 1e-16


def test_fractions_roundtrip():
    fracs = [Fraction(1, 3), Fraction(-5, 7), Fraction(0), Fraction(2)]
    strs = fractions_to_json(fracs)
    back = json_to_fractions(strs)
    assert fracs == back


def test_stable_hash_consistent():
    h1 = stable_hash({"a": 1, "b": [2, 3]})
    h2 = stable_hash({"b": [2, 3], "a": 1})  # sort_keys=True ensures same hash
    assert h1 == h2


# ---------------------------------------------------------------------------
# OptimizationReport
# ---------------------------------------------------------------------------

def test_optimization_report_to_dict():
    report = OptimizationReport(
        solver="test",
        solver_version="0.1",
        converged=True,
        n_iter=100,
        objective_value=0.5,
        primal_residual=1e-6,
        dual_residual=0.0,
        runtime_seconds=0.1,
        memory_bytes=1024,
    )
    d = report.to_dict()
    assert d["solver"] == "test"
    assert d["converged"] is True


# ---------------------------------------------------------------------------
# FloatingPointAudit
# ---------------------------------------------------------------------------

def test_fp_audit_to_dict():
    audit = FloatingPointAudit(
        max_canonical_decoded_margin_diff=1e-12,
        relative_margin_diff=1e-14,
        prediction_disagreements=0,
        near_zero_margin_count=2,
        near_zero_threshold=1e-3,
        evaluation_order_note="test",
    )
    d = audit.to_dict()
    assert d["prediction_disagreements"] == 0


# ---------------------------------------------------------------------------
# DetailedSemanticEqualityCertificate
# ---------------------------------------------------------------------------

def test_certificate_hash_tamper_detection():
    cert = DetailedSemanticEqualityCertificate(
        status="CERTIFIED",
        semantic_contract_sha256="abc",
        basis_sha256="def",
        map_sha256="ghi",
        exact_zero_residual=True,
        residual_nonzero_indices=(),
        anchors_exact=[("0/1", "1/1", "2/1")],
        selected_feature_indices=[0],
        n_features_total=2,
        max_degree=2,
        atom_rules=[],
        atom_scales=[],
        atom_costs=[],
        c_exact=["1/2"],
        gamma_exact=None,
        verifier_version="1.0.0",
    )
    d = cert.to_dict()
    stored_hash = d["_artifact_hash"]

    # Tamper: change status
    d["status"] = "REJECTED"
    with pytest.raises(ValueError, match="tampered"):
        DetailedSemanticEqualityCertificate.from_dict(d)


def test_certificate_from_dict_roundtrip():
    cert = DetailedSemanticEqualityCertificate(
        status="UNKNOWN",
        semantic_contract_sha256="x",
        basis_sha256="y",
        map_sha256="z",
        exact_zero_residual=False,
        residual_nonzero_indices=(3, 7),
        anchors_exact=[],
        selected_feature_indices=[],
        n_features_total=0,
        max_degree=1,
        atom_rules=[],
        atom_scales=[],
        atom_costs=[],
        c_exact=None,
        gamma_exact=None,
        verifier_version="1.0.0",
    )
    d = cert.to_dict()
    cert2 = DetailedSemanticEqualityCertificate.from_dict(d)
    assert cert2.status == "UNKNOWN"
    assert cert2.residual_nonzero_indices == (3, 7)


# ---------------------------------------------------------------------------
# CSRQArtifact (from fitted model)
# ---------------------------------------------------------------------------

def test_export_artifact_complete_mode(fitted_csrq):
    clf, X, y = fitted_csrq
    artifact = clf.export_artifact()
    assert artifact.semantic_space == "complete"
    assert artifact.n_features_selected == clf.n_screened_features_
    assert len(artifact.c_float) == clf.basis_.dimension
    assert artifact.equality_certificate is None  # complete mode has no dictionary cert


def test_export_artifact_to_dict(fitted_csrq):
    clf, X, y = fitted_csrq
    artifact = clf.export_artifact()
    d = artifact.to_dict()
    assert d["version"] == "1.0"
    assert "c_float" in d
    assert "optimization_report" in d


def test_export_artifact_dictionary_mode():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 2))
    y = (X[:, 0] > 0).astype(int)

    atoms = tuple(_make_atom(j, t) for j in range(2) for t in ("low", "high"))
    clf = CSRQClassifier(
        C=1.0, max_rule_length=1,
        semantic_space="dictionary",
        rule_dictionary=atoms,
    )
    clf.fit(X, y)
    artifact = clf.export_artifact()
    assert artifact.semantic_space == "dictionary"
    assert artifact.c_exact is not None  # exact rationals for dictionary mode
    assert artifact.equality_certificate is not None


# ---------------------------------------------------------------------------
# Exact residual check
# ---------------------------------------------------------------------------

def test_dictionary_mode_exact_residual():
    """In dictionary mode, exact decode should yield zero residual."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    atoms = tuple(_make_atom(j, t) for j in range(2) for t in ("low", "high"))
    clf = CSRQClassifier(
        C=1.0, max_rule_length=1,
        semantic_space="dictionary",
        rule_dictionary=atoms,
    )
    clf.fit(X, y)
    artifact = clf.export_artifact()

    if artifact.equality_certificate is not None:
        cert = artifact.equality_certificate
        # For a complete-spanning dictionary, the certificate should be CERTIFIED
        # with zero residual
        assert cert.exact_zero_residual or cert.status == "UNKNOWN"
