"""Fuzzy rule-space max-margin classifiers."""

from importlib.metadata import version

from fysvm.atomic import HullCertificationResult, QuotientAtomicFuzzySVM, certify_hull_equality
from fysvm.csrq import CSRQClassifier
from fysvm.csrq_artifacts import (
    CSRQArtifact,
    DetailedSemanticEqualityCertificate,
    FloatingPointAudit,
    OptimizationReport,
)
from fysvm.datasets import DatasetSpec, PreparedDataset, list_datasets, load_dataset
from fysvm.membership import MembershipLogisticL1, MembershipSVM
from fysvm.quotient import (
    CanonicalBasis,
    CanonicalLiteral,
    CanonicalMonomial,
    RuleAtom,
    SemanticEqualityCertificate,
    SemanticMap,
    build_semantic_map,
    canonical_basis,
    canonical_dimension,
)
from fysvm.rule_svm import (
    FuzzyRule,
    FuzzyRuleSVM,
    RuleCondition,
    SparseMaxMarginFuzzyRuleMachine,
)

__version__ = version("fysvm")

__all__ = [
    "CanonicalBasis",
    "CanonicalLiteral",
    "CanonicalMonomial",
    "CSRQArtifact",
    "CSRQClassifier",
    "DatasetSpec",
    "DetailedSemanticEqualityCertificate",
    "FloatingPointAudit",
    "FuzzyRule",
    "FuzzyRuleSVM",
    "HullCertificationResult",
    "MembershipLogisticL1",
    "MembershipSVM",
    "OptimizationReport",
    "PreparedDataset",
    "QuotientAtomicFuzzySVM",
    "RuleAtom",
    "RuleCondition",
    "SemanticEqualityCertificate",
    "SemanticMap",
    "SparseMaxMarginFuzzyRuleMachine",
    "__version__",
    "build_semantic_map",
    "canonical_basis",
    "canonical_dimension",
    "certify_hull_equality",
    "list_datasets",
    "load_dataset",
]
