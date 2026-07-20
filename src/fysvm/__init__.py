"""Fuzzy rule-space max-margin classifiers."""

from importlib.metadata import version

from fysvm.datasets import DatasetSpec, PreparedDataset, list_datasets, load_dataset
from fysvm.membership import MembershipLogisticL1, MembershipSVM
from fysvm.rule_svm import (
    FuzzyRule,
    FuzzyRuleSVM,
    RuleCondition,
    SparseMaxMarginFuzzyRuleMachine,
)

__version__ = version("fysvm")

__all__ = [
    "DatasetSpec",
    "FuzzyRule",
    "FuzzyRuleSVM",
    "MembershipLogisticL1",
    "MembershipSVM",
    "PreparedDataset",
    "RuleCondition",
    "SparseMaxMarginFuzzyRuleMachine",
    "__version__",
    "list_datasets",
    "load_dataset",
]
