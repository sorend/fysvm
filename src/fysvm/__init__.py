"""Fuzzy rule-space max-margin classifiers."""

from importlib.metadata import version

__version__ = version("fysvm")

from fysvm.rule_svm import (
    FuzzyRule,
    FuzzyRuleSVM,
    RuleCondition,
    SparseMaxMarginFuzzyRuleMachine,
)

__all__ = [
    "FuzzyRule",
    "FuzzyRuleSVM",
    "RuleCondition",
    "SparseMaxMarginFuzzyRuleMachine",
    "__version__",
]
