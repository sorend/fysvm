"""Model-selection policy for fuzzy rule-space operating regimes."""

from __future__ import annotations

from collections.abc import Mapping


def recommend_model(
    *,
    n_features: int,
    retention_ratio: float | None = None,
    inner_cv_scores: Mapping[str, float] | None = None,
) -> str:
    """Recommend a model family from dimensionality and inner-CV diagnostics."""

    if inner_cv_scores:
        best_model = max(inner_cv_scores.items(), key=lambda item: item[1])[0]
        best_score = inner_cv_scores[best_model]
        fuzzy_score = inner_cv_scores.get("FuzzyRuleSVM")
        if fuzzy_score is None or best_score - fuzzy_score > 0.01:
            return best_model

    if n_features <= 32:
        return "FuzzyRuleSVM length-2"
    if n_features <= 100:
        return "MembershipSVM or length-1 FuzzyRuleSVM"
    if retention_ratio is not None and retention_ratio < 0.25:
        return "MembershipSVM or EBM"
    return "screened length-2 FuzzyRuleSVM"
