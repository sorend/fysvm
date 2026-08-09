"""MCLTA evaluation harness.

Provides cross-validated evaluation of the Minimal Certified Linguistic
Transition Atlas on real and synthetic datasets. Produces aggregate metrics
suitable for the paper's empirical evaluation section.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

from fysvm.rule_svm import FuzzyRuleSVM, SparseMaxMarginFuzzyRuleMachine
from fysvm.transition_atlas import (
    ContextGrammar,
    MaterialityPolicy,
    TransitionAtlas,
    synthesize_transition_atlas,
    verify_transition_atlas,
)
from fysvm.transition_envelopes import (
    ContextLiteral,
    LinearDomain,
    MilpConfig,
    TransitionQuery,
)

_TERMS = ("low", "medium", "high")
_TERM_TRANSITIONS = [
    ("low", "medium"),
    ("medium", "high"),
    ("low", "high"),
]


@dataclass
class AtlasRunRecord:
    """Result of a single MCLTA evaluation run."""

    dataset_name: str
    fold_index: int
    repeat_index: int
    transition_feature: int
    transition_feature_name: str
    source_term: str
    destination_term: str
    target_class: Any
    atlas_status: str
    n_feasible_atoms: int
    n_candidates: int
    n_selected: int
    min_cardinality_lower: int | None
    min_cardinality_upper: int | None
    n_grammar_limited_atoms: int
    max_envelope_expansion: float | None
    balanced_accuracy: float
    runtime_seconds: float
    warnings: tuple[str, ...]


@dataclass
class DatasetEvaluationResult:
    """Aggregated results for one dataset across all folds and transitions."""

    dataset_name: str
    n_folds: int
    n_transitions: int
    n_runs: int
    runs: list[AtlasRunRecord]
    summary: dict[str, Any]


def _make_domain_from_fold(
    X_train: np.ndarray,
    percentile_lo: float = 1.0,
    percentile_hi: float = 99.0,
) -> LinearDomain:
    """Build a box domain from training-fold percentiles."""
    lower = np.percentile(X_train, percentile_lo, axis=0)
    upper = np.percentile(X_train, percentile_hi, axis=0)
    # Ensure non-degenerate bounds
    for i in range(len(lower)):
        if lower[i] >= upper[i]:
            eps = max(abs(lower[i]) * 1e-6, 1e-6)
            lower[i] -= eps
            upper[i] += eps
    return LinearDomain(
        lower=tuple(float(v) for v in lower),
        upper=tuple(float(v) for v in upper),
        provenance=f"training fold {percentile_lo:.0f}th-{percentile_hi:.0f}th percentile",
    )


def _make_materiality_from_fold(
    model: SparseMaxMarginFuzzyRuleMachine,
    X_train: np.ndarray,
    target_sign: int,
) -> MaterialityPolicy:
    """Compute materiality policy from training fold score distribution."""
    from fysvm.transition_envelopes import _resolve_target_sign
    scores = model.decision_function(X_train) * target_sign
    q25, q75 = float(np.percentile(scores, 25)), float(np.percentile(scores, 75))
    iqr = q75 - q25
    if iqr > 0:
        s = iqr
    else:
        lo, hi = float(np.min(scores)), float(np.max(scores))
        s = hi - lo if hi > lo else 1.0

    eps_dir = 0.05 * s
    eta_merge = 0.10
    grammar_limited_width = 0.5

    return MaterialityPolicy(
        margin_scale=s,
        direction_epsilon=eps_dir,
        merge_tolerance=eta_merge,
        grammar_limited_width=grammar_limited_width,
    )


def _make_grammar_from_fold(
    model: SparseMaxMarginFuzzyRuleMachine,
    transition_feature: int,
    max_context_features: int = 5,
    gamma: float = 0.5,
    max_clause_literals: int = 2,
) -> ContextGrammar:
    """Build a grammar from model's context features.

    Only includes features with non-degenerate triangular partitions
    (q_low < q_mid < q_high).  Features with tied anchors (e.g., integer
    ordinal variables) are excluded; the CLTE MILP rejects tied anchors
    and the proposal defers discrete support to a later extension.
    """
    ctx_feats: set[int] = set()
    for k, beta in enumerate(model.coef_):
        if beta != 0.0:
            for cond in model.rules_[k].conditions:
                if cond.feature != transition_feature:
                    p = model.partitions_[cond.feature]
                    if p.low < p.medium < p.high:  # non-degenerate only
                        ctx_feats.add(cond.feature)

    sorted_feats = sorted(ctx_feats)[:max_context_features]
    bins_by_feature = tuple(
        (
            ContextLiteral(fi, "low", gamma),
            ContextLiteral(fi, "medium", gamma),
            ContextLiteral(fi, "high", gamma),
        )
        for fi in sorted_feats
    )
    return ContextGrammar(
        feature_indices=tuple(sorted_feats),
        bins_by_feature=bins_by_feature,
        max_clause_literals=max_clause_literals,
    )


def evaluate_dataset_clte(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    model_params: dict | None = None,
    n_splits: int = 5,
    n_repeats: int = 1,
    source_alpha: float = 0.75,
    destination_alpha: float = 0.75,
    grammar_gamma: float = 0.5,
    max_context_features: int = 5,
    max_clause_literals: int = 2,
    envelope_solver: MilpConfig | None = None,
    set_cover_time_limit: float = 60.0,
    random_state: int = 42,
    continuous_feature_mask: np.ndarray | None = None,
) -> DatasetEvaluationResult:
    """Run cross-validated MCLTA evaluation on a dataset.

    Parameters
    ----------
    X, y:
        Feature matrix and labels (any format; will be type-checked).
    dataset_name:
        Name for logging.
    model_params:
        Additional FuzzyRuleSVM parameters.
    n_splits:
        Number of CV folds.
    n_repeats:
        Number of CV repetitions (> 1 for repeated K-fold).
    source_alpha, destination_alpha:
        Membership thresholds.
    grammar_gamma:
        Grammar bin threshold.
    max_context_features:
        Cap on grammar features.
    max_clause_literals:
        Cap on clause size.
    envelope_solver:
        MILP configuration.
    set_cover_time_limit:
        Time budget for set cover.
    random_state:
        Seed for reproducibility.
    continuous_feature_mask:
        Boolean mask (n_features,) for which features to consider. None = all.

    Returns
    -------
    DatasetEvaluationResult
    """
    if model_params is None:
        model_params = {}
    if envelope_solver is None:
        envelope_solver = MilpConfig(time_limit_seconds=30.0)

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    n_samples, n_features = X_arr.shape

    if continuous_feature_mask is not None:
        cont_mask = np.asarray(continuous_feature_mask, dtype=bool)
    else:
        cont_mask = np.ones(n_features, dtype=bool)

    # Default model params
    default_params = {
        "and_operator": "min",
        "penalty": "l1",
        "C": 1.0,
        "max_rule_length": 2,
        "max_rules": 64,
        "min_rule_coverage": 0.01,
        "rule_length_penalty": 0.35,
        "feature_screening": "anova",
        "screen_top_k": 4,
        "class_weight": "balanced",
        "partition_quantiles": (0.05, 0.5, 0.95),
        "rule_generation": "enumeration",
        "max_iter": 20000,
        "tol": 1e-6,
    }
    default_params.update(model_params)

    # CV setup
    if n_repeats == 1:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(cv.split(X_arr, y_arr))
        repeat_indices = [0] * len(splits)
    else:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        splits = list(cv.split(X_arr, y_arr))
        repeat_indices = [i // n_splits for i in range(len(splits))]

    runs: list[AtlasRunRecord] = []

    for fold_num, ((train_idx, test_idx), repeat_idx) in enumerate(
        zip(splits, repeat_indices)
    ):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        # Restrict to continuous features
        X_train_cont = X_train[:, cont_mask]
        X_test_cont = X_test[:, cont_mask]

        # Impute NaN with training-fold column medians (inside the fold, as required)
        if np.any(np.isnan(X_train_cont)):
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="median")
            X_train_cont = imputer.fit_transform(X_train_cont)
            X_test_cont = imputer.transform(X_test_cont)

        fold_seed = random_state + fold_num * 37
        params_with_seed = dict(default_params)
        params_with_seed["random_state"] = fold_seed

        model = FuzzyRuleSVM(**params_with_seed)
        model.fit(X_train_cont, y_train)

        # Balanced accuracy
        from sklearn.metrics import balanced_accuracy_score
        y_pred = model.predict(X_test_cont)
        bacc = float(balanced_accuracy_score(y_test, y_pred))

        pos_class = model.classes_[1]

        # Build domain in SCREENED coordinate space.
        # The MILP uses domain.lower[screened_feat_idx], so the domain must
        # have n_screened dimensions (one per screened feature), NOT n_cont
        # dimensions (which would index original features).
        sel = model.selected_feature_indices_
        X_train_screened = X_train_cont[:, sel]
        dom = _make_domain_from_fold(X_train_screened)

        # Materiality from training fold
        from fysvm.transition_envelopes import _resolve_target_sign
        try:
            target_sign = _resolve_target_sign(model, pos_class)
        except ValueError:
            target_sign = 1

        materiality = _make_materiality_from_fold(model, X_train_cont, target_sign)

        # Evaluate each transition feature
        n_screened = len(model.partitions_)
        for feat_idx in range(n_screened):
            feat_name = str(model.feature_names_in_[feat_idx])

            # Check this feature is continuous (per mask)
            # In screened space, feature index corresponds to selected_feature_indices_[feat_idx]
            orig_feat = int(model.selected_feature_indices_[feat_idx])
            if orig_feat < len(cont_mask) and not cont_mask[orig_feat]:
                continue

            # Skip transition features with degenerate partitions (tied anchors).
            p_j = model.partitions_[feat_idx]
            if not (p_j.low < p_j.medium < p_j.high):
                continue

            # Skip if any nonzero rule containing this transition feature also
            # contains a co-antecedent with a degenerate partition.  The MILP
            # rejects tied relevant anchors; such rules make the query INVALID.
            has_degenerate_co_antecedent = False
            for k, beta in enumerate(model.coef_):
                if beta == 0.0:
                    continue
                conds = model.rules_[k].conditions
                feat_ids = [c.feature for c in conds]
                if feat_idx not in feat_ids:
                    continue
                for c in conds:
                    if c.feature == feat_idx:
                        continue
                    p_c = model.partitions_[c.feature]
                    if not (p_c.low < p_c.medium < p_c.high):
                        has_degenerate_co_antecedent = True
                        break
                if has_degenerate_co_antecedent:
                    break
            if has_degenerate_co_antecedent:
                continue

            grammar = _make_grammar_from_fold(
                model, feat_idx,
                max_context_features=max_context_features,
                gamma=grammar_gamma,
                max_clause_literals=max_clause_literals,
            )

            # Skip if no context features (unconditional only)
            if len(grammar.feature_indices) == 0:
                continue

            # Evaluate three standard transitions
            for src_term, dst_term in _TERM_TRANSITIONS:
                t_run_start = time.perf_counter()

                query = TransitionQuery(
                    feature_index=feat_idx,
                    source_term=src_term,  # type: ignore[arg-type]
                    destination_term=dst_term,  # type: ignore[arg-type]
                    source_alpha=source_alpha,
                    destination_alpha=destination_alpha,
                    target_class=pos_class,
                    domain=dom,
                    enforce_term_order=True,
                )

                try:
                    atlas = synthesize_transition_atlas(
                        model, query, grammar, materiality,
                        envelope_solver=envelope_solver,
                        set_cover_time_limit_seconds=set_cover_time_limit,
                    )
                except Exception as exc:
                    atlas = None
                    atlas_status = f"ERROR: {exc}"
                else:
                    atlas_status = atlas.status

                run_time = time.perf_counter() - t_run_start

                # Aggregate stats
                if atlas is not None and atlas.status != "INVALID":
                    n_feasible = atlas.feasible_atom_count
                    n_cands = len(atlas.candidates)
                    n_sel = len(atlas.selected_candidate_indices)
                    n_gl = sum(
                        1 for a in atlas.atoms
                        if a.grammar_limited_status == "DEFINITELY_GRAMMAR_LIMITED"
                    )
                    n_lo = atlas.min_cardinality_lower
                    n_up = atlas.min_cardinality_upper
                    warnings = atlas.warnings

                    # Max envelope expansion
                    max_exp: float | None = None
                    for ci in atlas.selected_candidate_indices:
                        if ci < len(atlas.candidates):
                            cand = atlas.candidates[ci]
                            for ai in cand.extension_atom_indices:
                                atom = atlas.atoms[ai]
                                # Compute expansion
                                if all(
                                    x is not None for x in [
                                        atom.envelope.dual_lower, atom.envelope.primal_lower,
                                        atom.envelope.primal_upper, atom.envelope.dual_upper,
                                        cand.derived_dual_lower, cand.derived_primal_lower,
                                        cand.derived_primal_upper, cand.derived_dual_upper,
                                    ]
                                ):
                                    from fysvm.transition_atlas import _normalised_expansion
                                    _, ell_hi = _normalised_expansion(
                                        atom.envelope.dual_lower, atom.envelope.primal_lower,
                                        atom.envelope.primal_upper, atom.envelope.dual_upper,
                                        cand.derived_dual_lower, cand.derived_primal_lower,
                                        cand.derived_primal_upper, cand.derived_dual_upper,
                                        materiality.margin_scale,
                                    )
                                    max_exp = max(max_exp or 0.0, ell_hi)
                else:
                    n_feasible = 0
                    n_cands = 0
                    n_sel = 0
                    n_gl = 0
                    n_lo = None
                    n_up = None
                    max_exp = None
                    warnings = ()

                runs.append(AtlasRunRecord(
                    dataset_name=dataset_name,
                    fold_index=fold_num % n_splits,
                    repeat_index=repeat_idx,
                    transition_feature=feat_idx,
                    transition_feature_name=feat_name,
                    source_term=src_term,
                    destination_term=dst_term,
                    target_class=pos_class,
                    atlas_status=atlas_status,
                    n_feasible_atoms=n_feasible,
                    n_candidates=n_cands,
                    n_selected=n_sel,
                    min_cardinality_lower=n_lo,
                    min_cardinality_upper=n_up,
                    n_grammar_limited_atoms=n_gl,
                    max_envelope_expansion=max_exp,
                    balanced_accuracy=bacc,
                    runtime_seconds=run_time,
                    warnings=tuple(warnings),
                ))

    # Build summary
    total_runs = len(runs)
    completed = [r for r in runs if not r.atlas_status.startswith("ERROR") and r.atlas_status not in ("INVALID",)]
    min_cert = [r for r in completed if r.atlas_status == "MINIMUM_SOLVER_CERTIFIED"]
    near_min_cert = [r for r in completed if r.atlas_status == "NEAR_MINIMUM_SOLVER_CERTIFIED"]
    infeasible = [r for r in completed if r.atlas_status == "INFEASIBLE_TRANSITION"]

    summary = {
        "total_runs": total_runs,
        "completed_runs": len(completed),
        "minimum_certified_count": len(min_cert),
        "near_minimum_certified_count": len(near_min_cert),
        "infeasible_transition_count": len(infeasible),
        "completion_rate": len(completed) / total_runs if total_runs > 0 else 0.0,
        "min_cert_rate": len(min_cert) / total_runs if total_runs > 0 else 0.0,
        "median_n_selected": float(np.median([r.n_selected for r in completed])) if completed else 0.0,
        "median_n_atoms": float(np.median([r.n_feasible_atoms for r in completed])) if completed else 0.0,
        "median_runtime_seconds": float(np.median([r.runtime_seconds for r in runs])) if runs else 0.0,
        "mean_balanced_accuracy": float(np.mean([r.balanced_accuracy for r in runs])) if runs else 0.0,
    }

    return DatasetEvaluationResult(
        dataset_name=dataset_name,
        n_folds=n_splits,
        n_transitions=len(_TERM_TRANSITIONS),
        n_runs=total_runs,
        runs=runs,
        summary=summary,
    )


def results_to_dataframe(result: DatasetEvaluationResult) -> pd.DataFrame:
    """Convert evaluation results to a pandas DataFrame."""
    rows = []
    for r in result.runs:
        rows.append({
            "dataset": r.dataset_name,
            "fold": r.fold_index,
            "repeat": r.repeat_index,
            "feature": r.transition_feature_name,
            "src_term": r.source_term,
            "dst_term": r.destination_term,
            "status": r.atlas_status,
            "n_atoms": r.n_feasible_atoms,
            "n_candidates": r.n_candidates,
            "n_selected": r.n_selected,
            "N_L": r.min_cardinality_lower,
            "N_U": r.min_cardinality_upper,
            "n_grammar_limited": r.n_grammar_limited_atoms,
            "max_expansion": r.max_envelope_expansion,
            "balanced_accuracy": r.balanced_accuracy,
            "runtime_s": r.runtime_seconds,
        })
    return pd.DataFrame(rows)


def summarise_results(results: list[DatasetEvaluationResult]) -> pd.DataFrame:
    """Build per-dataset summary statistics table."""
    rows = []
    for res in results:
        completed = [r for r in res.runs if r.atlas_status not in ("INVALID",) and not r.atlas_status.startswith("ERROR")]
        min_cert = [r for r in completed if r.atlas_status == "MINIMUM_SOLVER_CERTIFIED"]
        near_cert = [r for r in completed if r.atlas_status in ("MINIMUM_SOLVER_CERTIFIED", "NEAR_MINIMUM_SOLVER_CERTIFIED")]

        n_completed = len(completed)
        rows.append({
            "dataset": res.dataset_name,
            "n_runs": res.n_runs,
            "completion_rate": n_completed / res.n_runs if res.n_runs > 0 else 0.0,
            "min_cert_rate": len(min_cert) / n_completed if n_completed > 0 else 0.0,
            "near_min_cert_rate": len(near_cert) / n_completed if n_completed > 0 else 0.0,
            "median_n_selected": float(np.median([r.n_selected for r in completed])) if completed else np.nan,
            "median_n_atoms": float(np.median([r.n_feasible_atoms for r in completed])) if completed else np.nan,
            "median_runtime": float(np.median([r.runtime_seconds for r in completed])) if completed else np.nan,
            "mean_balanced_accuracy": float(np.mean([r.balanced_accuracy for r in res.runs])) if res.runs else np.nan,
        })
    return pd.DataFrame(rows)
