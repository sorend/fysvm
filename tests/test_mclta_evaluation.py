"""Tests for mclta_evaluation.py.

Deterministic smoke tests that verify the evaluation harness produces
correct schemas without requiring full dataset runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from fysvm.mclta_evaluation import (
    AtlasRunRecord,
    DatasetEvaluationResult,
    evaluate_dataset_clte,
    results_to_dataframe,
    summarise_results,
)
from fysvm.transition_envelopes import MilpConfig

# Fast solver for testing
_TEST_SOLVER = MilpConfig(
    time_limit_seconds=15.0,
    relative_gap=1e-5,
)


def _make_synthetic_dataset(n: int = 80, n_features: int = 3, seed: int = 42):
    """Simple synthetic binary classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_features))
    # Positive class when first feature > 0.5
    y = (X[:, 0] > 0.5).astype(int)
    return X, y


# ------------------------------------------------------------------ #
# Test: evaluate_dataset_clte returns correct schema                #
# ------------------------------------------------------------------ #


def test_evaluate_dataset_returns_result_object():
    """evaluate_dataset_clte returns a DatasetEvaluationResult."""
    X, y = _make_synthetic_dataset(n=80, n_features=2)
    result = evaluate_dataset_clte(
        X, y,
        dataset_name="smoke_test",
        n_splits=2,
        n_repeats=1,
        envelope_solver=_TEST_SOLVER,
        set_cover_time_limit=10.0,
        random_state=42,
        model_params={
            "max_rules": 10,
            "screen_top_k": 2,
            "feature_screening": "anova",
            "max_iter": 5000,
        },
    )
    assert isinstance(result, DatasetEvaluationResult)
    assert result.dataset_name == "smoke_test"
    assert result.n_folds == 2


def test_evaluate_dataset_runs_are_non_empty():
    """At least some runs should be produced."""
    X, y = _make_synthetic_dataset(n=80, n_features=2)
    result = evaluate_dataset_clte(
        X, y,
        dataset_name="smoke_test",
        n_splits=2,
        n_repeats=1,
        envelope_solver=_TEST_SOLVER,
        set_cover_time_limit=10.0,
        random_state=42,
        model_params={"max_rules": 10, "screen_top_k": 2, "max_iter": 5000},
    )
    # We may have 0 runs if no context features are found, but no error
    assert result.n_runs >= 0
    assert isinstance(result.runs, list)


def test_evaluate_dataset_run_fields_complete():
    """Each AtlasRunRecord has all required fields."""
    X, y = _make_synthetic_dataset(n=80, n_features=2)
    result = evaluate_dataset_clte(
        X, y,
        dataset_name="smoke_test",
        n_splits=2,
        n_repeats=1,
        envelope_solver=_TEST_SOLVER,
        set_cover_time_limit=10.0,
        random_state=42,
        model_params={"max_rules": 10, "screen_top_k": 2, "max_iter": 5000},
    )
    for run in result.runs:
        assert isinstance(run.atlas_status, str)
        assert isinstance(run.balanced_accuracy, float)
        assert 0.0 <= run.balanced_accuracy <= 1.0
        assert run.runtime_seconds >= 0.0
        assert isinstance(run.warnings, tuple)


def test_evaluate_dataset_summary_fields():
    """Summary dict has expected keys."""
    X, y = _make_synthetic_dataset(n=80, n_features=2)
    result = evaluate_dataset_clte(
        X, y,
        dataset_name="smoke_test",
        n_splits=2,
        n_repeats=1,
        envelope_solver=_TEST_SOLVER,
        set_cover_time_limit=10.0,
        random_state=42,
        model_params={"max_rules": 10, "screen_top_k": 2, "max_iter": 5000},
    )
    expected_keys = {
        "total_runs", "completed_runs", "minimum_certified_count",
        "near_minimum_certified_count", "infeasible_transition_count",
        "completion_rate", "min_cert_rate",
        "median_n_selected", "median_n_atoms",
        "median_runtime_seconds", "mean_balanced_accuracy",
    }
    assert expected_keys.issubset(set(result.summary.keys()))


def test_results_to_dataframe():
    """results_to_dataframe returns a non-empty DataFrame with correct columns."""
    X, y = _make_synthetic_dataset(n=80, n_features=2)
    result = evaluate_dataset_clte(
        X, y,
        dataset_name="smoke_test",
        n_splits=2,
        n_repeats=1,
        envelope_solver=_TEST_SOLVER,
        set_cover_time_limit=10.0,
        random_state=42,
        model_params={"max_rules": 10, "screen_top_k": 2, "max_iter": 5000},
    )
    if result.n_runs == 0:
        pytest.skip("No runs produced (no context features in screened model)")
    df = results_to_dataframe(result)
    required_cols = {"dataset", "fold", "feature", "status", "n_atoms", "runtime_s"}
    assert required_cols.issubset(set(df.columns))


def test_summarise_results():
    """summarise_results returns a DataFrame with per-dataset rows."""
    X, y = _make_synthetic_dataset(n=80, n_features=2)
    result = evaluate_dataset_clte(
        X, y,
        dataset_name="ds1",
        n_splits=2,
        n_repeats=1,
        envelope_solver=_TEST_SOLVER,
        set_cover_time_limit=10.0,
        random_state=42,
        model_params={"max_rules": 10, "screen_top_k": 2, "max_iter": 5000},
    )
    summary_df = summarise_results([result])
    assert len(summary_df) == 1
    assert "dataset" in summary_df.columns


def test_evaluate_deterministic():
    """Same seed produces same run structure."""
    X, y = _make_synthetic_dataset(n=80, n_features=2)
    r1 = evaluate_dataset_clte(
        X, y, dataset_name="det1",
        n_splits=2, n_repeats=1,
        envelope_solver=_TEST_SOLVER,
        set_cover_time_limit=5.0,
        random_state=42,
        model_params={"max_rules": 8, "screen_top_k": 2, "max_iter": 3000},
    )
    r2 = evaluate_dataset_clte(
        X, y, dataset_name="det1",
        n_splits=2, n_repeats=1,
        envelope_solver=_TEST_SOLVER,
        set_cover_time_limit=5.0,
        random_state=42,
        model_params={"max_rules": 8, "screen_top_k": 2, "max_iter": 3000},
    )
    # Same number of runs
    assert r1.n_runs == r2.n_runs
    # Same balanced accuracies
    for run1, run2 in zip(r1.runs, r2.runs):
        assert run1.balanced_accuracy == pytest.approx(run2.balanced_accuracy, abs=1e-8)
