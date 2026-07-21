import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import (
    ablation_membership_svm,
    compare_modern_baselines,
    compare_recommendations,
    generate_tables,
)


def _dataset(n_features: int) -> SimpleNamespace:
    return SimpleNamespace(X=np.zeros((4, n_features)))


@pytest.mark.parametrize(
    "grid_builder",
    [
        compare_recommendations._fuzzy_rule_svm_grid,
        ablation_membership_svm._fuzzy_rule_svm_grid,
    ],
)
def test_primary_grids_use_fixed_rule_policy(grid_builder) -> None:
    low_dimensional = grid_builder(_dataset(8))
    assert {row["max_rule_length"] for row in low_dimensional} == {2}
    assert {row["max_rules"] for row in low_dimensional} == {24}

    high_dimensional = grid_builder(_dataset(754))
    assert len(high_dimensional) == 6
    assert {row["max_rule_length"] for row in high_dimensional} == {1}
    assert {row["max_rules"] for row in high_dimensional} == {256}
    assert all("feature_screening" not in row for row in high_dimensional)


def test_membership_ablation_holm_correction_is_monotone() -> None:
    corrected = ablation_membership_svm._holm_bonferroni([0.04, 0.01, 0.03])
    assert corrected == pytest.approx([0.06, 0.03, 0.06])


def test_balanced_ebm_fit_does_not_fall_back_to_unweighted() -> None:
    class Model:
        pass

    class Pipeline:
        named_steps = {"model": Model()}

        def fit(self, X, y, **kwargs):
            if kwargs:
                raise TypeError("sample weights unsupported")
            raise AssertionError("unweighted fallback must not run")

    with pytest.raises(RuntimeError, match="sample_weight support"):
        compare_modern_baselines._fit_pipeline(
            Pipeline(),
            np.zeros((4, 1)),
            np.array([0, 0, 1, 1]),
            use_balanced_sample_weight=True,
        )


def test_generated_table_rows_have_latex_line_breaks(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "model,balanced_accuracy_mean,f1_macro_mean\nExample,0.8,0.7\n",
        encoding="utf-8",
    )
    table = generate_tables._overall_table(metrics, "tab:test", "Test.")
    assert "Example & 0.800 & 0.700 " + r"\\" in table
