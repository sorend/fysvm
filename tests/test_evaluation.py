import numpy as np

from fysvm import FuzzyRuleSVM
from fysvm.datasets import DatasetSpec, PreparedDataset
from fysvm.evaluation import evaluate_classifier


def test_evaluation_harness_reports_rule_specific_metrics(tmp_path):
    rng = np.random.default_rng(3)
    negative = rng.normal(loc=(-1.0, -1.0), scale=0.2, size=(18, 2))
    positive = rng.normal(loc=(1.0, 1.0), scale=0.2, size=(18, 2))
    dataset = PreparedDataset(
        spec=DatasetSpec(
            slug="synthetic",
            name="Synthetic",
            domain="test",
            expected_samples=36,
            expected_features=2,
            task="binary",
            target="negative vs positive",
            source="test",
        ),
        X=np.vstack([negative, positive]),
        y=np.asarray(["negative"] * 18 + ["positive"] * 18),
        feature_names=["a", "b"],
        target_names=["negative", "positive"],
    )
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    np.savez_compressed(
        prepared_dir / "synthetic.npz",
        X=dataset.X,
        y=dataset.y,
        feature_names=np.asarray(dataset.feature_names),
        target_names=np.asarray(dataset.target_names),
        metadata=np.asarray('{"slug": "synthetic"}'),
    )

    def factory(prepared, random_state):
        return FuzzyRuleSVM(
            C=10.0,
            penalty="l2",
            max_rule_length=1,
            max_rules=6,
            feature_names=prepared.feature_names,
            random_state=random_state,
        )

    result = evaluate_classifier(
        factory,
        dataset_slugs=["synthetic"],
        data_dir=prepared_dir,
        output_dir=tmp_path / "out",
        n_splits=3,
    )

    summary = result.summary_metrics[0]
    assert summary["accuracy_mean"] > 0.9
    assert "rule_support_rule_count_mean" in summary
    assert summary["rule_explanation_fidelity_max_abs_error_mean"] < 1e-10
    assert (tmp_path / "out" / "metrics.csv").exists()
    assert (tmp_path / "out" / "fold_metrics.csv").exists()
