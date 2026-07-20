import numpy as np

from fysvm.datasets import DATASET_SPECS, load_dataset, prepare_dataset


def test_registry_contains_requested_twenty_datasets():
    slugs = {spec.slug for spec in DATASET_SPECS}

    assert len(slugs) == 20
    assert "breast_cancer_diagnostic" in slugs
    assert "parkinsons_disease_classification" in slugs
    assert "digits" in slugs


def test_prepare_and_load_builtin_dataset(tmp_path):
    output_path = prepare_dataset("iris", tmp_path)
    dataset = load_dataset("iris", tmp_path)

    assert output_path.exists()
    assert dataset.X.shape == (150, 4)
    assert len(np.unique(dataset.y)) == 3
    assert dataset.feature_names
    assert dataset.target_names
