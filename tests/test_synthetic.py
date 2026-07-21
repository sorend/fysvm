import numpy as np

from fysvm.synthetic import (
    make_additive_main_effects,
    make_pairwise_fuzzy_interaction,
    make_sparse_fuzzy_rule_ground_truth,
    make_xor_interaction,
)


def test_synthetic_generators_return_binary_datasets():
    generators = [
        make_additive_main_effects,
        make_pairwise_fuzzy_interaction,
        make_xor_interaction,
        make_sparse_fuzzy_rule_ground_truth,
    ]
    for generator in generators:
        data = generator(n_samples=50, n_noise=3, random_state=1)
        assert data.X.shape[0] == 50
        assert data.X.shape[1] == len(data.feature_names)
        assert set(np.unique(data.y)) == {0, 1}
        assert data.regime
