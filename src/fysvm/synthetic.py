"""Synthetic regimes for fuzzy-rule representation studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticDataset:
    """A generated binary classification dataset with regime metadata."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    regime: str


def make_additive_main_effects(
    *,
    n_samples: int = 600,
    n_noise: int = 0,
    random_state: int = 0,
) -> SyntheticDataset:
    rng = np.random.default_rng(random_state)
    X_signal = rng.normal(size=(n_samples, 3))
    score = 1.2 * X_signal[:, 0] - 0.9 * X_signal[:, 1] + 0.6 * X_signal[:, 2]
    return _finish(X_signal, score, n_noise, rng, "additive_main_effects")


def make_pairwise_fuzzy_interaction(
    *,
    n_samples: int = 600,
    n_noise: int = 0,
    random_state: int = 0,
) -> SyntheticDataset:
    rng = np.random.default_rng(random_state)
    X_signal = rng.normal(size=(n_samples, 4))
    high_a = _linear_up(X_signal[:, 0], -0.2, 0.8)
    high_b = _linear_up(X_signal[:, 1], -0.2, 0.8)
    low_c = _linear_down(X_signal[:, 2], -0.8, 0.2)
    score = 2.0 * np.minimum(high_a, high_b) - 1.2 * low_c + 0.3 * X_signal[:, 3]
    return _finish(X_signal, score, n_noise, rng, "pairwise_fuzzy_interaction")


def make_xor_interaction(
    *,
    n_samples: int = 600,
    n_noise: int = 0,
    random_state: int = 0,
) -> SyntheticDataset:
    rng = np.random.default_rng(random_state)
    X_signal = rng.normal(size=(n_samples, 2))
    score = np.where(X_signal[:, 0] * X_signal[:, 1] >= 0, 1.0, -1.0)
    score += 0.15 * rng.normal(size=n_samples)
    return _finish(X_signal, score, n_noise, rng, "xor_interaction")


def make_sparse_fuzzy_rule_ground_truth(
    *,
    n_samples: int = 600,
    n_noise: int = 0,
    random_state: int = 0,
) -> SyntheticDataset:
    rng = np.random.default_rng(random_state)
    X_signal = rng.normal(size=(n_samples, 5))
    rule1 = np.minimum(_linear_up(X_signal[:, 0], -0.3, 0.7), _linear_up(X_signal[:, 1], 0.0, 1.0))
    rule2 = np.minimum(_linear_down(X_signal[:, 2], -1.0, 0.0), _linear_up(X_signal[:, 3], -0.5, 0.5))
    rule3 = _linear_down(X_signal[:, 4], -0.2, 0.4)
    score = 2.2 * rule1 - 1.6 * rule2 + 1.0 * rule3 - 0.7
    return _finish(X_signal, score, n_noise, rng, "sparse_fuzzy_rule_ground_truth")


def add_irrelevant_noise_features(
    X: np.ndarray,
    *,
    n_noise: int,
    random_state: int = 0,
) -> np.ndarray:
    """Append independent Gaussian noise features."""

    if n_noise <= 0:
        return X
    rng = np.random.default_rng(random_state)
    return np.column_stack([X, rng.normal(size=(X.shape[0], n_noise))])


def _finish(
    X_signal: np.ndarray,
    score: np.ndarray,
    n_noise: int,
    rng: np.random.Generator,
    regime: str,
) -> SyntheticDataset:
    score = score + 0.25 * rng.normal(size=score.shape[0])
    y = np.where(score >= np.median(score), 1, 0)
    X = np.column_stack([X_signal, rng.normal(size=(X_signal.shape[0], n_noise))]) if n_noise else X_signal
    names = [f"x{index}" for index in range(X.shape[1])]
    return SyntheticDataset(X=X.astype(float), y=y.astype(int), feature_names=names, regime=regime)


def _linear_down(values: np.ndarray, start: float, end: float) -> np.ndarray:
    if end <= start:
        return (values <= start).astype(float)
    return np.clip((end - values) / (end - start), 0.0, 1.0)


def _linear_up(values: np.ndarray, start: float, end: float) -> np.ndarray:
    if end <= start:
        return (values >= end).astype(float)
    return np.clip((values - start) / (end - start), 0.0, 1.0)
