from dataclasses import replace
from fractions import Fraction as F
from itertools import combinations, product
import unittest

import numpy as np

from sklearn.exceptions import NotFittedError

from fysvm import DirectionalExplainer, FuzzyRule, FuzzyRuleSVM, RuleCondition
from fysvm.rule_svm import _FuzzyPartition, _TERM_NAMES


def model(atoms, dimensions=3, selected=None, input_count=None, *, operator="product"):
    if selected is None:
        selected = tuple(range(dimensions))
    fitted = FuzzyRuleSVM(and_operator=operator)
    fitted.n_features_in_ = input_count or dimensions
    fitted.selected_feature_indices_ = selected
    fitted.feature_names_in_ = [f"x{i}" for i in selected]
    fitted.partitions_ = [_FuzzyPartition(-1.0, 0.0, 1.0) for _ in selected]
    fitted.coef_ = [weight for weight, _ in atoms]
    fitted.rules_ = [
        FuzzyRule(tuple(RuleCondition(j, _TERM_NAMES[t]) for j, t in rule))
        for _, rule in atoms
    ]
    return fitted


def reference_membership(x, term):
    if term == 0:
        return min(1, max(0, -x))
    if term == 1:
        return max(0, 1 - abs(x))
    return min(1, max(0, x))


def reference_effect(atoms, x, step):
    def score(z):
        return sum(w * np.prod([reference_membership(z[j], t) for j, t in rule])
                   for w, rule in atoms)
    return score([step, *x[1:]]) - score([step - 1, *x[1:]])


class CertificatesTest(unittest.TestCase):
    def test_shortest_can_require_two_and_ties_are_stable(self):
        # On low -> medium, delta = -5 + 6 H(x1) + 6 H(x2).
        fitted = model([(-5, [(0, 1)]), (6, [(0, 1), (1, 2)]), (6, [(0, 1), (2, 2)])])
        result = DirectionalExplainer(fitted).explain([0, 1, 1], 0, 0)
        self.assertEqual(result.status, "certified")
        self.assertEqual([c.feature for c in result.conditions], [1, 2])
        self.assertEqual(result.conditions[0].lower, F(1, 2))
        self.assertIsNone(result.conditions[0].upper)
        self.assertEqual((result.point_effect, result.bound), (7, 1))

    def test_exact_tolerance_and_zero(self):
        fitted = model([(0.5, [(0, 1)])])
        self.assertEqual(DirectionalExplainer(fitted, F(1, 2)).explain([0, 0, 0], 0, 0).status, "zero")
        self.assertEqual(DirectionalExplainer(fitted, F(1, 2) - F(1, 10**30))
                         .explain([0, 0, 0], 0, 0).bound, F(1, 2))

    def test_unresolved_at_closed_cut_edge(self):
        fitted = model([(-1, [(0, 1)]), (2, [(0, 1), (1, 2)])])
        result = DirectionalExplainer(fitted).explain([0, 0.9, 0], 0, 0)
        self.assertEqual((result.status, result.available_bound), ("unresolved", 0))
        tied = DirectionalExplainer(fitted).explain([0, -0.5, 0], 0, 0)
        self.assertEqual([c.term for c in tied.conditions], ["low"])

    def test_canceled_interaction_and_screened_indices(self):
        atoms = [(2, [(0, 1), (1, t)]) for t in range(3)]
        fitted = model(atoms, selected=(1, 3), input_count=5)
        explainer = DirectionalExplainer(fitted)
        self.assertEqual(explainer.involved_features(1, 0), ())
        result = explainer.explain([100, 1, 100, -100, 100], 1, 0)
        self.assertEqual((result.feature_name, result.conditions, result.bound), ("x1", (), 2))
        with self.assertRaises(ValueError):
            explainer.explain([1, 2, 3, 4, 5], 0, 0)

    def test_unsupported_grammar_is_rejected(self):
        for atoms, operator in (
            ([(1, [(0, 0), (1, 0), (2, 0)])], "product"),
            ([(1, [(0, 0), (0, 1)])], "product"),
            ([(1, [(0, 0)])], "min"),
        ):
            with self.assertRaises(ValueError):
                DirectionalExplainer(model(atoms, operator=operator))
        fitted = model([])
        fitted.partitions_[1] = replace(fitted.partitions_[1], low=0.0)
        with self.assertRaises(ValueError):
            DirectionalExplainer(fitted)

    def test_unfitted_estimator_is_rejected(self):
        with self.assertRaises(NotFittedError):
            DirectionalExplainer(FuzzyRuleSVM(and_operator="product"))

    def test_against_exhaustive_box_vertices_and_all_subsets(self):
        rng = np.random.default_rng(234)
        tolerance = F(1, 10)
        grid = tuple(F(i, 2) for i in range(-2, 3))
        for _ in range(75):
            d = int(rng.integers(2, 6))
            atoms = [(int(rng.integers(-4, 5)), [(0, term)]) for term in range(3)]
            for pair in combinations(range(d), 2):
                # Includes cycles and terms that exclude the queried feature.
                for _ in range(3):
                    terms = rng.integers(0, 3, 2)
                    atoms.append((int(rng.integers(-5, 6)), list(zip(pair, terms))))
            case = [F(int(rng.integers(-6, 7)), 4) for _ in range(d)]
            explainer = DirectionalExplainer(model(atoms, d), tolerance)
            winner = {
                j: max(range(3), key=lambda t: reference_membership(case[j], t))
                for j in range(1, d)
            }
            for step in (0, 1):
                cert = explainer.explain(case, 0, step)
                point = reference_effect(atoms, case, step)
                self.assertEqual(cert.point_effect, point)
                if abs(point) <= tolerance:
                    self.assertEqual(cert.status, "zero")
                    continue
                sign = 1 if point > 0 else -1
                vertices = [(values, sign * reference_effect(atoms, [0, *values], step))
                            for values in product(grid, repeat=d - 1)]

                def bound(subset):
                    return min(delta for values, delta in vertices
                               if all(reference_membership(values[j - 1], winner[j]) >= F(1, 2)
                                      for j in subset))

                feasible = {}
                for length in range(d):
                    for subset in combinations(range(1, d), length):
                        worst = bound(subset)
                        if worst > tolerance:
                            feasible[subset] = worst
                self.assertEqual(cert.available_bound, bound(range(1, d)))
                if not feasible:
                    self.assertEqual(cert.status, "unresolved")
                else:
                    chosen = tuple(sorted(c.feature for c in cert.conditions))
                    self.assertEqual(cert.status, "certified")
                    self.assertEqual(cert.bound, feasible[chosen])
                    self.assertEqual(len(chosen), min(map(len, feasible)))
                    self.assertEqual(cert.bound, max(v for k, v in feasible.items() if len(k) == len(chosen)))


class FittedModelTest(unittest.TestCase):
    def test_endpoint_difference_and_interpolation_with_feature_screening(self):
        rng = np.random.default_rng(13)
        X = rng.uniform(-1, 1, (220, 6))
        y = (2 * X[:, 0] * X[:, 1] + X[:, 2] > 0).astype(int)
        common = dict(and_operator="product", max_rules=128, random_state=13)
        for settings in ({}, {"feature_screening": "anova", "screen_top_k": 3}):
            clf = FuzzyRuleSVM(**common, **settings).fit(X, y)
            explainer = DirectionalExplainer(clf)
            case = X[2]
            certificates = explainer.explain_case(case)
            self.assertEqual(
                [(cert.feature, cert.transition) for cert in certificates],
                list(product(clf.selected_feature_indices_, range(2))),
            )
            for cert in certificates:
                samples = np.tile(case, (3, 1))
                a, b = map(float, cert.segment)
                samples[:, cert.feature] = [a, (a + b) / 2, b]
                fa, fm, fb = clf.decision_function(samples)
                self.assertAlmostEqual(fb - fa, float(cert.point_effect), places=12)
                self.assertAlmostEqual((fa + fb) / 2, fm, places=12)
