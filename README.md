# fysvm

Intrinsically interpretable fuzzy SVM-style classifiers built on scikit-learn.

`fysvm` trains linear max-margin classifiers over fuzzy rule activations instead
of raw feature coordinates. Every learned dimension is a human-readable
linguistic rule such as:

```
IF glucose is high AND bmi is high THEN positive
```

## Installation

```bash
uv pip install fysvm
```

Requires Python ≥ 3.14 and depends only on `numpy`, `scipy`, and `scikit-learn`.

## Quickstart

```python
from fysvm import FuzzyRuleSVM
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)

clf = FuzzyRuleSVM(
    max_rule_length=2,
    max_rules=128,
    penalty="l1",
    feature_names=data.feature_names,
    random_state=0,
)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
```

## How it works

The estimator has three stages:

**1. Fuzzy concepts.** Each numeric feature is summarized by three linguistic
terms — `low`, `medium`, `high` — defined by triangular membership functions
anchored at data quantiles. Every sample gets a membership score in [0, 1] for
each concept.

**2. Rule generation.** All conjunctions of up to `max_rule_length` concepts are
enumerated as candidate rules (e.g. `glucose is high AND bmi is high`).
Candidates are scored by discriminative power and coverage; the top `max_rules`
are kept.

**3. Max-margin learning.** Each sample is mapped to a firing-strength vector
`φ(x) ∈ [0, 1]^K` where entry `k` is the fuzzy AND of the concepts in rule `k`.
A sparse linear SVM is trained on this activation space:

```
f(x) = Σ_k β_k · φ_k(x) + b
```

## Explaining predictions

Because the decision function is a weighted sum of fuzzy rule firings,
explanations *are* the model computation — no proxy, no approximation.

### Per-sample explanation

```python
explanation = clf.explain(X_test[:1])[0]
```

Returns the bias, the net margin, and the top-contributing rules sorted by
absolute contribution. Each rule has:
- `rule` — a human-readable string like `IF feature is low THEN class_A`
- `firing` — how strongly the sample matches the rule antecedent in [0, 1]
- `weight` — the learned SVM coefficient `β_k`
- `contribution` — `firing × weight`, the rule's dollar-value impact on the margin

Contributions sum with the bias to the predicted margin:

```python
assert abs(explanation["margin"]
           - (explanation["bias"] + explanation["net_rule_contribution"])) < 1e-12
```

### Global rule inspection

```python
for item in clf.support_rules():
    print(item["rule"], item["weight"])
```

Returns rules with non-zero coefficients, sorted by absolute weight.

### Fuzzy concept membership

```python
concepts = clf.concept_memberships(X_test[:1])[0]
# {"glucose": {"low": 0.0, "medium": 0.3, "high": 0.7}, ...}
```

### Fuzzy margin violations

```python
violations = clf.fuzzy_violations(X_test, y_test)
# Each item: {"slack": 0.42, "memberships": {"cleanly_classified": 0.58,
#             "borderline": 0.42, "strong_violation": 0.0}}
```

## API Reference

### `FuzzyRuleSVM` (aliased as `SparseMaxMarginFuzzyRuleMachine`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | `1.0` | Inverse regularization strength |
| `penalty` | `"l1"` | `"l1"` or `"l2"` SVM penalty |
| `max_rule_length` | `2` | Max conjuncts per rule (≤ n_features) |
| `max_rules` | `256` | Max candidate rules kept |
| `min_rule_coverage` | `0.02` | Minimum fuzzy support for a candidate |
| `and_operator` | `"min"` | `"min"`, `"product"`, or `"softmin"` |
| `feature_names` | `None` | Column names for readable rule strings |
| `class_weight` | `None` | `"balanced"` or dict for imbalanced classes |

See the docstrings for the full parameter list.

## Example notebooks

See the [`examples/`](examples/) directory:

- [`basic_usage.py`](examples/basic_usage.py) — fit, predict, evaluate
- [`explain_predictions.py`](examples/explain_predictions.py) — detailed explanation walkthrough

## Citation

If you use fysvm in published work, please cite the accompanying paper:

```bibtex
@misc{sorensen2025fuzzy,
  author = {Sørensen, S. D.},
  title  = {Fuzzy Rule-Space Max-Margin Classifiers},
  year   = {2025}
}
```
