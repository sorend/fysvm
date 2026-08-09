# fysvm

[![GitHub Release](https://img.shields.io/github/v/release/sorend/fysvm)](https://github.com/sorend/fysvm/releases)
[![Build Status](https://img.shields.io/github/actions/workflow/status/sorend/fysvm/build.yml?branch=main)](https://github.com/sorend/fysvm/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/sorend/fysvm/branch/main/graph/badge.svg)](https://codecov.io/gh/sorend/fysvm)

Intrinsically interpretable fuzzy SVM-style classifiers built on scikit-learn.

`fysvm` trains linear max-margin classifiers over fuzzy rule activations instead
of raw feature coordinates. Every learned dimension is a human-readable
linguistic rule such as:

```
IF glucose is high AND bmi is high THEN positive
```

Two classifiers are provided:

- **`FuzzyRuleSVM`** — trains in raw rule activation space; fast and interpretable.
- **`CSRQClassifier`** — trains in the canonical quotient space; solves the
  parameterisation-invariance problem described below.

## Installation

```bash
uv pip install fysvm
```

`CSRQClassifier` requires `sympy` for exact rational arithmetic, which is
included in the default dependencies. The optional Variant B atomic norm
classifier (`QuotientAtomicFuzzySVM`) additionally requires `osqp`:

```bash
uv pip install "fysvm[csrq-atomic]"
```

Requires Python ≥ 3.14.

## Quickstart: `FuzzyRuleSVM`

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

## How `FuzzyRuleSVM` works

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

## Explaining `FuzzyRuleSVM` predictions

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
- `contribution` — `firing × weight`, the rule's impact on the margin

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

---

## The parameterisation problem

`FuzzyRuleSVM` trains in raw rule activation space with a coefficient-space
norm. This norm is not invariant to the algebraic structure of the rule basis.

When the product t-norm is used with strict triangular anchors, the partition
satisfies two pointwise identities:

```
L_j(x) + M_j(x) + H_j(x) = 1    (Ruspini identity)
L_j(x) · H_j(x) = 0              (orthogonality)
```

These identities generate a polynomial ideal that makes many rule dictionaries
*semantically equivalent*: they span the same hypothesis space. Replacing a
`medium` atom with its equivalent low/high expansion, permuting the rule
ordering, or rescaling atom weights are all parameterisations of the same
function space. Yet the L2 penalty changes with each such reparameterisation,
and the trained model typically changes with it.

**In practice:** two analysts designing semantically equivalent rule grammars
independently — one using explicit `medium` atoms, the other using `low`/`high`
expansions — obtain different trained models under `FuzzyRuleSVM`, even though
their hypothesis spaces are identical. This is a reproducibility and
interpretability artefact.

`CSRQClassifier` eliminates this artefact.

---

## Quickstart: `CSRQClassifier`

```python
from fysvm import CSRQClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)

clf = CSRQClassifier(
    C=1.0,
    max_rule_length=2,
    feature_names=data.feature_names,
)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
```

## How `CSRQClassifier` works

`CSRQClassifier` constructs the **canonical quotient space** `Q_{d,r}`
explicitly before training. Medium terms are eliminated by exact algebraic
substitution — `M_j → 1 - L_j - H_j` — so the canonical basis contains only
`low` and `high` monomials. Its dimension is:

```
D_{d,r} = Σ_{l=0}^{r} C(d,l) · 2^l
```

strictly smaller than the full grammar size `N_{d,r} = Σ C(d,l) · 3^l`.

A positive-definite degree-weighted metric `G = diag(p_q²)` on this basis
yields a **unique ideal minimiser** that is independent of the rule dictionary
used to construct the training data. Two training modes are available:

- **`complete`** (default) — trains over all `D` canonical coordinates.
  Fully dictionary-invariant: any rule dictionary produces the same solution.
- **`dictionary`** — trains over the exact RREF subspace of a supplied rule
  dictionary. Invariant across all dictionaries with the same semantic span.

Every fitted model is accompanied by an exact certificate `A_D · γ = c`
verified in rational arithmetic, enabling independent audit of semantic
consistency.

## Explaining `CSRQClassifier` predictions

```python
explanation = clf.explain(X_test[:1])[0]
print(explanation["prediction"])
print(explanation["margin"])

for rule in explanation["top_rules"]:
    print(f"{rule['monomial']:40s}  weight={rule['weight']:+.4f}  firing={rule['firing']:.3f}")
```

### Global support rules

```python
for item in clf.support_rules():
    print(item["rule"], item["weight"])
```

### Decoding back to the original grammar

The canonical basis eliminates `medium` atoms for training invariance, but
you can decode the canonical coefficients back to original rule weights at
any time:

```python
decoded = clf.decode(method="minimum_l2")
for atom, weight in zip(decoded["atoms"], decoded["weights"]):
    print(f"{atom}  →  {weight:+.4f}")
```

### Exporting a reproducibility artifact

```python
artifact = clf.export_artifact(X_val=X_test)
# artifact.semantic_equality_certificate.max_abs_residual is the
# exact rational certificate residual (should be 0)
print(artifact.optimization_report.n_iter)
print(artifact.semantic_equality_certificate.is_certified)
```

---

## Dictionary mode: preserving hand-crafted grammars

If you have domain-specific rules that you want to keep intact (including
`medium` atoms), supply them explicitly and use `dictionary` mode:

```python
from fysvm import CSRQClassifier
from fysvm.quotient import RuleAtom
from fysvm.rule_svm import FuzzyRule, RuleCondition

atoms = (
    RuleAtom(FuzzyRule((RuleCondition(0, "high"), RuleCondition(1, "high"))), scale=1.0, cost=1.0),
    RuleAtom(FuzzyRule((RuleCondition(0, "low"),)), scale=1.0, cost=1.0),
    RuleAtom(FuzzyRule((RuleCondition(1, "medium"),)), scale=1.0, cost=1.0),
)

clf = CSRQClassifier(
    semantic_space="dictionary",
    rule_dictionary=atoms,
)
clf.fit(X_train, y_train)
```

The training problem is invariant across any other dictionary that spans the
same semantic subspace (same RREF row space).

---

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

### `CSRQClassifier`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | `1.0` | SVM regularization strength |
| `max_rule_length` | `2` | Max degree of canonical monomials |
| `degree_penalty` | `1.0` | Weight increment per extra degree above 1 (η) |
| `intercept_penalty` | `1.0` | Regularization weight for the intercept (p₀ > 0) |
| `semantic_space` | `"complete"` | `"complete"` or `"dictionary"` |
| `rule_dictionary` | `None` | `tuple[RuleAtom, ...]` for dictionary mode |
| `partition_quantiles` | `(0.25, 0.5, 0.75)` | Quantiles for fuzzy partition anchors |
| `feature_names` | `None` | Column names for readable rule strings |
| `class_weight` | `None` | `"balanced"` or dict for imbalanced classes |

See the docstrings for the full parameter list.

## Example notebooks

See the [`examples/`](examples/) directory:

- [`basic_usage.py`](examples/basic_usage.py) — fit, predict, evaluate
- [`explain_predictions.py`](examples/explain_predictions.py) — detailed explanation walkthrough

## Citations

If you use `FuzzyRuleSVM` in published work, please cite:

```bibtex
@misc{davidsen2026fuzzy,
  author = {Davidsen, S. A. and Padmavathamma, M.},
  title  = {Faithful Regularised Classification over Linguistic Fuzzy Rule Activations},
  year   = {2026},
  note   = {Under review}
}
```

If you use `CSRQClassifier`, please also cite:

```bibtex
@misc{davidsen2026csrq,
  author = {Davidsen, S. A. and Padmavathamma, M.},
  title  = {Quotient-Invariant Max-Margin Training for Product Fuzzy Rule Classifiers},
  year   = {2026},
  note   = {Under review}
}
```
