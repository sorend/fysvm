"""Basic usage: fit, predict, and evaluate FuzzyRuleSVM."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from fysvm import FuzzyRuleSVM

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)

clf = FuzzyRuleSVM(
    C=1.0,
    penalty="l1",
    max_rule_length=2,
    max_rules=128,
    min_rule_coverage=0.02,
    feature_names=data.feature_names,
    class_weight="balanced",
    random_state=0,
)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.3f}")
print(f"Rules used: {clf.n_rules_} (non-zero: {len(clf.active_rule_indices_)})")

print("\nTop-5 support rules:")
for item in clf.support_rules()[:5]:
    print(f"  {item['rule']:60s} weight={item['weight']:.4f}")
