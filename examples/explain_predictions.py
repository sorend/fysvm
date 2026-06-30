"""Demonstrate native prediction explanations."""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from fysvm import FuzzyRuleSVM

data = load_iris()
# Binarize: setosa vs rest for binary classification
y_binary = (data.target == 0).astype(int)
feature_names = [name.replace(" (cm)", "") for name in data.feature_names]

X_train, X_test, y_train, y_test = train_test_split(
    data.data, y_binary, test_size=0.3, random_state=0
)

clf = FuzzyRuleSVM(
    C=5.0,
    penalty="l1",
    max_rule_length=2,
    max_rules=32,
    feature_names=feature_names,
    random_state=0,
)
clf.fit(X_train, y_train)

sample = X_test[:1]
prediction = clf.predict(sample)[0]
print(f"Prediction: {'setosa' if prediction == 1 else 'not setosa'}\n")

explanation = clf.explain(sample, top_n=5)[0]
print(f"Bias:                 {explanation['bias']:.4f}")
print(f"Net rule contribution: {explanation['net_rule_contribution']:.4f}")
print(f"Margin:               {explanation['margin']:.4f}")
print()

print("Top contributing rules:")
for item in explanation["top_rules"]:
    print(f"  {item['rule']:55s}  firing={item['firing']:.3f}  "
          f"weight={item['weight']:+.4f}  contribution={item['contribution']:+.4f}")
