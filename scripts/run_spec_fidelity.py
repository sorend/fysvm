"""Specification Fidelity — Conformance Harness Script.

Runs conformance and metamorphic relation checks for FuzzyRuleSVM across
three synthetic datasets and three t-norm operators. Saves all artefacts
to runs/spec_fidelity/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from dataclasses import asdict

import numpy as np

# Ensure project root is on path so imports work when run with uv run python
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# Ensure tests/ is on path so reference_implementations can be imported
_tests_dir = _project_root / "tests"
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from fysvm import FuzzyRuleSVM
from fysvm.conformance import check_conformance, check_metamorphic_relations
from fysvm.run_metadata import write_run_metadata


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def make_synthetic_dataset(n: int, d: int, seed: int):
    """Binary classification synthetic dataset: two Gaussian blobs."""
    rng = np.random.default_rng(seed)
    n_pos = n // 2
    n_neg = n - n_pos
    X_pos = rng.normal(loc=1.0, scale=0.5, size=(n_pos, d))
    X_neg = rng.normal(loc=-1.0, scale=0.5, size=(n_neg, d))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [0] * n_neg)
    return X, y


DATASETS = [
    ("toy_separable_2d",  100,  2,  7),
    ("toy_medium_5d",     200,  5, 42),
    ("toy_noisy_10d",     300, 10, 99),
]

AND_OPERATORS = ["min", "product", "softmin"]


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def main():
    output_dir = Path("runs/spec_fidelity")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Compute and save reference hash BEFORE any comparison
    # ------------------------------------------------------------------
    ref_file = Path("tests/reference_implementations.py")
    sha256 = hashlib.sha256(ref_file.read_bytes()).hexdigest()
    hash_line = f"{sha256}  tests/reference_implementations.py\n"
    (output_dir / "reference_hash.txt").write_text(hash_line)
    print(f"Reference SHA-256: {sha256}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    all_conformance = []
    all_metamorphic = []

    print()
    print("=" * 72)
    print(f"{'Dataset':<26} {'Operator':<10} {'Status':<22} {'MaxErr':>12}")
    print("=" * 72)

    for dataset_name, n, d, seed in DATASETS:
        X, y = make_synthetic_dataset(n, d, seed)

        for operator in AND_OPERATORS:
            clf = FuzzyRuleSVM(
                C=1.0,
                penalty="l1",
                and_operator=operator,  # type: ignore[arg-type]
                max_rule_length=2,
                max_rules=64,
                rule_generation="enumeration",
                softmin_temperature=0.1,
                random_state=42,
            )
            clf.fit(X, y)

            # Conformance check
            conf = check_conformance(clf, X, f"{dataset_name}_{operator}")
            conf_dict = asdict(conf)
            conf_dict["dataset_name_raw"] = dataset_name
            conf_dict["n"] = n
            conf_dict["d"] = d
            conf_dict["seed"] = seed
            all_conformance.append(conf_dict)

            status_str = conf.status
            err_str = f"{conf.max_abs_error:.2e}"
            print(f"{dataset_name:<26} {operator:<10} {status_str:<22} {err_str:>12}")

            # Metamorphic relations (only on the "min" operator to avoid redundancy,
            # but also run on product and softmin)
            mr_results = check_metamorphic_relations(clf, X, y)
            for mr in mr_results:
                mr_dict = asdict(mr)
                mr_dict["dataset_name"] = dataset_name
                mr_dict["and_operator"] = operator
                all_metamorphic.append(mr_dict)

    print("=" * 72)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    (output_dir / "conformance_results.json").write_text(
        json.dumps(all_conformance, indent=2, allow_nan=False)
    )
    (output_dir / "metamorphic_results.json").write_text(
        json.dumps(all_metamorphic, indent=2, allow_nan=False)
    )

    # ------------------------------------------------------------------
    # Metamorphic summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Metamorphic relation summary")
    print("=" * 72)

    mr_by_name: dict[str, list[dict]] = {}
    for mr in all_metamorphic:
        name = mr["relation_name"]
        mr_by_name.setdefault(name, []).append(mr)

    print(f"{'Relation':<35} {'Total':>6} {'Passed':>7} {'Failed':>7} {'MaxViol':>12}")
    print("-" * 72)
    for mr_name, records in mr_by_name.items():
        total = len(records)
        passed = sum(1 for r in records if r["passed"])
        failed = total - passed
        max_viol = max(r["max_violation"] for r in records)
        viol_str = f"{max_viol:.2e}"
        print(f"{mr_name:<35} {total:>6} {passed:>7} {failed:>7} {viol_str:>12}")

    print("=" * 72)

    # ------------------------------------------------------------------
    # Overall verdict
    # ------------------------------------------------------------------
    certified_count = sum(1 for c in all_conformance if c["status"] == "CERTIFIED")
    counterex_count = sum(1 for c in all_conformance if c["status"] == "COUNTEREXAMPLE")
    unknown_count = sum(1 for c in all_conformance if c["status"] == "UNKNOWN")
    eligible_count = sum(
        1 for c in all_conformance if c["certificate_eligibility_status"] == "ELIGIBLE"
    )
    ineligible_count = sum(
        1 for c in all_conformance if c["certificate_eligibility_status"] == "INELIGIBLE"
    )
    all_mr_passed = all(r["passed"] for r in all_metamorphic)

    print()
    print(f"Conformance: {certified_count} CERTIFIED, "
          f"{counterex_count} COUNTEREXAMPLE, {unknown_count} UNKNOWN")
    print(f"Property-certificate eligibility: {eligible_count} ELIGIBLE, "
          f"{ineligible_count} INELIGIBLE")
    print(f"Metamorphic: {'ALL PASSED' if all_mr_passed else 'FAILURES DETECTED'}")
    print()
    write_run_metadata(
        output_dir,
        command=["uv", "run", "python", *sys.argv],
        config={
            "datasets": DATASETS,
            "and_operators": AND_OPERATORS,
            "rule_generation": "enumeration",
            "softmin_temperature": 0.1,
            "reference_sha256": sha256,
            "provenance_scope": (
                "Repository-local integrity record; not an externally timestamped preregistration."
            ),
        },
    )
    print(f"Artefacts saved to {output_dir.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
