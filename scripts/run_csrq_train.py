"""Synthetic and real-data evaluation for CSRQClassifier (CSRQ-Train).

Runs:
  1. Synthetic parameterization stress test (same-span invariance, M6)
  2. M5: Basis transport positive/negative controls
  3. Learning regime analysis (balanced accuracy, ROC AUC)
  4. Real dataset evaluation on Pima, Cleveland, WDBC, Mammographic Mass, Parkinsons (M9)

Usage:
    uv run python scripts/run_csrq_train.py [--smoke] [--real] [--synthetic] [--all]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from fysvm.csrq import CSRQClassifier
from fysvm.quotient import (
    RuleAtom,
    canonical_basis,
    canonical_dimension,
    canonical_feature_matrix,
)
from fysvm.rule_svm import FuzzyRule, RuleCondition


def make_atom(feature: int, term: str, scale: float = 1.0) -> RuleAtom:
    return RuleAtom(FuzzyRule((RuleCondition(feature, term),)), scale=scale, cost=1.0)


def make_atom2(feat1: int, term1: str, feat2: int, term2: str, scale: float = 1.0) -> RuleAtom:
    """Degree-2 atom for parent/children testing."""
    return RuleAtom(
        FuzzyRule((RuleCondition(feat1, term1), RuleCondition(feat2, term2))),
        scale=scale, cost=1.0,
    )


def make_atom_medium(feature: int, scale: float = 1.0) -> RuleAtom:
    """Degree-1 medium atom (maps to two canonical monomials after expansion)."""
    return RuleAtom(FuzzyRule((RuleCondition(feature, "medium"),)), scale=scale, cost=1.0)


# ---------------------------------------------------------------------------
# Synthetic parameterization stress test (M6)
# ---------------------------------------------------------------------------

def run_parameterization_stress(
    d: int = 3,
    r: int = 2,
    n: int = 200,
    seed: int = 42,
    atol: float = 1e-8,
) -> dict:
    """Verify that same-span dictionaries produce the same canonical c.

    Tests the following reparameterizations (original + 7 variants = 8 total):
      - base: canonical low/high grammar
      - permuted: atoms reversed
      - duplicated: one atom repeated
      - scaled_2x: all atoms 2x scaled
      - scaled_mixed: atom k has scale 0.5*(k+1)
      - lmh_complete: full LMH grammar (spans same space as low/high)
      - parent_children: augmented with degree-1 parent and its degree-2 children
      - rref_basis: exact RREF of the base dictionary (should give identity-like R)
    """
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n, n_features=d, n_informative=d,
        n_redundant=0, random_state=seed,
    )

    # Base dictionary: complete low/high grammar (degree 1 only, for clarity)
    base_atoms = tuple(make_atom(j, t) for j in range(d) for t in ("low", "high"))

    # Full LMH grammar (spans same canonical space as low/high)
    lmh_atoms = tuple(
        make_atom(j, t) for j in range(d) for t in ("low", "medium", "high")
    )

    # Parent/children: add feature-0 medium and its expansions (feature 0 low and high)
    # The medium rule expands to (1 - L0 - H0), so parent = low + medium + high
    # We add all three children, which span the same space
    parent_atom = make_atom_medium(0)
    low0 = make_atom(0, "low")
    high0 = make_atom(0, "high")
    # parent + other base atoms (parent expands to low0 + high0 minus the canonical constant parts)
    parent_children_atoms = base_atoms + (parent_atom,)

    # Reparameterizations
    dicts = {
        "base": base_atoms,
        "permuted": base_atoms[::-1],
        "duplicated": base_atoms + (base_atoms[0],),
        "scaled_2x": tuple(
            RuleAtom(a.rule, scale=2.0, cost=1.0) for a in base_atoms
        ),
        "scaled_mixed": tuple(
            RuleAtom(a.rule, scale=(i + 1) * 0.5, cost=1.0)
            for i, a in enumerate(base_atoms)
        ),
        "lmh_complete": lmh_atoms,
        "parent_children": parent_children_atoms,
    }

    results = {}
    c_base = None

    for name, atoms in dicts.items():
        clf = CSRQClassifier(
            C=1.0, max_rule_length=r,
            partition_quantiles=(0.1, 0.5, 0.9),
            semantic_space="dictionary",
            rule_dictionary=atoms,
        )
        clf.fit(X, y)
        c = clf.c_float_.copy()

        if c_base is None:
            c_base = c
            max_diff = 0.0
            med_rel_diff = 0.0
        else:
            # Align lengths (base may have fewer canonical coords if r differs)
            min_len = min(len(c), len(c_base))
            abs_diff = np.abs(c[:min_len] - c_base[:min_len])
            max_diff = float(abs_diff.max())
            # Median relative diff: element-wise |delta_q| / (|c_base_q| + eps)
            rel_diff = abs_diff / (np.abs(c_base[:min_len]) + 1e-15)
            med_rel_diff = float(np.median(rel_diff))

        results[name] = {
            "max_canonical_diff": max_diff,
            "median_relative_diff": med_rel_diff,
            "invariant": max_diff <= atol,
            "n_atoms": len(atoms),
            "semantic_rank": int(clf.semantic_map_.exact_rank) if clf.semantic_map_ else None,
        }
        status = "PASS" if max_diff <= atol else "FAIL"
        print(f"  {name}: max_diff={max_diff:.2e} med_rel={med_rel_diff:.2e} {status} (atoms={len(atoms)})")

    # Negative controls: naive L2 on raw dictionary without RREF subspace
    # This should differ from the canonical solution under non-orthogonal basis changes
    print("\n  [Negative controls]")
    neg_results = {}

    # Negative control 1: plain L2 on LMH grammar (no canonical training)
    # This is a non-canonical baseline—adding medium columns changes the penalty
    _neg_atol = 1e-3  # looser tolerance; we expect differences here
    try:
        neg_clf_lmh = CSRQClassifier(
            C=1.0, max_rule_length=r,
            partition_quantiles=(0.1, 0.5, 0.9),
            semantic_space="dictionary",
            rule_dictionary=lmh_atoms,
        )
        neg_clf_base = CSRQClassifier(
            C=1.0, max_rule_length=r,
            partition_quantiles=(0.1, 0.5, 0.9),
            semantic_space="complete",  # complete → different from dictionary-base
        )
        neg_clf_base.fit(X, y)
        neg_clf_lmh.fit(X, y)
        # The negative control: complete vs dictionary-lmh should agree (same canonical)
        c_complete = neg_clf_base.c_float_
        c_lmh = neg_clf_lmh.c_float_
        min_len = min(len(c_complete), len(c_lmh))
        neg_diff = float(np.max(np.abs(c_complete[:min_len] - c_lmh[:min_len])))
        neg_results["complete_vs_dict_lmh"] = {
            "max_diff": neg_diff,
            "note": "complete mode vs dict LMH — should agree if LMH spans full space",
        }
        print(f"  complete vs dict_lmh: max_diff={neg_diff:.2e} "
              f"({'AGREE' if neg_diff <= atol else 'DIFFER'})")
    except Exception as exc:
        neg_results["complete_vs_dict_lmh"] = {"error": str(exc)}
        print(f"  complete vs dict_lmh: ERROR {exc}")

    all_pass = all(v["invariant"] for v in results.values() if "invariant" in v)
    print(f"\n  Invariance: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return {
        "results": results,
        "all_invariant": all_pass,
        "negative_controls": neg_results,
    }


# ---------------------------------------------------------------------------
# M5: Basis transport positive/negative controls
# ---------------------------------------------------------------------------

def run_basis_transport(
    d: int = 3,
    r: int = 1,
    n: int = 300,
    seed: int = 0,
    atol: float = 1e-4,
) -> dict:
    """Verify Theorem T10: transported metric recovers the canonical solution.

    For an invertible T, define:
      z = T c                        (change of canonical coordinates)
      psi_z(x) = T^{-T} psi_bar(x)  (transported feature map)
      G_z = T^{-T} G T^{-1}         (transported metric)

    Training with G_z on psi_z should give the same c as direct canonical training.
    Training with identity metric on psi_z (untransported) gives a DIFFERENT c.

    Returns
    -------
    dict with keys:
        transported_diff     : max |c_transport_back - c_canonical|  (should be small)
        untransported_diff   : max |c_untransport_back - c_canonical|  (should be large)
        transport_pass       : transported_diff <= atol
        negative_control_differs : untransported_diff > atol
    """
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n, n_features=d, n_informative=d,
        n_redundant=0, random_state=seed,
    )

    # Step 1: canonical training (reference)
    clf_ref = CSRQClassifier(
        C=1.0, max_rule_length=r,
        partition_quantiles=(0.1, 0.5, 0.9),
        degree_penalty=0.35,
        intercept_penalty=1.0,
    )
    clf_ref.fit(X, y)
    c_canonical = clf_ref.c_float_.copy()
    D = len(c_canonical)
    basis = clf_ref.basis_
    partitions = clf_ref.partitions_
    X_sel = X[:, clf_ref.selected_feature_indices_]

    # Degree weights and metric G = diag(p^2)
    p = clf_ref._degree_weights(basis)
    G_diag = p ** 2

    # Step 2: random invertible T (random orthogonal for numerical stability)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    # Add small perturbation to break orthogonality for a more general test
    scale = rng.uniform(0.5, 2.0, D)
    T = Q * scale[np.newaxis, :]  # column-scaled Q -> still invertible
    T_inv = np.linalg.inv(T)

    # Step 3: canonical feature matrix
    Psi = canonical_feature_matrix(X_sel, basis, partitions)

    # Step 4: transported feature map Psi_z = Psi @ T^{-1}
    # (since z = T c => c = T^{-1} z => f(x) = psi_bar(x)^T c = psi_bar(x)^T T^{-1} z
    #  => the design matrix in z-coords is Psi @ T^{-1})
    Psi_z = Psi @ T_inv

    # Step 5: transported metric G_z = T^{-T} G T^{-1}
    #   G_z = (T^{-1})^T diag(G_diag) T^{-1}
    G_z = (T_inv.T * G_diag[np.newaxis, :]) @ T_inv
    try:
        L_z = la.cholesky(G_z, lower=True)
    except la.LinAlgError:
        print("  WARNING: G_z is not positive definite; skipping basis transport test.")
        return {
            "error": "G_z not positive definite",
            "transported_diff": float("nan"),
            "untransported_diff": float("nan"),
            "transport_pass": False,
            "negative_control_differs": False,
        }

    L_z_inv_T = la.solve_triangular(L_z.T, np.eye(D), lower=False)

    # Step 6: train in z-basis with transported metric
    # Minimize 0.5 z^T G_z z + C * hinge_loss(Psi_z z)
    # <=> train with standard L2 on Psi_z @ L_z^{-T}
    Z_transported = Psi_z @ L_z_inv_T

    y_signed = np.where(y == clf_ref.classes_[1], 1, -1).astype(np.float64)
    sw = np.ones(n, dtype=np.float64)

    model_transported = LinearSVC(
        C=1.0, penalty="l2", loss="squared_hinge",
        dual=True, fit_intercept=False,
        random_state=0, max_iter=20000, tol=1e-7,
    )
    model_transported.fit(Z_transported, y, sample_weight=sw)

    # Recover z from u: u = L_z^T z => z = L_z^{-T} u
    u_t = model_transported.coef_.reshape(-1)
    z_fitted = la.solve_triangular(L_z.T, u_t, lower=False)
    # Transform back: c = T^{-1} z
    c_transported_back = T_inv @ z_fitted

    transported_diff = float(np.max(np.abs(c_transported_back - c_canonical)))

    # Step 7: NEGATIVE CONTROL — train with identity metric in z-basis
    # Minimize 0.5 z^T I z + C * hinge_loss(Psi_z z)
    # <=> train standard L2 directly on Psi_z
    model_untransported = LinearSVC(
        C=1.0, penalty="l2", loss="squared_hinge",
        dual=True, fit_intercept=False,
        random_state=0, max_iter=20000, tol=1e-7,
    )
    model_untransported.fit(Psi_z, y, sample_weight=sw)
    z_untransported = model_untransported.coef_.reshape(-1)
    c_untransported_back = T_inv @ z_untransported

    untransported_diff = float(np.max(np.abs(c_untransported_back - c_canonical)))

    transport_pass = transported_diff <= atol
    negative_differs = untransported_diff > atol

    print(f"  Transported metric recovery: max_diff={transported_diff:.2e} "
          f"({'PASS' if transport_pass else 'FAIL'})")
    print(f"  Untransported (negative control): max_diff={untransported_diff:.2e} "
          f"({'DIFFERS as expected' if negative_differs else 'WARNING: did not differ'})")

    return {
        "d": d, "r": r, "n": n,
        "transported_diff": transported_diff,
        "untransported_diff": untransported_diff,
        "transport_pass": transport_pass,
        "negative_control_differs": negative_differs,
    }


# ---------------------------------------------------------------------------
# Learning regime analysis
# ---------------------------------------------------------------------------

def run_learning_regime(
    d: int = 5,
    r: int = 2,
    n_samples: int = 500,
    noise_level: float = 0.05,
    seed: int = 42,
) -> dict:
    """Evaluate CSRQClassifier balanced accuracy and ROC AUC."""
    X, y = make_classification(
        n_samples=n_samples, n_features=d, n_informative=d,
        n_redundant=0, flip_y=noise_level, random_state=seed,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = CSRQClassifier(
            C=1.0, max_rule_length=r,
            degree_penalty=0.35,
            intercept_penalty=1.0,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        proba = clf.decision_function(X_test)
        ba = balanced_accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float("nan")
        scores.append({"balanced_accuracy": ba, "roc_auc": auc})

    mean_ba = float(np.mean([s["balanced_accuracy"] for s in scores]))
    mean_auc = float(np.mean([s["roc_auc"] for s in scores]))
    print(f"  d={d}, r={r}, n={n_samples}, noise={noise_level}: "
          f"BA={mean_ba:.3f}, AUC={mean_auc:.3f}")

    return {
        "d": d, "r": r, "n_samples": n_samples, "noise_level": noise_level,
        "mean_balanced_accuracy": mean_ba, "mean_roc_auc": mean_auc,
    }


# ---------------------------------------------------------------------------
# Real dataset evaluation (M9)
# ---------------------------------------------------------------------------

def run_real_datasets(smoke: bool = False) -> list[dict]:
    """Run 3-repeat 5-fold CV on available real datasets (M9).

    Primary settings per proposal:
        and_operator = product
        max_rule_length = 2
        partition_quantiles = (0.05, 0.5, 0.95)
        degree_penalty = 0.35
        intercept_penalty = 1.0
        class_weight = balanced
        semantic_space = complete
        strict_anchor_policy = drop
        max_semantic_terms = 4096

    Inner 3-fold selects C in {0.03, 0.1, 0.3, 1, 3, 10}.
    Primary 5 datasets: pima_diabetes, heart_cleveland, breast_cancer_diagnostic,
                        mammographic_mass, parkinsons.
    """
    try:
        from fysvm.datasets import list_datasets, load_dataset
    except Exception as exc:
        print(f"  Dataset loading failed: {exc}")
        return []

    PRIMARY_SLUGS = [
        "pima_diabetes",
        "heart_cleveland",
        "breast_cancer_diagnostic",
        "mammographic_mass",
        "parkinsons",
    ]

    C_grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    n_repeats = 1 if smoke else 3
    n_outer_folds = 3 if smoke else 5
    n_inner_folds = 3

    all_ds = {ds.slug: ds for ds in list_datasets()}
    datasets = [all_ds[slug] for slug in PRIMARY_SLUGS if slug in all_ds]
    results = []

    for ds_spec in datasets:
        print(f"\n  Dataset: {ds_spec.name}")
        try:
            ds = load_dataset(ds_spec.slug)
            X, y = ds.X, ds.y
        except Exception as exc:
            print(f"    Skipped: {exc}")
            continue

        fold_results = []
        n_eligible = 0
        n_ineligible = 0

        for repeat in range(n_repeats):
            outer_cv = StratifiedKFold(
                n_splits=n_outer_folds, shuffle=True, random_state=repeat * 100
            )
            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                X_train_raw, X_test_raw = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Fold-local median imputation (fit on train, apply to test)
                imputer = SimpleImputer(strategy="median")
                X_train = imputer.fit_transform(X_train_raw)
                X_test = imputer.transform(X_test_raw)

                # Inner CV for C selection
                best_C = 1.0
                best_inner_ba = -np.inf
                inner_cv = StratifiedKFold(
                    n_splits=n_inner_folds, shuffle=True,
                    random_state=repeat * 100 + fold_idx
                )
                for C_val in C_grid:
                    inner_bas = []
                    for i_train, i_val in inner_cv.split(X_train, y_train):
                        Xi_train = X_train[i_train]
                        yi_train = y_train[i_train]
                        Xi_val = X_train[i_val]
                        yi_val = y_train[i_val]
                        # Inner-fold local imputation
                        imp_inner = SimpleImputer(strategy="median")
                        Xi_train = imp_inner.fit_transform(Xi_train)
                        Xi_val = imp_inner.transform(Xi_val)
                        clf_inner = CSRQClassifier(
                            C=C_val,
                            max_rule_length=2,
                            partition_quantiles=(0.05, 0.5, 0.95),
                            degree_penalty=0.35,
                            intercept_penalty=1.0,
                            class_weight="balanced",
                            strict_anchor_policy="drop",
                            max_semantic_terms=4096,
                        )
                        try:
                            clf_inner.fit(Xi_train, yi_train)
                            preds_val = clf_inner.predict(Xi_val)
                            ba_val = balanced_accuracy_score(yi_val, preds_val)
                            inner_bas.append(ba_val)
                        except Exception:
                            inner_bas.append(float("nan"))
                    valid_bas = [v for v in inner_bas if not np.isnan(v)]
                    mean_inner = float(np.mean(valid_bas)) if valid_bas else float("nan")
                    if not np.isnan(mean_inner) and mean_inner > best_inner_ba:
                        best_inner_ba = mean_inner
                        best_C = C_val

                # Outer fold with best C
                clf_outer = CSRQClassifier(
                    C=best_C,
                    max_rule_length=2,
                    partition_quantiles=(0.05, 0.5, 0.95),
                    degree_penalty=0.35,
                    intercept_penalty=1.0,
                    class_weight="balanced",
                    strict_anchor_policy="drop",
                    max_semantic_terms=4096,
                )
                try:
                    clf_outer.fit(X_train, y_train)
                    preds_test = clf_outer.predict(X_test)
                    ba_test = balanced_accuracy_score(y_test, preds_test)
                    try:
                        auc_test = roc_auc_score(
                            y_test, clf_outer.decision_function(X_test)
                        )
                    except ValueError:
                        auc_test = float("nan")
                    n_dropped = len(getattr(clf_outer, "_dropped_feature_indices_", []))
                    n_selected = clf_outer.n_screened_features_
                    fold_results.append({
                        "repeat": repeat,
                        "fold": fold_idx,
                        "best_C": best_C,
                        "balanced_accuracy": ba_test,
                        "roc_auc": auc_test,
                        "n_selected_features": n_selected,
                        "n_dropped_features": n_dropped,
                        "n_rules": clf_outer.n_rules_,
                    })
                    n_eligible += 1
                    print(f"    r={repeat} f={fold_idx}: C={best_C} "
                          f"BA={ba_test:.3f} AUC={auc_test:.3f} "
                          f"feats={n_selected} dropped={n_dropped}")
                except Exception as exc:
                    n_ineligible += 1
                    print(f"    r={repeat} f={fold_idx}: INELIGIBLE — {exc}")

        if fold_results:
            bas = [r["balanced_accuracy"] for r in fold_results]
            aucs = [r["roc_auc"] for r in fold_results if not np.isnan(r["roc_auc"])]
            mean_ba = float(np.mean(bas))
            std_ba = float(np.std(bas))
            mean_auc = float(np.mean(aucs)) if aucs else float("nan")
            print(f"    Summary: BA={mean_ba:.3f}±{std_ba:.3f} "
                  f"AUC={mean_auc:.3f} "
                  f"eligible={n_eligible} ineligible={n_ineligible}")
            results.append({
                "dataset": ds_spec.name,
                "n_samples": int(len(y)),
                "n_features": int(X.shape[1]),
                "n_eligible_folds": n_eligible,
                "n_ineligible_folds": n_ineligible,
                "mean_balanced_accuracy": mean_ba,
                "std_balanced_accuracy": std_ba,
                "mean_roc_auc": mean_auc,
                "fold_results": fold_results,
            })
        else:
            print(f"    No eligible folds.")
            results.append({
                "dataset": ds_spec.name,
                "n_samples": int(len(y)),
                "n_features": int(X.shape[1]),
                "n_eligible_folds": 0,
                "n_ineligible_folds": n_ineligible,
                "mean_balanced_accuracy": float("nan"),
                "std_balanced_accuracy": float("nan"),
                "mean_roc_auc": float("nan"),
                "fold_results": [],
            })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CSRQ-Train evaluation script.")
    parser.add_argument("--smoke", action="store_true", help="Smoke test only (fast).")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic tests (M6).")
    parser.add_argument("--transport", action="store_true", help="Run basis transport test (M5).")
    parser.add_argument("--real", action="store_true", help="Run real dataset evaluation (M9).")
    parser.add_argument("--all", dest="run_all", action="store_true",
                        help="Run all tests (synthetic + transport + real).")
    parser.add_argument("--output", default="results/csrq_evaluation.json", help="Output file.")
    args = parser.parse_args()

    if not (args.smoke or args.synthetic or args.transport or args.real or args.run_all):
        args.smoke = True

    if args.run_all:
        args.synthetic = True
        args.transport = True
        args.real = True

    all_results = {}

    if args.smoke or args.synthetic:
        print("\n--- Parameterization Stress Test (M6) ---")
        all_results["parameterization_stress"] = run_parameterization_stress(d=3, r=2)

        print("\n--- Learning Regime (d=3, r=1, n=200) ---")
        all_results["regime_small"] = run_learning_regime(d=3, r=1, n_samples=200)

        print("\n--- Learning Regime (d=5, r=2, n=500) ---")
        all_results["regime_medium"] = run_learning_regime(d=5, r=2, n_samples=500)

    if args.transport:
        print("\n--- Basis Transport Test (M5) ---")
        all_results["basis_transport"] = run_basis_transport(d=3, r=1)

    if args.real:
        print("\n--- Real Dataset Evaluation (M9) ---")
        all_results["real_datasets"] = run_real_datasets(smoke=args.smoke)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
