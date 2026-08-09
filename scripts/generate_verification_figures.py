#!/usr/bin/env python3
"""Generate all verification figures and LaTeX tables for the FuzzyRuleSVM paper.

Outputs:
  paper/figures/fig1_conformance_overview.pdf
  paper/figures/fig2_metamorphic_matrix.pdf
  paper/figures/fig3_certificate_rates.pdf
  paper/figures/fig4_product_vs_min.pdf
  paper/figures/fig5_bootstrap_agreement.pdf
  paper/figures/fig6_rashomon_accuracy_curves.pdf
  paper/figures/fig7_certificate_retention.pdf
  paper/generated_tables.tex
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Any

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
for style_name in ('seaborn-v0_8-paper', 'seaborn-paper', 'seaborn'):
    try:
        plt.style.use(style_name)
        break
    except OSError:
        continue

DOUBLE_COL = 7.0   # inches

# ── Load data ─────────────────────────────────────────────────────────────────
def _load_json(path: str) -> Any:
    """Load a strict JSON artifact."""
    def reject_constant(value: str):
        raise ValueError(f"Non-standard JSON constant {value!r} in {path}")

    with open(path) as fh:
        return json.load(fh, parse_constant=reject_constant)


conformance = _load_json("runs/spec_fidelity/conformance_results.json")
metamorphic = _load_json("runs/spec_fidelity/metamorphic_results.json")
prop_cert_data = _load_json("runs/prop_cert/results.json")
stability = _load_json("runs/finite_c_grid_stability/stability_results.json")

primary_summary = prop_cert_data['summary']

print("Data loaded.")
print(f"  conformance entries  : {len(conformance)}")
print(f"  metamorphic entries  : {len(metamorphic)}")
print(f"  prop_cert primary summary keys : {list(primary_summary.keys())}")
print(f"  finite C-grid stability datasets : {list(stability.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Specification Fidelity Conformance Overview (heatmap grid)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_conformance_overview():
    datasets  = ['toy_separable_2d', 'toy_medium_5d', 'toy_noisy_10d']
    operators = ['min', 'product', 'softmin']
    op_labels = ['Min', 'Product', 'Softmin']

    conf_lut = {(r['dataset_name_raw'], r['and_operator']): r for r in conformance}
    ds_names = ['Sep. 2D', 'Med. 5D', 'Noisy 10D']
    ds_labels = [
        f'{name}\n(n={conf_lut[(dataset, "min")]["n"]}, '
        f'd={conf_lut[(dataset, "min")]["d"]})'
        for dataset, name in zip(datasets, ds_names)
    ]

    conformance_statuses = ['CERTIFIED', 'COUNTEREXAMPLE', 'UNKNOWN']
    eligibility_statuses = ['ELIGIBLE', 'INELIGIBLE', 'UNKNOWN']
    conformance_mat = np.zeros((3, 3))
    eligibility_mat = np.zeros((3, 3))
    for i, ds in enumerate(datasets):
        for j, op in enumerate(operators):
            result = conf_lut[(ds, op)]
            conformance_mat[i, j] = conformance_statuses.index(result['status'])
            eligibility_mat[i, j] = eligibility_statuses.index(
                result['certificate_eligibility_status']
            )

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.7))
    panels = [
        (
            axes[0], conformance_mat, conformance_statuses,
            ListedColormap(['#2ca02c', '#d62728', '#7f7f7f']),
            'Measured Reference Conformance',
        ),
        (
            axes[1], eligibility_mat, eligibility_statuses,
            ListedColormap(['#1f77b4', '#ff7f0e', '#7f7f7f']),
            'Property-Certificate Eligibility',
        ),
    ]
    for ax, matrix, statuses, cmap, title in panels:
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect='auto')
        ax.set_xticks(range(3)); ax.set_xticklabels(op_labels, fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels(ds_labels, fontsize=7.5)
        ax.set_title(title, fontsize=9, pad=6)
        ax.set_xlabel('T-Norm Operator', fontsize=8)

        for i, ds in enumerate(datasets):
            for j, op in enumerate(operators):
                result = conf_lut[(ds, op)]
                status = statuses[int(matrix[i, j])]
                if ax is axes[0]:
                    error = result['max_abs_error']
                    error_text = '0' if error == 0.0 else f'{error:.2e}'
                    text = f'{status}\nerr = {error_text}'
                else:
                    text = status
                ax.text(j, i, text, ha='center', va='center', fontsize=6.3,
                        color='white', fontweight='bold')

    axes[0].set_ylabel('Dataset', fontsize=8)
    fig.suptitle(
        'Specification Fidelity: Conformance and Independent Certificate Eligibility',
        fontsize=9.5,
    )

    plt.tight_layout()
    out = OUT_DIR / 'fig1_conformance_overview.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Specification Fidelity Metamorphic Matrix
# ─────────────────────────────────────────────────────────────────────────────
def fig2_metamorphic_matrix():
    datasets  = ['toy_separable_2d', 'toy_medium_5d', 'toy_noisy_10d']
    ds_labels = ['Sep. 2D', 'Med. 5D', 'Noisy 10D']
    relations = [
        'MR1_row_permutation',
        'MR2_partition_determinism',
        'MR3_contribution_monotonicity',
        'MR4_membership_boundaries',
        'MR5_explanation_additivity',
        'MR6_unit_invariance',
    ]
    rel_labels = [
        'MR1\nRow Perm.',
        'MR2\nDeterminism',
        'MR3\nContrib.\nMono.',
        'MR4\nMembership\nBdry',
        'MR5\nExplan.\nAddit.',
        'MR6\nUnit\nInvar.',
    ]

    # Aggregate worst-case (max violation, any operator) per (dataset, relation)
    passed_mat = np.ones((3, 6), dtype=bool)
    viol_mat   = np.zeros((3, 6))

    for r in metamorphic:
        ds  = r['dataset_name']
        rel = r['relation_name']
        if ds not in datasets or rel not in relations:
            continue
        i = datasets.index(ds)
        j = relations.index(rel)
        if not r['passed']:
            passed_mat[i, j] = False
        viol_mat[i, j] = max(viol_mat[i, j], r['max_violation'])

    cmap = ListedColormap(['#d62728', '#2ca02c'])   # red=FAIL, green=PASS

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.72, 2.6))
    ax.imshow(passed_mat.astype(float), cmap=cmap, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(6)); ax.set_xticklabels(rel_labels, fontsize=6.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(ds_labels, fontsize=8)

    for i in range(3):
        for j in range(6):
            v   = viol_mat[i, j]
            txt = 'PASS' if passed_mat[i, j] else 'FAIL'
            vs  = '0' if v == 0.0 else f'{v:.1e}'
            ax.text(j, i, f'{txt}\n{vs}',
                    ha='center', va='center', fontsize=6,
                    color='white', fontweight='bold')

    ax.legend(
        handles=[
            mpatches.Patch(color='#2ca02c', label='PASS'),
            mpatches.Patch(color='#d62728', label='FAIL'),
        ],
        loc='upper center', bbox_to_anchor=(0.5, -0.15),
        ncol=2, fontsize=7.5, frameon=False,
    )
    ax.set_title('Specification Fidelity: Metamorphic Relation Results', fontsize=9, pad=6)
    ax.set_xlabel('Metamorphic Relation', fontsize=8)
    ax.set_ylabel('Dataset', fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / 'fig2_metamorphic_matrix.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Property Certification CERTIFIED Rates by Certificate Type (min t-norm)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_certificate_rates():
    datasets  = ['pima_diabetes', 'heart_cleveland', 'mammographic_mass']
    ds_labels = ['Pima Diabetes', 'Heart Cleveland', 'Mammo. Mass']
    cert_types = ['Monotonicity', 'Robustness', 'Exclusion', 'Safe-Region']

    def rates_min(ds):
        s = primary_summary[f'{ds}_min']
        return [
            s['monotonicity']['certified_rate'] * 100,
            s['robustness']['mean_certified_rate'] * 100,
            s['exclusion']['certified_rate'] * 100,
            s['safe_region']['certified_rate'] * 100,
        ]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    x = np.arange(len(cert_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))

    for k, (ds, label, color) in enumerate(zip(datasets, ds_labels, colors)):
        ax.bar(x + (k - 1) * width, rates_min(ds), width,
               label=label, color=color, alpha=0.85, edgecolor='white', linewidth=0.4)

    ax.set_xticks(x); ax.set_xticklabels(cert_types, fontsize=9)
    ax.set_ylabel('CERTIFIED Rate (%)', fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_title('Property Certification: CERTIFIED Rates by Certificate Type (min t-norm)', fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out = OUT_DIR / 'fig3_certificate_rates.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Min vs Product T-Norm Certificate Rates
# ─────────────────────────────────────────────────────────────────────────────
def fig4_product_vs_min():
    datasets  = ['pima_diabetes', 'heart_cleveland', 'mammographic_mass']
    ds_labels = ['Pima', 'Heart', 'Mammo.']

    mono_min  = [primary_summary[f'{ds}_min']['monotonicity']['certified_rate'] * 100 for ds in datasets]
    mono_prod = [primary_summary[f'{ds}_product']['monotonicity']['certified_rate'] * 100 for ds in datasets]
    rob_min   = [primary_summary[f'{ds}_min']['robustness']['mean_certified_rate'] * 100 for ds in datasets]
    rob_prod  = [primary_summary[f'{ds}_product']['robustness']['mean_certified_rate'] * 100 for ds in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6), sharey=True)

    for ax, (v_min, v_prod, title) in zip(
        axes,
        [(mono_min, mono_prod, 'Monotonicity'), (rob_min, rob_prod, 'Robustness')],
    ):
        ax.bar(x - width / 2, v_min,  width, label='Min',     color='#1f77b4', alpha=0.85, edgecolor='white', lw=0.4)
        ax.bar(x + width / 2, v_prod, width, label='Product', color='#ff7f0e', alpha=0.85, edgecolor='white', lw=0.4)
        ax.set_xticks(x); ax.set_xticklabels(ds_labels, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=7.5)

    axes[0].set_ylabel('CERTIFIED Rate (%)', fontsize=9)
    fig.suptitle('Property Certification: Min vs Product T-Norm Certificate Rates', fontsize=9, y=1.02)

    plt.tight_layout()
    out = OUT_DIR / 'fig4_product_vs_min.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Bootstrap Prediction Stability
# ─────────────────────────────────────────────────────────────────────────────
def fig5_bootstrap_agreement():
    ds_order  = ['pima_diabetes', 'heart_cleveland', 'breast_cancer_diagnostic',
                 'mammographic_mass', 'parkinsons']
    ds_labels = ['Pima\nDiab.', 'Heart\nClev.', 'Breast\nCancer', 'Mammo.\nMass', 'Parkin-\nsons']

    arrays = [
        np.asarray(stability[ds]['bootstrap']['per_bootstrap_agreement'], dtype=float)
        for ds in ds_order
    ]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.78, 3.2))

    vp = ax.violinplot(arrays, positions=range(len(ds_order)),
                       showmedians=True, showextrema=True)

    for body, color in zip(vp['bodies'], colors):
        body.set_facecolor(color)
        body.set_alpha(0.72)
        body.set_edgecolor('black')
        body.set_linewidth(0.5)

    vp['cmedians'].set_color('black')
    vp['cmedians'].set_linewidth(1.5)
    for part in ('cmaxes', 'cmins', 'cbars'):
        vp[part].set_color('gray')
        vp[part].set_linewidth(0.8)

    ax.set_xticks(range(len(ds_order))); ax.set_xticklabels(ds_labels, fontsize=8)
    ax.set_ylabel('Per-Bootstrap Prediction Agreement', fontsize=9)
    ax.set_ylim(0.0, 1.06)
    ax.set_title('Bootstrap Test-Set Prediction Agreement', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out = OUT_DIR / 'fig5_bootstrap_agreement.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Finite C-Grid Balanced Accuracy Curves
# ─────────────────────────────────────────────────────────────────────────────
def fig6_finite_c_grid_accuracy_curves():
    ds_order  = ['pima_diabetes', 'heart_cleveland', 'breast_cancer_diagnostic',
                 'mammographic_mass', 'parkinsons']
    ds_labels = ['Pima Diab.', 'Heart Clev.', 'Breast Cancer', 'Mammo. Mass', 'Parkinsons']
    colors    = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.6))

    for ds, label, color in zip(ds_order, ds_labels, colors):
        grid = stability[ds]['finite_c_grid']
        c_grid = np.asarray(grid['c_grid'], dtype=float)
        bacc_by_c = grid['val_balanced_accuracies']
        baccs = np.asarray([bacc_by_c[str(c)] for c in grid['c_grid']], dtype=float)
        near_optimal_set = set(grid['near_optimal_set'])
        in_near_optimal = np.asarray(
            [c in near_optimal_set for c in grid['c_grid']], dtype=bool
        )

        ax.plot(c_grid, baccs, color=color, lw=1.5, alpha=0.65, label=label, zorder=2)

        ax.scatter(c_grid[in_near_optimal], baccs[in_near_optimal],
                   color=color, s=55, zorder=5, marker='o',
                   edgecolors='black', linewidths=0.6)
        ax.scatter(c_grid[~in_near_optimal], baccs[~in_near_optimal],
                   facecolors='white', edgecolors=color, s=35, zorder=5,
                   marker='o', linewidths=1.4)

    ax.set_xscale('log')
    ax.set_xlabel('C  (regularisation, log scale)', fontsize=9)
    ax.set_ylabel('Balanced Accuracy (validation)', fontsize=9)
    deltas = {stability[ds]['finite_c_grid']['delta_bacc'] for ds in ds_order}
    delta_text = f'{next(iter(deltas)):.3f}' if len(deltas) == 1 else 'dataset-specific'
    ax.set_title(
        f'Finite C-Grid Near-Optimal Validation Analysis (additive BAcc tolerance {delta_text})',
        fontsize=9,
    )
    ax.legend(fontsize=7.5, loc='lower left', ncol=2)
    ax.grid(alpha=0.3, linewidth=0.5)

    ax.text(0.99, 0.03, 'Filled = finite-grid near-optimal   Open = outside tolerance',
            transform=ax.transAxes, fontsize=6.5, ha='right', va='bottom',
            color='gray', style='italic')

    plt.tight_layout()
    out = OUT_DIR / 'fig6_rashomon_accuracy_curves.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Certificate Status Retention
# ─────────────────────────────────────────────────────────────────────────────
def fig7_certificate_retention():
    ds_order  = ['pima_diabetes', 'heart_cleveland', 'breast_cancer_diagnostic',
                 'mammographic_mass', 'parkinsons']
    ds_labels = ['Pima\nDiab.', 'Heart\nClev.', 'Breast\nCancer', 'Mammo.\nMass', 'Parkin-\nsons']

    cert_results = [stability[ds]['monotonicity_certificate_retention'] for ds in ds_order]
    certified_rates = [result['certified_rate'] for result in cert_results]
    retention_rates = [result['reference_status_retention_rate'] for result in cert_results]
    ref_stats = [result['reference_status'] for result in cert_results]
    feat_names = [result['feature_name'] for result in cert_results]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.72, 2.9))
    x = np.arange(len(ds_order))
    width = 0.36

    ax.bar(x - width / 2, certified_rates, width, color='#1f77b4', alpha=0.85,
           edgecolor='black', linewidth=0.5, label='Certified rate')
    ax.bar(x + width / 2, retention_rates, width, color='#ff7f0e', alpha=0.85,
           edgecolor='black', linewidth=0.5, label='Reference-status retention')

    for xi, (retention, status, feat) in enumerate(
        zip(retention_rates, ref_stats, feat_names)
    ):
        ax.text(xi + width / 2, retention + 0.025, f'Ref: {status}',
                ha='center', va='bottom', fontsize=5.8, fontweight='bold', rotation=35)
        ax.text(xi, -0.07, feat, ha='center', va='top', fontsize=5.5,
                color='gray', style='italic', rotation=0)

    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_xticks(x); ax.set_xticklabels(ds_labels, fontsize=8)
    ax.set_ylabel('Bootstrap Fraction', fontsize=9)
    ax.set_ylim(0, 1.27)
    ax.set_title('Monotonicity Certificate Outcomes Across Bootstrap Resamples', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out = OUT_DIR / 'fig7_certificate_retention.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX Tables
# ─────────────────────────────────────────────────────────────────────────────
def generate_latex_tables():
    status_order = [
        ('CERTIFIED', 'C'),
        ('CERTIFIED-TRIVIAL', 'CT'),
        ('UNKNOWN', 'U'),
        ('COUNTEREXAMPLE', 'CE'),
        ('NO-MATCH', 'NM'),
    ]

    def _latex_escape(value):
        return str(value).replace('_', r'\_').replace('%', r'\%')

    def _status_label(status):
        return rf'\textsc{{{_latex_escape(status)}}}'

    def _status_counts(counts):
        known = {name for name, _ in status_order}
        entries = [f'{abbr}={counts.get(name, 0)}' for name, abbr in status_order]
        entries.extend(
            f'{_latex_escape(name)}={count}'
            for name, count in sorted(counts.items())
            if name not in known
        )
        return ', '.join(entries)

    def _certificate_cell(certificate, rate_key):
        rate = 100.0 * certificate[rate_key]
        return rf'{rate:.1f}\% [{_status_counts(certificate["status_counts"])}]'

    lines = [
        "% Auto-generated verification tables.",
        "% Regenerate with:  uv run python scripts/generate_verification_figures.py",
        "%",
        "% Requires \\usepackage{booktabs,graphicx} in the main .tex file.",
        "",
    ]

    # ── Table 1: Specification Fidelity Conformance ────────────────────────────────────────
    lines += [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Specification Fidelity reports two independent dimensions. Conformance",
        r"\textsc{Certified} means the measured reference-production error is below the",
        r"recorded tolerance; \textsc{Counterexample} identifies a measured discrepancy, and",
        r"\textsc{Unknown} means no conformance determination. Property-certificate eligibility",
        r"indicates whether the operator is covered by the certificate soundness assumptions;",
        r"ineligibility does not negate measured conformance.}",
        r"\label{tab:conformance}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Dataset & Operator & $n$ & $d$ & Conformance & Eligibility & Max Error & Mean Error \\",
        r"\midrule",
    ]

    conf_lut = {(r['dataset_name_raw'], r['and_operator']): r for r in conformance}
    ds_display = {
        'toy_separable_2d': r'Toy Sep.\ 2D',
        'toy_medium_5d':    r'Toy Med.\ 5D',
        'toy_noisy_10d':    'Toy Noisy 10D',
    }
    ds_order = ['toy_separable_2d', 'toy_medium_5d', 'toy_noisy_10d']
    op_order = ['min', 'product', 'softmin']

    for i, ds in enumerate(ds_order):
        for j, op in enumerate(op_order):
            r = conf_lut[(ds, op)]
            max_err = r['max_abs_error']
            mean_err = r['mean_abs_error']
            ds_str = ds_display[ds] if j == 0 else ''
            max_error_text = r'$0$' if max_err == 0.0 else f'${max_err:.2e}$'
            mean_error_text = r'$0$' if mean_err == 0.0 else f'${mean_err:.2e}$'
            lines.append(
                f'  {ds_str} & {op} & {r["n"]} & {r["d"]} & '
                f'{_status_label(r["status"])} & '
                f'{_status_label(r["certificate_eligibility_status"])} & '
                f'{max_error_text} & {mean_error_text} \\\\'
            )
        if i < len(ds_order) - 1:
            lines.append(r'  \midrule')

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ]

    # Table 2: primary property-certificate outcomes
    lines += [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Primary-configuration predictive balanced accuracy (BAcc) and property-certificate",
        r"outcomes. Each certificate cell gives the certified rate followed by complete status counts",
        r"[C, CT, U, CE, NM]. C is \textsc{Certified}; CT is \textsc{Certified-Trivial}; U is",
        r"\textsc{Unknown}, which is inconclusive and not a counterexample; CE is \textsc{Counterexample}",
        r"with a concrete witness; and NM is \textsc{No-Match} for an unresolved declared feature.",
        r"Rates retain all attempted cases in the denominator and combine C with CT where the primary",
        r"summary defines CT as certified.}",
        r"\label{tab:certificate-rates}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Dataset & Operator & BAcc & Monotonicity & Robustness & Exclusion & Safe Region \\",
        r"\midrule",
    ]

    ds01 = ['pima_diabetes', 'heart_cleveland', 'mammographic_mass']
    ds01_disp = {
        'pima_diabetes':     'Pima Diabetes',
        'heart_cleveland':   'Heart Cleveland',
        'mammographic_mass': r'Mammo.\ Mass',
    }

    for dataset_index, ds in enumerate(ds01):
        for operator_index, operator in enumerate(('min', 'product')):
            result = primary_summary[f'{ds}_{operator}']
            dataset_text = ds01_disp[ds] if operator_index == 0 else ''
            lines.append(
                f'  {dataset_text} & {operator} & '
                f'{result["mean_held_out_balanced_accuracy"]:.3f} & '
                f'{_certificate_cell(result["monotonicity"], "certified_rate")} & '
                f'{_certificate_cell(result["robustness"], "mean_certified_rate")} & '
                f'{_certificate_cell(result["exclusion"], "certified_rate")} & '
                f'{_certificate_cell(result["safe_region"], "certified_rate")} \\\\'
            )
        if dataset_index < len(ds01) - 1:
            lines.append(r'  \midrule')

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ]

    # Table 3: bootstrap and finite C-grid prediction stability
    lines += [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Predictive performance and stability. Bootstrap agreement is summarized over",
        r"per-bootstrap test-set agreement values as mean [q05--q95]. Grid size is the number of",
        r"finite C-grid values in the additive-BAcc near-optimal set divided by the full grid size.",
        r"Grid agreement is the fraction of held-out samples for which all near-optimal models agree.",
        r"Reference and grid-selected predictive performance are held-out balanced accuracies.}",
        r"\label{tab:stability}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Dataset & Ref. BAcc & Boot. Agr. mean [q05--q95] & Grid Size & Grid Agr. & Grid-Best BAcc \\",
        r"\midrule",
    ]

    ds03_order = ['pima_diabetes', 'heart_cleveland', 'breast_cancer_diagnostic',
                  'mammographic_mass', 'parkinsons']
    ds03_disp = {
        'pima_diabetes':           'Pima Diabetes',
        'heart_cleveland':         'Heart Cleveland',
        'breast_cancer_diagnostic':'Breast Cancer',
        'mammographic_mass':       r'Mammo.\ Mass',
        'parkinsons':              'Parkinsons',
    }

    for ds in ds03_order:
        bootstrap = stability[ds]['bootstrap']
        grid = stability[ds]['finite_c_grid']
        lines.append(
            f'  {ds03_disp[ds]} & {bootstrap["reference_test_balanced_accuracy"]:.3f} & '
            f'{bootstrap["mean_prediction_agreement"]:.3f} '
            f'[{bootstrap["q05_prediction_agreement"]:.3f}--'
            f'{bootstrap["q95_prediction_agreement"]:.3f}] & '
            f'{grid["near_optimal_size"]}/{len(grid["c_grid"])} & '
            f'{grid["test_prediction_agreement"]:.3f} & '
            f'{grid["best_model_test_balanced_accuracy"]:.3f} \\\\'
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ]

    # Table 4: monotonicity certificate outcomes over bootstrap models
    lines += [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Bootstrap monotonicity certificate outcomes. Status counts use C, CT, U, CE,",
        r"and NM as defined in Table~\ref{tab:certificate-rates}. Certified rate combines C and CT;",
        r"reference-status retention is the fraction exactly matching the displayed reference",
        r"status. Thus a reference status of \textsc{Unknown} remains unknown and is never",
        r"interpreted as a counterexample.}",
        r"\label{tab:monotonicity-bootstrap-status}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllccc}",
        r"\toprule",
        r"Dataset & Feature & Reference Status & Status Counts & Certified Rate & Ref.-Status Retention \\",
        r"\midrule",
    ]

    for ds in ds03_order:
        certificate = stability[ds]['monotonicity_certificate_retention']
        lines.append(
            f'  {ds03_disp[ds]} & {_latex_escape(certificate["feature_name"])} & '
            f'{_status_label(certificate["reference_status"])} & '
            f'{_status_counts(certificate["status_counts"])} & '
            f'{100.0 * certificate["certified_rate"]:.1f}\\% & '
            f'{100.0 * certificate["reference_status_retention_rate"]:.1f}\\% \\\\'
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ]

    out = Path("paper/generated_tables.tex")
    out.write_text("\n".join(lines))
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n=== Generating verification figures ===")
    fig1_conformance_overview()
    fig2_metamorphic_matrix()
    fig3_certificate_rates()
    fig4_product_vs_min()
    fig5_bootstrap_agreement()
    fig6_finite_c_grid_accuracy_curves()
    fig7_certificate_retention()

    print("\n=== Generating LaTeX tables ===")
    generate_latex_tables()

    print("\n=== Summary ===")
    print("Figures saved to  : paper/figures/")
    print("Tables saved to   : paper/generated_tables.tex")
