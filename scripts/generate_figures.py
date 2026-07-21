"""Generate paper figures for the FuzzyRuleSVM paper.

Produces eight PDF figures saved to paper/figures/:
  fig_architecture.pdf       - Pipeline architecture diagram (Figure 0 / intro)
  fig_waterfall.pdf          - Rule contribution waterfall (Figure 1)
  fig_membership.pdf         - Fuzzy membership functions (Figure 2)
  fig_scatter.pdf            - Per-dataset paired scatter vs FURIA (Figure 3)
  fig_cd.pdf                 - Critical Difference diagram (Figure 5)
  fig_interpretability.pdf   - Interpretability analysis: compactness, stability,
                               dimensionality (Figure 6)
  fig_ablation_semantic.pdf  - Side-by-side MembershipSVM vs FuzzyRuleSVM
                               explanation panel (Figure 7)
  fig_ablation_delta.pdf     - Per-dataset FRS - MEM balanced accuracy delta
                               bar chart (Figure 8)

Usage:
    uv run python scripts/generate_figures.py
    uv run python scripts/generate_figures.py --data-dir datasets/prepared \
        --runs-dir runs --output-dir paper/figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from fysvm.datasets import load_dataset
from fysvm.rule_svm import FuzzyRuleSVM, _linear_down, _linear_up


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

_COLORS = {
    "positive": "#2166ac",   # blue
    "negative": "#d6604d",   # red/orange
    "neutral":  "#4dac26",   # green (bias / net)
    "frs":      "#1b7837",
    "furia":    "#762a83",
    "zero":     "#bbbbbb",
}

plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":    150,
    "savefig.dpi":   300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})


# ---------------------------------------------------------------------------
# Figure 0 – Architecture pipeline diagram
# ---------------------------------------------------------------------------

def figure_architecture(output_dir: Path) -> None:
    """Six-stage pipeline architecture diagram for FuzzyRuleSVM.

    Illustrative values are chosen to be internally consistent and to match
    the running Pima Diabetes example quoted in Section 3.4 of the paper:
        R1: IF glucose is high AND bmi is high  → fires at 0.81, β=+0.90, contrib=+0.73
        R2: IF age is high AND glucose is high  → fires at 0.67, β=+0.50, contrib=+0.34
        R3: IF glucose is medium AND bmi is low → fires at 0.18, β=−0.80, contrib=−0.14
        Net margin: f(x) = +0.93 → class +1

    Layout: two rows of three stages (3 stages per row), which keeps each box
    wide enough to display content clearly at the paper's textwidth (~6.3 in).
    An elbow arrow connects the end of row 1 to the start of row 2.
    """

    # --- Layout constants (all in data-coordinate units = inches) -----------
    # Designed at ~6.5 in wide to match paper textwidth so font sizes are
    # correct at final print size (no unwanted scale-down).
    FIG_W, FIG_H = 6.5, 5.5
    N_COL = 3          # stages per row
    BOX_W = 1.80       # box width
    GAP_H = 0.25       # horizontal gap between boxes in the same row
    GAP_V = 0.48       # vertical gap between the two rows
    BOX_H = 2.05       # total box height (header + content)
    HDR_H = 0.30       # header strip height

    total_w = N_COL * BOX_W + (N_COL - 1) * GAP_H  # = 5.90
    LEFT_MARGIN = (FIG_W - total_w) / 2              # = 0.30

    # Row bounding boxes (top / bottom y-coords)
    ROW1_TOP = FIG_H - 0.12   # 5.38
    ROW1_BOT = ROW1_TOP - BOX_H   # 3.33
    ROW2_TOP = ROW1_BOT - GAP_V   # 2.85
    ROW2_BOT = ROW2_TOP - BOX_H   # 0.80

    def _box_coords(stage_idx: int) -> tuple[float, float, float]:
        """Return (x0, box_bot, box_top) for stage_idx (0-based)."""
        col = stage_idx % N_COL
        row = stage_idx // N_COL
        x0 = LEFT_MARGIN + col * (BOX_W + GAP_H)
        top = ROW1_TOP if row == 0 else ROW2_TOP
        bot = ROW1_BOT if row == 0 else ROW2_BOT
        return x0, bot, top

    # --- Colour palette ----------------------------------------------------
    HDR_BG  = "#1b7837"
    HDR_FG  = "white"
    BOX_BG  = "#f5faf5"
    BOX_EDG = "#1b7837"
    POS_COL = "#2166ac"
    NEG_COL = "#d6604d"
    ZRO_COL = "#888888"
    ARR_COL = "#555555"

    # Font sizes (actual print sizes, since figure is designed at textwidth)
    HDR_FS  = 7.5
    CON_FS  = 7.0
    SUB_FS  = 7.0

    # --- Stage definitions -------------------------------------------------
    stages = [
        {
            "title": "1 · Input $x$",
            "lines": [
                "glucose = 140",
                "bmi     = 28.5",
                "age     = 45",
                r"$\vdots$",
            ],
            "sublabel": r"$x \in \mathbb{R}^d$",
        },
        {
            "title": "2 · Fuzzy Partitions",
            "lines": [
                r"$\mu_H$(glucose) = 0.95",
                r"$\mu_M$(bmi) = 0.76",
                r"$\mu_H$(age) = 0.73",
                r"$\vdots$ ($3d$ values)",
            ],
            "sublabel": r"$\mu_{jt}(x) \in [0,1]$",
        },
        {
            "title": "3 · Rule Bank",
            "lines": [
                r"$R_1$: gluc.$H$ $\wedge$ bmi.$H$",
                r"$R_2$: age.$H$ $\wedge$ gluc.$H$",
                r"$R_3$: gluc.$M$ $\wedge$ bmi.$L$",
                r"$\vdots$ (at most $K_{\max}$)",
            ],
            "sublabel": r"$\{R_k\}_{k=1}^{K}$",
        },
        {
            "title": r"4 · Activation $\Phi(x)$",
            "lines": [
                r"$\phi_1(x)=0.81$  $(R_1)$",
                r"$\phi_2(x)=0.67$  $(R_2)$",
                r"$\phi_3(x)=0.18$  $(R_3)$",
                r"$\phi_4(x)=0.00$ (zero)",
                r"$\vdots$ ($K$ values)",
            ],
            "sublabel": r"$\Phi(x)\in[0,1]^K$",
        },
        {
            "title": r"5 · Regularised $\beta$",
            "lines": [
                r"POS:$\beta_1=+0.90$  $(R_1)$",
                r"POS:$\beta_2=+0.50$  $(R_2)$",
                r"NEG:$\beta_3=-0.80$  $(R_3)$",
                r"ZRO:$\beta_4=\;0.00$ (zeroed)",
                "L1 $\\Rightarrow$ sparse",
            ],
            "sublabel": r"$\beta\in\mathbb{R}^K$ (L1/L2)",
        },
        {
            "title": r"6 · Prediction $f(x)$",
            "lines": [
                r"POS:$+0.73$  $R_1$: gluc.$H$$\wedge$bmi.$H$",
                r"POS:$+0.34$  $R_2$: age.$H$$\wedge$gluc.$H$",
                r"NEG:$-0.14$  $R_3$: gluc.$M$$\wedge$bmi.$L$",
                "RULE_SEP",
                r"$f(x) = +0.93 \;\to$ class $+1$",
            ],
            "sublabel": r"$f(x)=\sum_k\beta_k\phi_k(x)+b$",
        },
    ]

    # --- Drawing -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    for i, stage in enumerate(stages):
        x0, BOX_BOT, BOX_TOP = _box_coords(i)
        xc = x0 + BOX_W / 2

        # ── outer box ────────────────────────────────────────────────────
        ax.add_patch(mpatches.Rectangle(
            (x0, BOX_BOT), BOX_W, BOX_H,
            facecolor=BOX_BG, edgecolor=BOX_EDG, linewidth=1.0, zorder=2,
        ))

        # ── header strip ─────────────────────────────────────────────────
        ax.add_patch(mpatches.Rectangle(
            (x0, BOX_TOP - HDR_H), BOX_W, HDR_H,
            facecolor=HDR_BG, edgecolor="none", zorder=3,
        ))
        ax.plot(
            [x0, x0 + BOX_W], [BOX_TOP - HDR_H, BOX_TOP - HDR_H],
            color=BOX_EDG, linewidth=0.6, zorder=4,
        )
        ax.text(
            xc, BOX_TOP - HDR_H / 2, stage["title"],
            ha="center", va="center",
            fontsize=HDR_FS, fontweight="bold", color=HDR_FG, zorder=5,
        )

        # ── content lines ─────────────────────────────────────────────────
        content_top = BOX_TOP - HDR_H - 0.10
        n_lines = len(stage["lines"])
        available_h = BOX_H - HDR_H - 0.15
        line_h = available_h / (n_lines + 0.3)

        sep_y = None
        for j, raw_line in enumerate(stage["lines"]):
            y = content_top - j * line_h
            if raw_line.startswith("POS:"):
                color, line = POS_COL, raw_line[4:]
            elif raw_line.startswith("NEG:"):
                color, line = NEG_COL, raw_line[4:]
            elif raw_line.startswith("ZRO:"):
                color, line = ZRO_COL, raw_line[4:]
            elif raw_line == "RULE_SEP":
                sep_y = y + line_h * 0.35
                continue
            else:
                color, line = "#222222", raw_line
            ax.text(
                x0 + 0.09, y, line,
                ha="left", va="top",
                fontsize=CON_FS, color=color, zorder=5,
            )

        if sep_y is not None:
            ax.plot(
                [x0 + 0.08, x0 + BOX_W - 0.08], [sep_y, sep_y],
                color="#888888", linewidth=0.7, linestyle="-", zorder=5,
            )

        # ── sub-label below the box ───────────────────────────────────────
        ax.text(
            xc, BOX_BOT - 0.09, stage["sublabel"],
            ha="center", va="top",
            fontsize=SUB_FS, color="#444444", style="italic", zorder=5,
        )

        # ── arrows ───────────────────────────────────────────────────────
        col = i % N_COL
        row = i // N_COL
        y_mid = (BOX_BOT + BOX_TOP) / 2

        if col < N_COL - 1:
            # Horizontal arrow to the next box in the same row
            x_start = x0 + BOX_W + 0.02
            x_end   = x0 + BOX_W + GAP_H - 0.02
            ax.annotate(
                "",
                xy=(x_end, y_mid), xytext=(x_start, y_mid),
                arrowprops=dict(
                    arrowstyle="->", color=ARR_COL, lw=1.2, mutation_scale=10,
                ),
                zorder=6,
            )

    # ── Cross-row elbow arrow: right of box[2] → inter-row gap → top of box[3] ──
    # Route through the inter-row gap (not through the row-2 box interiors).
    x2_right = LEFT_MARGIN + 2 * (BOX_W + GAP_H) + BOX_W + 0.03
    y_r1_mid = (ROW1_BOT + ROW1_TOP) / 2
    x_elbow  = x2_right + 0.13   # step right before going down
    # Land in the middle of the inter-row gap, well below row-1 sub-labels
    y_gap    = ROW2_TOP + (ROW1_BOT - ROW2_TOP) * 0.45   # ~0.22 in above row-2 top
    # Target the horizontal centre of box[3] (stage index 3, first box in row 2)
    x3_cx    = LEFT_MARGIN + BOX_W / 2

    ax.plot(
        [x2_right, x_elbow, x_elbow, x3_cx],
        [y_r1_mid, y_r1_mid, y_gap,   y_gap],
        color=ARR_COL, linewidth=1.0, solid_capstyle="round", zorder=6,
    )
    ax.annotate(
        "",
        xy=(x3_cx, ROW2_TOP),   # arrowhead at top-centre of box[3] (row 2)
        xytext=(x3_cx, y_gap),  # stem starts at the gap-level line
        arrowprops=dict(arrowstyle="->", color=ARR_COL, lw=1.0, mutation_scale=9),
        zorder=7,
    )

    fig.tight_layout(pad=0.2)
    out = output_dir / "fig_architecture.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 1 – Rule contribution waterfall
# ---------------------------------------------------------------------------

def _fit_pima_model(data_dir: Path) -> tuple[FuzzyRuleSVM, np.ndarray, np.ndarray, list[str]]:
    """Fit FuzzyRuleSVM on a training split of Pima Diabetes."""
    ds = load_dataset("pima_diabetes", data_dir)
    X, y = ds.X, np.asarray(ds.y)

    # Reproducible 80/20 split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", FuzzyRuleSVM(
            C=1.0,
            penalty="l1",
            max_rule_length=2,
            max_rules=64,
            min_rule_coverage=0.01,
            rule_length_penalty=0.35,
            feature_names=ds.feature_names,
            class_weight="balanced",
            random_state=0,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    model: FuzzyRuleSVM = pipe.named_steps["model"]
    X_te_imp = pipe.named_steps["imputer"].transform(X_te)
    return model, X_te_imp, y_te, ds.feature_names


def _pick_median_instance(
    model: FuzzyRuleSVM,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> int:
    """Return the index of the instance closest to median explanation complexity.

    Complexity is measured by the number of contributing rules (|contrib| > 0.01),
    breaking ties by proximity to the median absolute margin.
    """
    expls = model.explain(X_test, top_n=len(model.rules_), min_abs_contribution=0.0)
    n_contributing = np.array([
        len([r for r in e["top_rules"] if abs(r["contribution"]) > 0.01])
        for e in expls
    ])
    margins = np.abs(model.decision_function(X_test))
    median_contrib = float(np.median(n_contributing))
    # Among instances with median rule count, pick one with median margin
    candidates = np.where(n_contributing == int(round(median_contrib)))[0]
    if len(candidates) == 0:
        candidates = np.arange(len(X_test))
    median_margin = float(np.median(margins[candidates]))
    dists = np.abs(margins[candidates] - median_margin)
    return int(candidates[np.argmin(dists)])


def figure_waterfall(data_dir: Path, output_dir: Path) -> None:
    """Rule contribution waterfall for a median-complexity Pima Diabetes instance."""
    model, X_te, y_te, feature_names = _fit_pima_model(data_dir)
    idx = _pick_median_instance(model, X_te, y_te)

    expl = model.explain(X_te[[idx]], top_n=20, min_abs_contribution=0.01)[0]
    rules = expl["top_rules"]
    # Sort descending by absolute contribution
    rules = sorted(rules, key=lambda r: abs(r["contribution"]), reverse=True)
    # Keep at most 10 for readability
    rules = rules[:10]

    labels = []
    values = []
    colors = []
    for r in rules:
        # Strip "IF ... THEN ..." to keep just the antecedent part, wrapped
        antecedent = r["rule"]
        # Remove consequent
        if " THEN " in antecedent:
            antecedent = antecedent.split(" THEN ")[0]
        # Shorten: remove "IF " prefix, replace " AND " with "\n∧ "
        antecedent = antecedent.replace("IF ", "").replace(" AND ", "\n∧ ")
        labels.append(antecedent)
        values.append(r["contribution"])
        colors.append(_COLORS["positive"] if r["contribution"] > 0 else _COLORS["negative"])

    # Add bias as its own bar
    labels.append("bias")
    values.append(expl["bias"])
    colors.append(_COLORS["neutral"])

    n = len(labels)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(5.5, 0.45 * n + 0.6))
    bars = ax.barh(y_pos, values, color=colors, height=0.65, edgecolor="white", linewidth=0.5)

    label_pad = 0.012
    ax.set_xlim(
        min(min(values), 0.0) - 0.12,
        max(max(values), 0.0) + 0.05,
    )

    # Annotate bars
    for bar, val in zip(bars, values):
        x_text = val + (label_pad if val >= 0 else -label_pad)
        ha = "left" if val >= 0 else "right"
        ax.text(x_text, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=7.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.7, linestyle="-")

    margin = expl["margin"]
    pred = expl["prediction"]
    ax.text(
        0.02, 0.98,
        f"Prediction: {pred}; $f(x)$ = {margin:+.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    ax.set_xlabel("Rule contribution to margin $f(x)$")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = output_dir / "fig_waterfall.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 – Fuzzy membership functions
# ---------------------------------------------------------------------------

def figure_membership(data_dir: Path, output_dir: Path) -> None:
    """Membership functions for glucose, BMI, and age on Pima Diabetes."""
    model, _, _, _ = _fit_pima_model(data_dir)
    ds = load_dataset("pima_diabetes", data_dir)

    # Feature indices (from ds.feature_names)
    feature_indices = {
        "glucose":      ds.feature_names.index("glucose"),
        "bmi":          ds.feature_names.index("bmi"),
        "age":          ds.feature_names.index("age"),
    }
    feature_labels = {
        "glucose": "Plasma glucose (mg/dL)",
        "bmi":     "Body mass index (kg/m²)",
        "age":     "Age (years)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2))
    term_styles = {
        "low":    {"color": "#4575b4", "linestyle": "-",  "label": "low"},
        "medium": {"color": "#fdae61", "linestyle": "--", "label": "medium"},
        "high":   {"color": "#d73027", "linestyle": "-.", "label": "high"},
    }
    term_order = ["low", "medium", "high"]

    for ax, (feat_key, feat_idx) in zip(axes, feature_indices.items()):
        partition = model.partitions_[feat_idx]
        lo, med, hi = partition.low, partition.medium, partition.high
        # Range: a bit beyond the [lo, hi] support
        x_min = lo - 0.15 * (hi - lo)
        x_max = hi + 0.15 * (hi - lo)
        x = np.linspace(x_min, x_max, 400)

        mu_low    = _linear_down(x, lo, med)
        mu_medium = np.minimum(
            _linear_up(x, lo, med),
            _linear_down(x, med, hi),
        )
        mu_high   = _linear_up(x, med, hi)
        memberships = {"low": mu_low, "medium": mu_medium, "high": mu_high}

        for term in term_order:
            st = term_styles[term]
            ax.plot(x, memberships[term],
                    color=st["color"], linestyle=st["linestyle"],
                    linewidth=1.5, label=st["label"])

        # Mark the three quantile anchors
        for anchor, label_str in [(lo, "q₅"), (med, "q₅₀"), (hi, "q₉₅")]:
            ax.axvline(anchor, color="gray", linewidth=0.6, linestyle=":", alpha=0.8)
            ax.text(anchor, 1.03, label_str, ha="center", fontsize=6.5,
                    color="gray", transform=ax.get_xaxis_transform())

        ax.set_xlabel(feature_labels[feat_key])
        ax.set_ylabel("$\\mu$" if ax is axes[0] else "")
        ax.set_ylim(-0.04, 1.12)
        ax.set_xlim(x_min, x_max)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks([0.0, 0.5, 1.0])

    # Shared legend under the three panels
    handles = [
        plt.Line2D([0], [0], color=term_styles[t]["color"],
                   linestyle=term_styles[t]["linestyle"], linewidth=1.5,
                   label=t)
        for t in term_order
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06), framealpha=0.9, fontsize=8)

    fig.suptitle(
        "Data-adaptive low / medium / high partitions (Pima Diabetes)",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    out = output_dir / "fig_membership.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 – Per-dataset paired scatter FuzzyRuleSVM vs FURIA
# ---------------------------------------------------------------------------

DATASET_SHORT_NAMES = {
    "breast_cancer_diagnostic":          "BC Diagnostic",
    "breast_cancer_original":            "BC Original",
    "mammographic_mass":                 "Mammo. Mass",
    "breast_tissue":                     "Breast Tissue",
    "heart_cleveland":                   "Heart Cleveland",
    "statlog_heart":                     "Statlog Heart",
    "spect_heart":                       "SPECT Heart",
    "spectf_heart":                      "SPECTF Heart",
    "pima_diabetes":                     "Pima Diabetes",
    "diabetic_retinopathy_debrecen":     "Diab. Retinop.",
    "parkinsons":                        "Parkinsons",
    "parkinsons_disease_classification": "Parkinson's Cls",
    "ilpd":                              "ILPD (Liver)",
    "dermatology":                       "Dermatology",
    "haberman_survival":                 "Haberman",
    "vertebral_column_2c":               "Vertebral 2C",
    "arrhythmia_binary":                 "Arrhythmia Bin.",
    "iris":                              "Iris",
    "wine":                              "Wine",
    "digits":                            "Digits",
}


def figure_scatter(runs_dir: Path, output_dir: Path) -> None:
    """Per-dataset paired scatter: FuzzyRuleSVM vs FURIA balanced accuracy."""
    metrics_path = runs_dir / "fuzzy-baselines-comparison" / "metrics.csv"
    df = pd.read_csv(metrics_path)

    frs = df[df["model_key"] == "fuzzy_rule_svm"].set_index("dataset")
    furia = df[df["model_key"] == "furia"].set_index("dataset")
    common = sorted(set(frs.index) & set(furia.index))

    x_vals = furia.loc[common, "balanced_accuracy_mean"].values
    y_vals = frs.loc[common, "balanced_accuracy_mean"].values
    names = [DATASET_SHORT_NAMES.get(d, d) for d in common]

    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    # Reference diagonal
    lim_lo = min(x_vals.min(), y_vals.min()) - 0.03
    lim_hi = max(x_vals.max(), y_vals.max()) + 0.03
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color="gray", linewidth=0.9, linestyle="--", zorder=1)

    # Scatter points
    ax.scatter(x_vals, y_vals, s=40, color=_COLORS["frs"],
               edgecolors="white", linewidths=0.5, zorder=3, alpha=0.92)

    # Dataset labels – simple offset, shift left/right to avoid diagonal
    for x, y, name in zip(x_vals, y_vals, names):
        # Offset away from diagonal
        dx = 0.003
        dy = 0.003
        if y > x:
            dy = 0.010
        ax.annotate(
            name, (x, y),
            xytext=(x + dx, y + dy),
            fontsize=5.5,
            ha="left", va="bottom",
            color="#333333",
        )

    ax.set_xlabel("FURIA — balanced accuracy")
    ax.set_ylabel("FuzzyRuleSVM — balanced accuracy")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Win/tie/loss annotation
    wins = int(np.sum(y_vals > x_vals))
    ax.text(0.03, 0.97, f"W/T/L = {wins}/0/{len(common)-wins}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color=_COLORS["frs"])

    fig.tight_layout()
    out = output_dir / "fig_scatter.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 5 – Critical Difference diagram
# ---------------------------------------------------------------------------

# Nemenyi critical q_alpha values (two-tailed, alpha=0.05)
# indexed by number of classifiers k.
_NEMENYI_Q: dict[int, float] = {
    2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def _nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Compute the Nemenyi critical difference for k classifiers, n datasets."""
    q = _NEMENYI_Q.get(k, 2.569)
    return q * np.sqrt(k * (k + 1) / (6 * n))


def figure_cd(runs_dir: Path, output_dir: Path) -> None:
    """Critical Difference diagram for the four fuzzy classifiers."""
    metrics_path = runs_dir / "fuzzy-baselines-comparison" / "metrics.csv"
    df = pd.read_csv(metrics_path)

    model_order = ["fuzzy_rule_svm", "furia", "farchd", "ivturs"]
    model_labels = {
        "fuzzy_rule_svm": "FuzzyRuleSVM",
        "furia":          "FURIA",
        "farchd":         "FARC-HD†",
        "ivturs":         "IVTURS†",
    }

    # Pivot: rows=datasets, columns=models
    pivot = df.pivot(index="dataset", columns="model_key",
                     values="balanced_accuracy_mean")
    pivot = pivot[model_order].dropna()
    n_datasets = len(pivot)

    # Rank within each dataset (1 = best = highest accuracy)
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    avg_ranks = ranks.mean(axis=0)

    k = len(model_order)
    cd = _nemenyi_cd(k, n_datasets)

    # Sort by average rank (best = leftmost on the axis)
    sorted_models = avg_ranks.sort_values().index.tolist()
    sorted_ranks = avg_ranks[sorted_models].values
    sorted_labels = [model_labels[m] for m in sorted_models]

    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    ax.set_xlim(0.75, k + 0.25)
    ax.set_ylim(-1.8, 1.2)
    ax.axis("off")

    # Draw rank axis
    ax.plot([1, k], [0, 0], color="black", linewidth=1.5, zorder=2)
    for r in range(1, k + 1):
        ax.plot([r, r], [-0.08, 0.08], color="black", linewidth=1.2, zorder=2)
        ax.text(r, -0.25, str(r), ha="center", va="top", fontsize=8)
    ax.text((1 + k) / 2, -0.55, "Average rank", ha="center", va="top", fontsize=8)

    # Draw classifier positions and name labels
    # Alternate above/below for readability
    label_y = [0.55, 0.90, 0.55, 0.90]
    for i, (r, lbl, ly) in enumerate(zip(sorted_ranks, sorted_labels, label_y)):
        ax.plot([r, r], [0, ly - 0.12], color="#555555", linewidth=0.8,
                linestyle=":", zorder=1)
        ax.plot(r, 0, "o", color=_COLORS["frs"] if sorted_models[i] == "fuzzy_rule_svm"
                else "#555555", markersize=7, zorder=3)
        ax.text(r, ly, lbl, ha="center", va="bottom",
                fontsize=8,
                fontweight="bold" if sorted_models[i] == "fuzzy_rule_svm" else "normal",
                color=_COLORS["frs"] if sorted_models[i] == "fuzzy_rule_svm" else "black")

    # Draw Critical Difference bar at the top
    best_rank = sorted_ranks[0]
    cd_end = best_rank + cd
    bar_y = -1.2
    ax.annotate("", xy=(cd_end, bar_y), xytext=(best_rank, bar_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text((best_rank + cd_end) / 2, bar_y - 0.22,
            f"CD = {cd:.2f}  (Nemenyi, α=0.05, N={n_datasets})",
            ha="center", va="top", fontsize=7.5)

    # Draw non-significant cliques.
    # A group is a maximal contiguous run of classifiers where
    # rank[last] - rank[first] <= CD (i.e., no pair is significantly different).
    # One horizontal thick bar is drawn per distinct group, below the axis.
    groups: list[tuple[float, float]] = []
    i = 0
    while i < k:
        # Extend group as far right as possible
        j = i
        while j + 1 < k and (sorted_ranks[j + 1] - sorted_ranks[i]) <= cd:
            j += 1
        if j > i:
            groups.append((sorted_ranks[i], sorted_ranks[j]))
        i += 1

    # Deduplicate identical spans
    seen_spans: set[tuple[float, float]] = set()
    clique_y = -0.5
    clique_step = 0.22
    for span in groups:
        if span not in seen_spans:
            seen_spans.add(span)
            ax.plot([span[0], span[1]], [clique_y, clique_y],
                    color="black", linewidth=2.5, solid_capstyle="round", zorder=4)
            clique_y -= clique_step

    ax.set_title(
        "Critical Difference diagram — four fuzzy classifiers\n"
        r"(balanced accuracy, 20 datasets; † = reduced GA budget)",
        fontsize=8.5, pad=4,
    )

    fig.tight_layout()
    out = output_dir / "fig_cd.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 6 – Interpretability analysis: compactness, stability, dimensionality
# ---------------------------------------------------------------------------

# Map dataset slugs to short display names for scatter labels
_DATASET_LABELS = {
    "breast_cancer_diagnostic":          "BC Diag.",
    "breast_cancer_original":            "BC Orig.",
    "mammographic_mass":                 "Mammo.",
    "breast_tissue":                     "Br. Tissue",
    "heart_cleveland":                   "Heart Clev.",
    "statlog_heart":                     "Statlog H.",
    "spect_heart":                       "SPECT",
    "spectf_heart":                      "SPECTF",
    "pima_diabetes":                     "Pima",
    "diabetic_retinopathy_debrecen":     "Diab. Ret.",
    "parkinsons":                        "Parkinsons",
    "parkinsons_disease_classification": "PD Classif.",
    "ilpd":                              "ILPD",
    "dermatology":                       "Dermatol.",
    "haberman_survival":                 "Haberman",
    "vertebral_column_2c":               "Vertebral",
    "arrhythmia_binary":                 "Arrhythmia",
    "iris":                              "Iris",
    "wine":                              "Wine",
    "digits":                            "Digits",
}

# Which datasets to highlight as high-dimensional (> 30 features)
_HIGH_DIM_DATASETS = {
    "arrhythmia_binary",
    "parkinsons_disease_classification",
    "digits",
    "spectf_heart",
    "dermatology",
    "parkinsons",
    "spect_heart",
}


def figure_interpretability(runs_dir: Path, output_dir: Path) -> None:
    """Three-panel interpretability analysis figure.

    Panel A – Compactness boxplot: rules needed for 90% decision mass by dataset
              (one box per dataset from 5-fold CV, sorted by median).
    Panel B – Stability vs compactness scatter: support-rule Jaccard vs median
              rules needed for 90% contribution (one point per dataset).
    Panel C – Dimensionality vs explanation size: number of features vs median
              rules needed for 90% contribution (one point per dataset).
    """
    fold_path = runs_dir / "modern-baselines-comparison" / "fold_metrics.csv"
    summary_path = runs_dir / "modern-baselines-comparison" / "metrics.csv"

    import pandas as pd  # noqa: PLC0415

    fold_df = pd.read_csv(fold_path)
    summary_df = pd.read_csv(summary_path)

    frs_fold = fold_df[fold_df["model_key"] == "fuzzy_rule_svm"].copy()
    frs_summary = summary_df[summary_df["model_key"] == "fuzzy_rule_svm"].copy()

    # Per-dataset median rules_for_90pct (for sorting in boxplot)
    median_90 = (
        frs_fold.groupby("dataset")["rule_mean_rules_for_90pct_contribution"]
        .median()
    )
    # Sort datasets by ascending median for boxplot
    datasets_sorted = median_90.sort_values().index.tolist()

    # Collect per-dataset fold values for boxplot
    box_data = [
        frs_fold.loc[
            frs_fold["dataset"] == ds, "rule_mean_rules_for_90pct_contribution"
        ].values
        for ds in datasets_sorted
    ]
    box_labels = [_DATASET_LABELS.get(ds, ds) for ds in datasets_sorted]
    box_colors = [
        "#c7251b" if ds in _HIGH_DIM_DATASETS else _COLORS["frs"]
        for ds in datasets_sorted
    ]

    # Summary-level data for scatter panels
    scatter = frs_summary.merge(
        median_90.rename("median_rules_90").reset_index(),
        on="dataset",
        how="left",
    )
    n_features = scatter["n_features"].values
    jaccard = scatter["support_rule_jaccard"].values
    med_rules = scatter["median_rules_90"].values
    ds_labels_sc = [_DATASET_LABELS.get(ds, ds) for ds in scatter["dataset"]]
    is_high_dim = np.array([ds in _HIGH_DIM_DATASETS for ds in scatter["dataset"]])

    # 2-on-top / 1-centred-bottom layout with square panels
    fig = plt.figure(figsize=(7.5, 7.5))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.45)
    axes = [
        fig.add_subplot(gs[0, 0:2]),   # (a) top-left
        fig.add_subplot(gs[0, 2:4]),   # (b) top-right
        fig.add_subplot(gs[1, 1:3]),   # (c) bottom-centre
    ]

    # ------------------------------------------------------------------
    # Panel A: compactness boxplot
    # ------------------------------------------------------------------
    ax = axes[0]
    bp = ax.boxplot(
        box_data,
        orientation="horizontal",
        patch_artist=True,
        widths=0.55,
        flierprops={"marker": "x", "markersize": 4, "linestyle": "none",
                    "markeredgecolor": "#888888"},
        medianprops={"color": "white", "linewidth": 1.5},
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
    for whisker in bp["whiskers"]:
        whisker.set(linewidth=0.8, color="#555555")
    for cap in bp["caps"]:
        cap.set(linewidth=0.8, color="#555555")

    ax.set_yticks(range(1, len(box_labels) + 1))
    ax.set_yticklabels(box_labels, fontsize=7.0)
    ax.set_xlabel("Rules for 90\\% decision mass", fontsize=8)
    ax.set_title("(a) Explanation compactness", fontsize=8.5, pad=4)
    ax.axvline(10, color="#aaaaaa", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend for colour
    hi_patch = mpatches.Patch(color="#c7251b", alpha=0.8, label="High-dim. ($d>30$)")
    lo_patch = mpatches.Patch(color=_COLORS["frs"], alpha=0.8, label="Low/mod.-dim.")
    ax.legend(handles=[lo_patch, hi_patch], loc="lower right",
              framealpha=0.85, fontsize=6.5)

    # ------------------------------------------------------------------
    # Panel B: stability vs compactness
    # ------------------------------------------------------------------
    ax = axes[1]
    colors_sc = np.where(is_high_dim, "#c7251b", _COLORS["frs"])
    sc = ax.scatter(med_rules, jaccard, c=colors_sc,
                    s=38, edgecolors="white", linewidths=0.5,
                    zorder=3, alpha=0.9)

    for x, y, name, hd in zip(med_rules, jaccard, ds_labels_sc, is_high_dim):
        ax.annotate(
            name, (x, y),
            xytext=(3, 2), textcoords="offset points",
            fontsize=6.5, color="#444444",
        )

    ax.set_xlabel("Median rules for 90\\% decision mass", fontsize=8)
    ax.set_ylabel("Support-rule Jaccard (stability)", fontsize=8)
    ax.set_title("(b) Stability vs.~compactness", fontsize=8.5, pad=4)
    ax.axhline(0.44, color="#bbbbbb", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_box_aspect(1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Panel C: dimensionality vs explanation size
    # ------------------------------------------------------------------
    ax = axes[2]
    ax.scatter(n_features, med_rules, c=colors_sc,
               s=38, edgecolors="white", linewidths=0.5,
               zorder=3, alpha=0.9)

    for x, y, name, hd in zip(n_features, med_rules, ds_labels_sc, is_high_dim):
        ax.annotate(
            name, (x, y),
            xytext=(3, 2), textcoords="offset points",
            fontsize=6.5, color="#444444",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of features ($d$, log scale)", fontsize=8)
    ax.set_ylabel("Median rules for 90\\% decision mass", fontsize=8)
    ax.set_title("(c) Dimensionality vs.~explanation size", fontsize=8.5, pad=4)
    ax.set_box_aspect(1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.2)
    out = output_dir / "fig_interpretability.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 7 – Side-by-side semantic comparison: MembershipSVM vs FuzzyRuleSVM
# ---------------------------------------------------------------------------

class _MembershipSVM:
    """Minimal MembershipSVM used for figure generation only.

    Trains L1-LinearSVC on raw 3×d fuzzy membership features (no conjunctions).
    Provides an explain() method that returns per-feature contributions in the
    same format as FuzzyRuleSVM.explain(), enabling direct side-by-side display.
    """

    def __init__(self, *, C: float = 1.0, penalty: str = "l1",
                 class_weight: str = "balanced", random_state: int | None = None,
                 max_iter: int = 20000) -> None:
        from sklearn.svm import LinearSVC
        self._svc = LinearSVC(C=C, penalty=penalty, loss="squared_hinge",
                              dual=False, class_weight=class_weight,
                              random_state=random_state, max_iter=max_iter)
        self._partitions: list = []
        self._feature_names: list[str] = []
        self.classes_: np.ndarray | None = None
        from sklearn.utils.multiclass import unique_labels
        self._unique_labels = unique_labels

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list[str] | None = None) -> "_MembershipSVM":
        from fysvm.rule_svm import _FuzzyPartition
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        self.classes_ = self._unique_labels(y)
        quants = np.quantile(X, [0.05, 0.50, 0.95], axis=0)
        self._partitions = [
            _FuzzyPartition(float(quants[0, j]), float(quants[1, j]), float(quants[2, j]))
            for j in range(X.shape[1])
        ]
        if feature_names is None:
            feature_names = [f"x{j}" for j in range(X.shape[1])]
        self._feature_names = feature_names
        M = self._memberships(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._svc.fit(M, y)
        return self

    def _memberships(self, X: np.ndarray) -> np.ndarray:
        parts = [p.transform(X[:, j]) for j, p in enumerate(self._partitions)]
        return np.clip(np.hstack(parts), 0.0, 1.0)

    def _mem_feature_names(self) -> list[str]:
        terms = ["low", "medium", "high"]
        names = []
        for fname in self._feature_names:
            for t in terms:
                names.append(f"{fname} {t}")
        return names

    def explain_instance(self, x: np.ndarray, *, top_n: int = 5) -> list[dict]:
        """Return top contributing membership features for one instance."""
        M = self._memberships(x.reshape(1, -1))[0]
        coef = self._svc.coef_[0] if self._svc.coef_.ndim == 2 else self._svc.coef_
        contribs = M * coef
        feat_names = self._mem_feature_names()
        order = np.argsort(np.abs(contribs))[::-1][:top_n]
        return [
            {"feature": feat_names[i], "activation": float(M[i]),
             "weight": float(coef[i]), "contribution": float(contribs[i])}
            for i in order if abs(contribs[i]) > 1e-8
        ]


def _fit_both_pima_models(
    data_dir: Path,
) -> tuple["FuzzyRuleSVM", "_MembershipSVM", np.ndarray, np.ndarray, list[str]]:
    """Fit both models on the same Pima Diabetes 80/20 split."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    ds = load_dataset("pima_diabetes", data_dir)
    X, y = ds.X, np.asarray(ds.y)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Median imputation (shared)
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median").fit(X_tr)
    X_tr_imp = imp.transform(X_tr)
    X_te_imp  = imp.transform(X_te)

    # FuzzyRuleSVM (reuses same config as _fit_pima_model)
    frs = FuzzyRuleSVM(
        C=1.0, penalty="l1", max_rule_length=2, max_rules=64,
        min_rule_coverage=0.01, rule_length_penalty=0.35,
        feature_names=ds.feature_names, class_weight="balanced", random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        frs.fit(X_tr_imp, y_tr)

    # MembershipSVM with same hyperparameters
    mem = _MembershipSVM(C=1.0, class_weight="balanced", random_state=0)
    mem.fit(X_tr_imp, y_tr, feature_names=ds.feature_names)

    return frs, mem, X_te_imp, y_te, ds.feature_names


def figure_ablation_semantic(data_dir: Path, output_dir: Path) -> None:
    """Side-by-side explanation: MembershipSVM (individual terms) vs FuzzyRuleSVM (conjunctions).

    Fits both models on the same Pima Diabetes 80/20 split, selects the
    median-complexity test instance (measured by FuzzyRuleSVM rule count),
    and displays the top-5 contributing features/rules for each model.

    This figure makes the semantic contrast tangible: MembershipSVM returns
    atomic membership terms (``glucose high``, ``bmi medium``), while
    FuzzyRuleSVM returns verbalizable conjunctive IF-AND rules
    (``glucose high ∧ bmi high``), providing richer interaction context.
    """
    frs, mem, X_te, y_te, _ = _fit_both_pima_models(data_dir)

    # Pick the same median-complexity instance as in figure_waterfall
    idx = _pick_median_instance(frs, X_te, y_te)

    # FuzzyRuleSVM explanation
    frs_expl = frs.explain(X_te[[idx]], top_n=5, min_abs_contribution=0.01)[0]
    frs_rules = sorted(frs_expl["top_rules"],
                       key=lambda r: abs(r["contribution"]), reverse=True)[:5]

    # MembershipSVM explanation
    mem_items = mem.explain_instance(X_te[idx], top_n=5)

    def _shorten(label: str) -> str:
        return label.replace(" AND ", "\n∧ ")

    def _shorten_mem(label: str) -> str:
        return label

    # Build bar data for each panel
    frs_labels = [_shorten(r["rule"].split(" THEN ")[0].replace("IF ", ""))
                  for r in frs_rules]
    frs_values = [r["contribution"] for r in frs_rules]
    frs_colors = [_COLORS["positive"] if v >= 0 else _COLORS["negative"]
                  for v in frs_values]

    mem_labels = [_shorten_mem(it["feature"]) for it in mem_items]
    mem_values = [it["contribution"] for it in mem_items]
    mem_colors = [_COLORS["positive"] if v >= 0 else _COLORS["negative"]
                  for v in mem_values]

    fig, (ax_mem, ax_frs) = plt.subplots(1, 2, figsize=(9.5, 3.4),
                                          gridspec_kw={"width_ratios": [1, 1]})

    def _set_padded_xlim(ax: plt.Axes, values: list[float]) -> float:
        lo = min(min(values), 0.0)
        hi = max(max(values), 0.0)
        span = hi - lo if hi > lo else 1.0
        pad = max(0.16 * span, 0.035)
        ax.set_xlim(lo - pad, hi + pad)
        return max(0.025 * span, 0.006)

    # --- Left panel: MembershipSVM ---
    n_mem = len(mem_labels)
    yp = np.arange(n_mem)
    bars = ax_mem.barh(yp, mem_values, color=mem_colors, height=0.65,
                        edgecolor="white", linewidth=0.5)
    mem_label_offset = _set_padded_xlim(ax_mem, mem_values)
    for bar, val in zip(bars, mem_values):
        x_t = val + (mem_label_offset if val >= 0 else -mem_label_offset)
        ha = "left" if val >= 0 else "right"
        ax_mem.text(x_t, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=7.5)
    ax_mem.set_yticks(yp)
    ax_mem.set_yticklabels(mem_labels, fontsize=8)
    ax_mem.invert_yaxis()
    ax_mem.axvline(0, color="black", linewidth=0.7)
    ax_mem.set_xlabel("Contribution to margin", fontsize=8)
    ax_mem.set_title("(a) MembershipSVM\n(individual membership terms)",
                      fontsize=8.5, pad=5)
    ax_mem.spines["top"].set_visible(False)
    ax_mem.spines["right"].set_visible(False)

    # --- Right panel: FuzzyRuleSVM ---
    n_frs = len(frs_labels)
    yp2 = np.arange(n_frs)
    bars2 = ax_frs.barh(yp2, frs_values, color=frs_colors, height=0.65,
                         edgecolor="white", linewidth=0.5)
    frs_label_offset = _set_padded_xlim(ax_frs, frs_values)
    for bar, val in zip(bars2, frs_values):
        x_t = val + (frs_label_offset if val >= 0 else -frs_label_offset)
        ha = "left" if val >= 0 else "right"
        ax_frs.text(x_t, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=7.5)
    ax_frs.set_yticks(yp2)
    ax_frs.set_yticklabels(frs_labels, fontsize=8)
    ax_frs.invert_yaxis()
    ax_frs.axvline(0, color="black", linewidth=0.7)
    ax_frs.set_xlabel("Contribution to margin", fontsize=8)
    ax_frs.set_title("(b) FuzzyRuleSVM\n(conjunctive IF-AND rules)",
                      fontsize=8.5, pad=5)
    ax_frs.spines["top"].set_visible(False)
    ax_frs.spines["right"].set_visible(False)

    # Shared legend
    pos_patch = mpatches.Patch(color=_COLORS["positive"], label="Supports positive class")
    neg_patch = mpatches.Patch(color=_COLORS["negative"], label="Supports negative class")
    fig.legend(handles=[pos_patch, neg_patch], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06), framealpha=0.9, fontsize=8)

    fig.tight_layout()
    out = output_dir / "fig_ablation_semantic.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 8 – Per-dataset delta: FuzzyRuleSVM vs MembershipSVM
# ---------------------------------------------------------------------------

# Map dataset slugs to short names for the delta chart
_ABLATION_LABELS = {
    "breast_cancer_diagnostic":          "BC Diagnostic ($d$=30)",
    "breast_cancer_original":            "BC Original ($d$=9)",
    "mammographic_mass":                 "Mammo. Mass ($d$=5)",
    "breast_tissue":                     "Breast Tissue ($d$=9)",
    "heart_cleveland":                   "Heart Cleveland ($d$=13)",
    "statlog_heart":                     "Statlog Heart ($d$=13)",
    "spect_heart":                       "SPECT Heart ($d$=22)",
    "spectf_heart":                      "SPECTF Heart ($d$=44)",
    "pima_diabetes":                     "Pima Diabetes ($d$=8)",
    "diabetic_retinopathy_debrecen":     "Diab. Retinop. ($d$=19)",
    "parkinsons":                        "Parkinsons ($d$=22)",
    "parkinsons_disease_classification": "PD Classif. ($d$=754)",
    "ilpd":                              "ILPD (Liver, $d$=10)",
    "dermatology":                       "Dermatology ($d$=34)",
    "haberman_survival":                 "Haberman ($d$=3)",
    "vertebral_column_2c":               "Vertebral 2C ($d$=6)",
    "arrhythmia_binary":                 "Arrhythmia ($d$=279)",
    "iris":                              "Iris ($d$=4)",
    "wine":                              "Wine ($d$=13)",
    "digits":                            "Digits ($d$=64)",
}

# Datasets where rule generation is capped to 1-antecedent rules (d > 32)
_HIGHDIM_TRUNCATED = {
    "arrhythmia_binary",
    "parkinsons_disease_classification",
    "digits",
    "spectf_heart",
    "dermatology",
}


def figure_ablation_delta(runs_dir: Path, output_dir: Path) -> None:
    """Per-dataset balanced accuracy delta: FuzzyRuleSVM minus MembershipSVM.

    Datasets are sorted by delta (ascending). Positive bars (blue) indicate
    that conjunctive rule generation improved accuracy; negative bars (red)
    indicate that training directly on membership features was preferable.
    Datasets where FuzzyRuleSVM is constrained to single-antecedent rules
    (d > 32, shown with hatching) are semantically equivalent to MembershipSVM
    plus L1 selection, explaining most negative deltas.
    """
    metrics_path = runs_dir / "ablation-membership" / "metrics.csv"
    if not metrics_path.exists():
        print("  Skipping fig_ablation_delta.pdf (no ablation-membership run).")
        return

    import pandas as pd  # noqa: PLC0415
    df = pd.read_csv(metrics_path)
    frs = df[df["model_key"] == "fuzzy_rule_svm"].set_index("dataset")
    mem = df[df["model_key"] == "membership_svm"].set_index("dataset")
    common = sorted(set(frs.index) & set(mem.index))

    deltas = np.array([frs.loc[d, "balanced_accuracy_mean"] - mem.loc[d, "balanced_accuracy_mean"]
                       for d in common])
    labels = [_ABLATION_LABELS.get(d, d) for d in common]
    is_truncated = np.array([d in _HIGHDIM_TRUNCATED for d in common])

    # Sort ascending by delta
    order = np.argsort(deltas)
    deltas_s = deltas[order]
    labels_s  = [labels[i] for i in order]
    trunc_s   = is_truncated[order]

    colors = np.where(deltas_s >= 0, _COLORS["positive"], _COLORS["negative"])

    n = len(deltas_s)
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * n + 1.2))
    ypos = np.arange(n)

    bars = ax.barh(ypos, deltas_s, color=colors, height=0.70,
                   edgecolor="white", linewidth=0.4, alpha=0.90)

    lo = min(float(deltas_s.min()), 0.0)
    hi = max(float(deltas_s.max()), 0.0)
    span = hi - lo if hi > lo else 1.0
    pad = max(0.22 * span, 0.035)
    label_offset = max(0.025 * span, 0.004)
    ax.set_xlim(lo - pad, hi + pad)

    # Hatch truncated (high-dim) datasets
    for i, (bar, trunc) in enumerate(zip(bars, trunc_s)):
        if trunc:
            bar.set_hatch("///")
            bar.set_edgecolor("#555555")
            bar.set_linewidth(0.6)

    # Value labels
    for bar, val in zip(bars, deltas_s):
        x_t = val + (label_offset if val >= 0 else -label_offset)
        ha = "left" if val >= 0 else "right"
        ax.text(x_t, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=6.8)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels_s, fontsize=7.2)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Balanced accuracy delta (FuzzyRuleSVM $-$ MembershipSVM)", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend for hatch
    import matplotlib.patches as mp  # noqa: PLC0415
    norm_patch = mp.Patch(facecolor=_COLORS["positive"], edgecolor="white",
                          alpha=0.9, label="Multi-antecedent rules ($d \\leq 32$)")
    trunc_patch = mp.Patch(facecolor=_COLORS["negative"], hatch="///",
                           edgecolor="#555555", alpha=0.9,
                           label="Single-antecedent only ($d > 32$)")
    ax.legend(handles=[norm_patch, trunc_patch], loc="lower right",
              framealpha=0.88, fontsize=7)

    fig.tight_layout()
    out = output_dir / "fig_ablation_delta.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 9 – Accuracy–compactness Pareto analysis
# ---------------------------------------------------------------------------

#: Complexity metric specification for each model family.
#  key       – model_key value in metrics CSV
#  label     – display name in the figure
#  col       – column name in the summary metrics CSV (mean across folds)
#  metric_lbl – short x-axis label describing what the count represents
#  color     – marker colour
#  marker    – matplotlib marker symbol
_PARETO_MODELS: list[dict] = [
    {
        "key":        "fuzzy_rule_svm",
        "label":      "FuzzyRuleSVM",
        "col":        "rule_support_rule_count_mean",
        "metric_lbl": "active support rules",
        "color":      "#1b7837",   # dark green
        "marker":     "o",
    },
    {
        "key":        "membership_svm",
        "label":      "MembershipSVM",
        "col":        "membership_nonzero_coefs_mean",
        "metric_lbl": "nonzero membership coefs",
        "color":      "#4393c3",   # steel blue
        "marker":     "s",
    },
    {
        "key":        "furia",
        "label":      "FURIA",
        "col":        "furia_n_rules_mean",
        "metric_lbl": "fuzzy rules",
        "color":      "#762a83",   # purple
        "marker":     "^",
    },
    {
        "key":        "ebm",
        "label":      "EBM",
        "col":        "ebm_term_count_mean",
        "metric_lbl": "feature terms",
        "color":      "#d6604d",   # orange-red
        "marker":     "D",
    },
    {
        "key":        "rulefit",
        "label":      "RuleFit",
        "col":        "rulefit_nonzero_rules_mean",
        "metric_lbl": "nonzero binary rules",
        "color":      "#8e6abc",   # lavender
        "marker":     "v",
    },
    {
        "key":        "logistic_l1",
        "label":      "Logistic L1",
        "col":        "linear_nonzero_coefs_mean",
        "metric_lbl": "nonzero coefs",
        "color":      "#969696",   # grey
        "marker":     "P",
    },
]

#: Data sources for each model family: list of run directories to search.
#  The first directory that contains a matching model row is used.
_PARETO_DATA_SOURCES = [
    "modern-baselines-comparison",
    "ablation-membership",
    "fuzzy-baselines-comparison",
    "recommended-comparison",
]


def _load_pareto_data(runs_dir: Path) -> pd.DataFrame:
    """Collect per-dataset mean balanced accuracy and complexity for Pareto plot.

    Reads the ``metrics.csv`` from each run directory in *_PARETO_DATA_SOURCES*,
    merges them into a single table, and returns one row per (dataset, model_key)
    containing ``balanced_accuracy_mean`` and the model-specific complexity column.

    Missing data (no run yet) is silently skipped so the figure degrades
    gracefully when some experiments have not been run.
    """
    frames: list[pd.DataFrame] = []
    for src_dir in _PARETO_DATA_SOURCES:
        path = runs_dir / src_dir / "metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Keep only rows for model keys we care about, drop duplicates (same
    # model_key + dataset may appear in multiple runs; keep first).
    wanted_keys = {m["key"] for m in _PARETO_MODELS}
    combined = combined[combined["model_key"].isin(wanted_keys)]
    combined = combined.drop_duplicates(subset=["dataset", "model_key"], keep="first")
    return combined


def figure_pareto(runs_dir: Path, output_dir: Path) -> None:
    """Two-panel accuracy--compactness Pareto figure.

    Panel (a): Aggregated scatter -- one point per (model family, dataset),
    balanced accuracy on the y-axis versus the model-specific complexity
    proxy on the log-scaled x-axis.  Models that are fewer but equally or
    more accurate appear in the upper-left (desirable) quadrant.

    Panel (b): Aggregated Pareto view -- one mean point per model family
    with ±1 SD error bars across the 20 benchmark datasets.  A step-function
    Pareto frontier is overlaid to highlight the accuracy--compactness
    trade-off surface.

    Complexity metrics differ by model family (listed in the caption) and
    are not directly commensurable; the figure is a proxy analysis.  The
    FuzzyRuleSVM ``active support rules'' count is chosen to match the
    linguistic rule count tracked for FURIA.
    """
    df = _load_pareto_data(runs_dir)
    if df.empty:
        print("  Skipping fig_pareto.pdf (no run data available).")
        return

    # ------------------------------------------------------------------
    # Collect per-dataset complexity and accuracy for each model spec
    # ------------------------------------------------------------------
    model_data: list[dict] = []
    for spec in _PARETO_MODELS:
        key = spec["key"]
        col = spec["col"]
        rows = df[df["model_key"] == key]
        if rows.empty or col not in rows.columns:
            continue
        rows = rows.dropna(subset=[col, "balanced_accuracy_mean"])
        if rows.empty:
            continue
        model_data.append({
            **spec,
            "acc_vals":  rows["balanced_accuracy_mean"].values,
            "comp_vals": rows[col].values,
            "n":         len(rows),
        })

    if not model_data:
        print("  Skipping fig_pareto.pdf (no usable data).")
        return

    # ------------------------------------------------------------------
    # Build figure: two panels side by side
    # ------------------------------------------------------------------
    # Design at paper textwidth so font sizes render correctly at print size.
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 5.0),
                              gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ── Panel (a): per-dataset scatter ─────────────────────────────────
    ax = axes[0]
    legend_handles: list = []

    for md in model_data:
        sc = ax.scatter(
            md["comp_vals"], md["acc_vals"],
            color=md["color"],
            marker=md["marker"],
            s=32, alpha=0.72, edgecolors="white", linewidths=0.4,
            zorder=3,
            label=md["label"],
        )
        legend_handles.append(sc)

    ax.set_xscale("log")
    ax.set_xlabel("Model complexity\n(named explanation units — proxy, see caption)", fontsize=8)
    ax.set_ylabel("Balanced accuracy", fontsize=8)
    ax.set_title("(a) Per-dataset accuracy--compactness scatter", fontsize=8.5, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate FRS's upper-left cluster to make the argument visible
    frs_data = next((m for m in model_data if m["key"] == "fuzzy_rule_svm"), None)
    if frs_data is not None:
        ax.axvline(
            np.median(frs_data["comp_vals"]),
            color="#1b7837", linewidth=0.7, linestyle=":", alpha=0.5,
        )

    # ── Panel (b): aggregated Pareto (mean ± SD) ───────────────────────
    ax = axes[1]
    means_x: list[float] = []
    means_y: list[float] = []

    for md in model_data:
        mx = float(np.mean(md["comp_vals"]))
        my = float(np.mean(md["acc_vals"]))
        sx = float(np.std(md["comp_vals"], ddof=1)) if len(md["comp_vals"]) > 1 else 0.0
        sy = float(np.std(md["acc_vals"],  ddof=1)) if len(md["acc_vals"])  > 1 else 0.0

        ax.errorbar(
            mx, my,
            xerr=sx, yerr=sy,
            fmt=md["marker"],
            color=md["color"],
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=0.5,
            elinewidth=1.0,
            capsize=3,
            zorder=4,
        )
        # Label each point
        ax.annotate(
            md["label"],
            (mx, my),
            xytext=(6, 3),
            textcoords="offset points",
            fontsize=7.5,
            color=md["color"],
            fontweight="bold" if md["key"] == "fuzzy_rule_svm" else "normal",
        )
        means_x.append(mx)
        means_y.append(my)

    # Draw a Pareto-frontier step function through non-dominated points
    if len(means_x) >= 2:
        pts = sorted(zip(means_x, means_y), key=lambda p: p[0])
        frontier_x: list[float] = []
        frontier_y: list[float] = []
        best_y = -np.inf
        for px, py in pts:
            if py >= best_y:
                frontier_x.append(px)
                frontier_y.append(py)
                best_y = py
        if len(frontier_x) >= 2:
            # Extend step function to left/right
            ax.step(
                [frontier_x[0] * 0.5] + frontier_x + [frontier_x[-1] * 2],
                [frontier_y[0]] + frontier_y + [frontier_y[-1]],
                where="post",
                color="#bbbbbb", linewidth=1.2, linestyle="--",
                zorder=2, label="Pareto frontier (proxy)",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Mean model complexity\n(named explanation units ± 1 SD, proxy)", fontsize=8)
    ax.set_ylabel("Mean balanced accuracy ± 1 SD", fontsize=8)
    ax.set_title("(b) Aggregated accuracy--compactness Pareto", fontsize=8.5, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shared legend at bottom
    from matplotlib.lines import Line2D  # noqa: PLC0415
    legend_items = [
        Line2D([0], [0],
               marker=md["marker"],
               color=md["color"],
               label=f"{md['label']} ({md['metric_lbl']})",
               linestyle="None",
               markersize=7,
               markeredgecolor="white",
               markeredgewidth=0.5)
        for md in model_data
    ]
    fig.tight_layout(pad=1.2)
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=min(len(legend_items), 3),
        bbox_to_anchor=(0.5, -0.14),
        fontsize=7.5,
        framealpha=0.9,
        title="Model family (complexity metric)",
        title_fontsize=7.5,
    )
    out = output_dir / "fig_pareto.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")



# ---------------------------------------------------------------------------
# Figure 10 – Forest plots of per-dataset deltas
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap percentile 95% CI for the mean of *values*.

    Resamples *values* with replacement *n_boot* times and returns the
    (alpha/2, 1-alpha/2) percentile interval of the bootstrap means.
    With only 5 folds this interval is wide; it reflects the within-dataset
    fold-level variability, not a claim of asymptotic coverage.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(values)
    boot_means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def figure_forest_deltas(runs_dir: Path, output_dir: Path) -> None:
    """Two-panel forest plot of per-dataset balanced-accuracy deltas.

    Panel (a): FuzzyRuleSVM minus FURIA, one row per dataset.
    Panel (b): FuzzyRuleSVM minus RBF SVM, one row per dataset.

    Each row shows the per-dataset mean delta (marker) with 95% bootstrap
    confidence intervals computed from the per-fold delta values (one delta
    per fold pair).  Bootstrap CIs with 5 folds are wide; they quantify
    within-dataset fold variability, not asymptotic guarantees.

    Datasets are sorted by ascending mean delta within each panel.
    Rows where the CI crosses zero are shown in a lighter shade to indicate
    that the within-dataset advantage is not consistently positive.
    """
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Load per-fold data
    # ------------------------------------------------------------------
    fuzzy_fold_path = runs_dir / "fuzzy-baselines-comparison" / "fold_metrics.csv"
    recom_fold_path = runs_dir / "recommended-comparison" / "fold_metrics.csv"

    if not fuzzy_fold_path.exists() or not recom_fold_path.exists():
        print("  Skipping fig_forest_deltas.pdf (missing fold_metrics.csv).")
        return

    fb_fold = pd.read_csv(fuzzy_fold_path)
    rec_fold = pd.read_csv(recom_fold_path)

    def _per_dataset_deltas_same(
        fold_df: pd.DataFrame,
        key_a: str,
        key_b: str,
    ) -> dict[str, np.ndarray]:
        """Return dict of dataset → array of per-fold (key_a - key_b) deltas.

        Both models must appear in *fold_df* and share the same fold indices.
        """
        a = fold_df[fold_df["model_key"] == key_a].set_index(["dataset", "fold"])
        b = fold_df[fold_df["model_key"] == key_b].set_index(["dataset", "fold"])
        common_idx = a.index.intersection(b.index)
        result: dict[str, list[float]] = {}
        for dataset, fold in common_idx:
            delta = float(a.loc[(dataset, fold), "balanced_accuracy"]) - \
                    float(b.loc[(dataset, fold), "balanced_accuracy"])
            result.setdefault(dataset, []).append(delta)
        return {ds: np.array(vals) for ds, vals in result.items()}

    def _per_dataset_deltas_cross(
        fold_a: pd.DataFrame,
        fold_b: pd.DataFrame,
        key_a: str,
        key_b: str,
    ) -> dict[str, np.ndarray]:
        """Return per-fold (key_a - key_b) deltas when the two models live in
        different DataFrames but share matched fold numbering on the same datasets.

        FRS values come from *fold_a*, the baseline from *fold_b*.
        """
        a = fold_a[fold_a["model_key"] == key_a].set_index(["dataset", "fold"])
        b = fold_b[fold_b["model_key"] == key_b].set_index(["dataset", "fold"])
        common_idx = a.index.intersection(b.index)
        result: dict[str, list[float]] = {}
        for dataset, fold in common_idx:
            delta = float(a.loc[(dataset, fold), "balanced_accuracy"]) - \
                    float(b.loc[(dataset, fold), "balanced_accuracy"])
            result.setdefault(dataset, []).append(delta)
        return {ds: np.array(vals) for ds, vals in result.items()}

    # FRS vs FURIA: use recommended-comparison FRS fold data (all 20 datasets)
    # paired with fuzzy-baselines FURIA fold data (19 datasets).
    frs_furia_deltas = _per_dataset_deltas_cross(
        rec_fold, fb_fold, "fuzzy_rule_svm", "furia"
    )
    # FRS vs RBF SVM: both are in recommended-comparison.
    frs_rbf_deltas = _per_dataset_deltas_same(rec_fold, "fuzzy_rule_svm", "rbf_svm")

    def _build_panel_data(
        delta_dict: dict[str, np.ndarray],
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        """Compute mean, CI lo, CI hi per dataset, sorted by ascending mean."""
        datasets_list = sorted(delta_dict, key=lambda d: np.mean(delta_dict[d]))
        means = np.array([float(np.mean(delta_dict[d])) for d in datasets_list])
        ci_lo = np.zeros(len(datasets_list))
        ci_hi = np.zeros(len(datasets_list))
        for i, ds in enumerate(datasets_list):
            lo, hi = _bootstrap_ci(delta_dict[ds], n_boot=2000, rng=rng)
            ci_lo[i] = lo
            ci_hi[i] = hi
        return datasets_list, means, ci_lo, ci_hi

    furia_ds, furia_means, furia_lo, furia_hi = _build_panel_data(frs_furia_deltas)
    rbf_ds,   rbf_means,   rbf_lo,   rbf_hi   = _build_panel_data(frs_rbf_deltas)

    short_names = DATASET_SHORT_NAMES

    def _draw_panel(
        ax: plt.Axes,
        datasets: list[str],
        means: np.ndarray,
        ci_lo: np.ndarray,
        ci_hi: np.ndarray,
        *,
        baseline_label: str,
        n_folds: int = 5,
    ) -> None:
        n = len(datasets)
        ypos = np.arange(n)
        # Colour: positive CI interval (lo > 0) → solid; uncertain (CI spans 0) → lighter
        for i in range(n):
            spans_zero = (ci_lo[i] <= 0) and (ci_hi[i] >= 0)
            color = "#1b7837" if means[i] > 0 else _COLORS["negative"]
            alpha = 0.50 if spans_zero else 0.90
            # Horizontal error bar
            ax.plot(
                [ci_lo[i], ci_hi[i]], [ypos[i], ypos[i]],
                color=color, linewidth=1.0, alpha=alpha, solid_capstyle="round",
                zorder=2,
            )
            # Mean marker
            ax.plot(
                means[i], ypos[i],
                marker="D" if spans_zero else "o",
                markersize=5,
                color=color,
                alpha=alpha,
                zorder=3,
                markeredgecolor="white",
                markeredgewidth=0.4,
            )
        # Reference line
        ax.axvline(0, color="#888888", linewidth=0.9, linestyle="--", zorder=1)
        # Dataset labels
        ax.set_yticks(ypos)
        ax.set_yticklabels(
            [short_names.get(ds, ds) for ds in datasets],
            fontsize=7.0,
        )
        ax.set_xlabel(
            f"Balanced accuracy delta\n(FuzzyRuleSVM $-$ {baseline_label})",
            fontsize=8,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Legend
        import matplotlib.patches as mp  # noqa: PLC0415
        ci_line = plt.Line2D([0], [0], color="#1b7837", linewidth=1.0,
                             label=f"95\\% bootstrap CI ({n_folds} folds per dataset)")
        z_marker = plt.Line2D([0], [0], marker="D", markersize=5, linestyle="none",
                              color="#888888", label="CI spans zero (uncertain)")
        ax.legend(handles=[ci_line, z_marker], loc="lower right",
                  framealpha=0.85, fontsize=6.5)

    n_furia = len(furia_ds)
    n_rbf   = len(rbf_ds)
    # Design at paper textwidth (~6.5 in) so font sizes render at their
    # actual values rather than being scaled down to ~3.7 pt.
    # row_h is kept moderate so the figure fits within a single page.
    row_h = 0.29
    fig_h = max(5.0, (max(n_furia, n_rbf)) * row_h + 1.8)

    fig, (ax_furia, ax_rbf) = plt.subplots(
        1, 2,
        figsize=(6.5, fig_h),
        gridspec_kw={"wspace": 0.40},
    )

    _draw_panel(
        ax_furia,
        furia_ds, furia_means, furia_lo, furia_hi,
        baseline_label="FURIA",
        n_folds=5,
    )
    ax_furia.set_title(
        "(a) FuzzyRuleSVM vs.~FURIA\n"
        f"({n_furia} datasets, 5-fold nested CV)",
        fontsize=8.5, pad=4,
    )

    _draw_panel(
        ax_rbf,
        rbf_ds, rbf_means, rbf_lo, rbf_hi,
        baseline_label="RBF SVM",
        n_folds=5,
    )
    ax_rbf.set_title(
        "(b) FuzzyRuleSVM vs.~RBF SVM\n"
        f"({n_rbf} datasets, 5-fold nested CV)",
        fontsize=8.5, pad=4,
    )

    fig.tight_layout(pad=1.0)
    out = output_dir / "fig_forest_deltas.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure – High-dimensional failure mode: rule-budget sensitivity
# ---------------------------------------------------------------------------

def figure_highdim_budget(runs_dir: Path, output_dir: Path) -> None:
    """Two-panel figure characterising the high-dimensional failure mode.

    Panel (a): Line plot of balanced accuracy vs max_rules budget for four
    representative datasets spanning low to very-high dimensionality.  Error
    bands show ±1 fold SE (5 folds).

    Panel (b): Horizontal stacked bar chart showing, for each dataset, the
    total candidate rules generated (after coverage filter) and how many
    are retained under the current default budget (256 rules).  Truncation
    is shown as the difference.
    """

    metrics_path = runs_dir / "highdim-analysis" / "metrics.csv"
    if not metrics_path.exists():
        print(f"  Skipping fig_highdim_budget.pdf (no highdim-analysis run: {metrics_path})")
        return

    df = pd.read_csv(metrics_path)

    # Dataset display names and dimensionality labels
    _DATASET_LABELS: dict[str, str] = {
        "pima_diabetes":                     "Pima ($d=8$)",
        "spectf_heart":                      "SPECTF Heart ($d=44$)",
        "arrhythmia_binary":                 "Arrhythmia ($d=279$)",
        "parkinsons_disease_classification": "Parkinson's ($d=754$)",
    }
    _DATASET_COLORS: dict[str, str] = {
        "pima_diabetes":                     "#4dac26",   # green (low-dim)
        "spectf_heart":                      "#f4a582",   # light orange (moderate)
        "arrhythmia_binary":                 "#d6604d",   # red (high-dim)
        "parkinsons_disease_classification": "#8e0152",   # dark purple (very high-dim)
    }

    budgets = sorted(df["max_rules"].unique())

    # Only use length-1 rows (cross-dataset comparable and used on all datasets)
    df1 = df[df["max_rule_length"] == 1].copy()

    # ---- Panel (a): accuracy vs budget ----
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), gridspec_kw={"wspace": 0.38})
    ax_a, ax_b = axes

    for slug, label in _DATASET_LABELS.items():
        sub = df1[df1["dataset"] == slug].sort_values("max_rules")
        if sub.empty:
            continue
        acc_mean = sub["balanced_accuracy_mean"].values
        acc_std  = sub["balanced_accuracy_std"].values
        n_folds  = sub["n_folds"].values
        se = acc_std / np.sqrt(n_folds)

        xs = sub["max_rules"].values
        color = _DATASET_COLORS[slug]
        ax_a.plot(xs, acc_mean, "o-", color=color, label=label, linewidth=1.6,
                  markersize=5, zorder=3)
        ax_a.fill_between(xs, acc_mean - se, acc_mean + se,
                          alpha=0.18, color=color, zorder=2)

    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(budgets)
    ax_a.set_xticklabels([str(b) for b in budgets], fontsize=7.5)
    ax_a.set_xlabel(r"Rule budget ($K_{\max}$)", fontsize=9)
    ax_a.set_ylabel("Balanced accuracy", fontsize=9)
    ax_a.set_title("(a) Budget sensitivity by dimensionality regime", fontsize=9)
    ax_a.legend(fontsize=7.5, loc="lower right", handlelength=1.8)
    ax_a.yaxis.set_minor_locator(plt.matplotlib.ticker.MultipleLocator(0.01))
    ax_a.grid(True, which="major", linewidth=0.4, color="#cccccc")
    ax_a.grid(True, which="minor", linewidth=0.2, color="#eeeeee")

    # Mark the current default budget (256) with a dashed vertical line
    ax_a.axvline(x=256, color="#888888", linestyle="--", linewidth=0.9,
                 label=r"Default ($K_{\max}=256$)", zorder=1)

    # ---- Panel (b): truncation severity bar chart ----
    # Show, for each dataset, how many candidates are generated vs retained at
    # the default budget of 256.
    default_budget = 256
    row_order = list(_DATASET_LABELS.keys())
    row_labels = [_DATASET_LABELS[s] for s in row_order]
    n_rows = len(row_order)

    retained_vals = []
    candidate_vals = []
    for slug in row_order:
        sub = df1[(df1["dataset"] == slug) & (df1["max_rules"] == default_budget)]
        if sub.empty:
            # fall back to highest available budget
            sub = df1[df1["dataset"] == slug].sort_values("max_rules").tail(1)
        if sub.empty:
            retained_vals.append(0.0)
            candidate_vals.append(0.0)
        else:
            row = sub.iloc[0]
            cands = float(row.get("n_candidate_rules_mean", 0.0) or 0.0)
            # retained = min(default_budget, candidates)
            retained = min(float(default_budget), cands)
            retained_vals.append(retained)
            candidate_vals.append(cands)

    y_pos = np.arange(n_rows)
    bar_h = 0.52

    # Truncated (discarded) = candidates - retained
    discarded_vals = [max(0.0, c - r) for c, r in zip(candidate_vals, retained_vals)]
    # Retention fraction for annotation
    retention_fracs = [
        (r / c) if c > 0 else 1.0
        for r, c in zip(retained_vals, candidate_vals)
    ]

    # Draw stacked horizontal bars: retained (blue) + discarded (light gray)
    bars_ret = ax_b.barh(y_pos, retained_vals, height=bar_h,
                         color=[_DATASET_COLORS[s] for s in row_order],
                         alpha=0.85, label="Retained ($K=256$)", zorder=3)
    bars_dis = ax_b.barh(y_pos, discarded_vals, height=bar_h,
                         left=retained_vals,
                         color="#cccccc", alpha=0.7, label="Discarded", zorder=2)

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(row_labels, fontsize=7.5)
    ax_b.set_xlabel("Candidate rules (after coverage filter)", fontsize=9)
    ax_b.set_title("(b) Rule truncation at default budget ($K=256$)", fontsize=9)
    ax_b.invert_yaxis()

    # Annotate retention fraction
    for i, (ret, cand, frac) in enumerate(zip(retained_vals, candidate_vals, retention_fracs)):
        if cand > 0:
            pct_str = f"{frac * 100:.0f}%" if frac < 0.999 else "100%"
            ax_b.text(
                cand + max(candidate_vals) * 0.01,
                y_pos[i],
                pct_str,
                va="center", ha="left", fontsize=7.5, color="#555555",
            )

    ax_b.legend(fontsize=7.5, loc="lower right")
    ax_b.grid(True, axis="x", linewidth=0.4, color="#cccccc")

    fig.tight_layout(pad=1.2)
    out = output_dir / "fig_highdim_budget.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def figure_budget_sensitivity(runs_dir: Path, output_dir: Path) -> None:
    """Generate GA budget sensitivity figure for FARC-HD/IVTURS if artifacts exist."""

    metrics_path = runs_dir / "fuzzy-highbudget" / "sensitivity_rows.csv"
    if not metrics_path.exists():
        print(f"  Skipping fig_budget_sensitivity.pdf ({metrics_path} not found).")
        return
    df = pd.read_csv(metrics_path)
    if df.empty or not {"model", "dataset", "budget", "balanced_accuracy"} <= set(df.columns):
        print(f"  Skipping fig_budget_sensitivity.pdf ({metrics_path} lacks expected columns).")
        return
    summary = df.groupby(["model", "budget"], as_index=False)["balanced_accuracy"].mean()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for model, group in summary.groupby("model"):
        group = group.sort_values("budget")
        ax.plot(group["budget"], group["balanced_accuracy"], marker="o", label=model)
    ax.set_xscale("log")
    ax.set_xlabel("GA fitness evaluations")
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_title("Budget-limited FARC-HD/IVTURS sensitivity")
    ax.grid(True, linewidth=0.4, color="#dddddd")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = output_dir / "fig_budget_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 12 – Broader ablation suite: accuracy bar chart + heatmap
# ---------------------------------------------------------------------------

#: Display order and labels for the nine ablation variants
_ABLATION_VARIANT_ORDER = [
    ("linear_svc_raw",     "LinearSVC-Raw"),
    ("membership_svm",     "MembershipSVM"),
    ("frs_1term_only",     "FRS-1TermOnly"),
    ("frs_wide_quantiles", "FRS-WideQuantiles"),
    ("frs_l2_forced",      "FRS-L2Forced"),
    ("frs_no_penalty",     "FRS-NoPenalty"),
    ("frs_softmin_tnorm",  "FRS-SoftminTnorm"),
    ("frs_product_tnorm",  "FRS-ProductTnorm"),
    ("frs_default",        "FRS-Default"),
]

_ABLATION_COLORS = {
    "frs_default":        "#1b7837",   # dark green (reference)
    "frs_no_penalty":     "#5aae61",
    "frs_l2_forced":      "#a6dba0",
    "frs_product_tnorm":  "#7fbfff",
    "frs_softmin_tnorm":  "#4393c3",
    "frs_wide_quantiles": "#f4a582",
    "frs_1term_only":     "#d6604d",
    "membership_svm":     "#b2abd2",
    "linear_svc_raw":     "#8073ac",
}


def figure_ablation_suite(runs_dir: Path, output_dir: Path) -> None:
    """Broader ablation suite figure.

    Panel (a): Horizontal bar chart of mean balanced accuracy per variant
               with ±1 SD error bars (across datasets).
    Panel (b): Per-dataset performance delta heatmap (variant − FRS-Default).
    """
    ablation_dir = runs_dir / "ablation-suite"
    metrics_path = ablation_dir / "metrics.csv"
    if not metrics_path.exists():
        print(f"    Skipping fig_ablation_suite.pdf ({metrics_path} not found).")
        return

    df = pd.read_csv(metrics_path)
    df["balanced_accuracy_mean"] = pd.to_numeric(df["balanced_accuracy_mean"], errors="coerce")

    # ---- Panel (a): summary bar chart ----
    variant_keys  = [k for k, _ in _ABLATION_VARIANT_ORDER]
    variant_labels = [lbl for _, lbl in _ABLATION_VARIANT_ORDER]

    means = []
    stds  = []
    for key in variant_keys:
        sub = df[df["model_key"] == key]["balanced_accuracy_mean"]
        means.append(float(sub.mean()))
        stds.append(float(sub.std(ddof=1)))

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(14, 4.8),
        gridspec_kw={"width_ratios": [1, 2]},
    )

    y_pos = np.arange(len(variant_keys))
    bar_colors = [_ABLATION_COLORS[k] for k in variant_keys]
    bars = ax_a.barh(y_pos, means, xerr=stds, height=0.6,
                     color=bar_colors, alpha=0.9,
                     error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "#555555"},
                     zorder=3)

    # Annotate values
    for i, (m, s) in enumerate(zip(means, stds)):
        ax_a.text(m + s + 0.001, y_pos[i], f"{m:.3f}", va="center", ha="left",
                  fontsize=7.5, color="#333333")

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(variant_labels, fontsize=8)
    ax_a.set_xlabel("Mean balanced accuracy (20 datasets)", fontsize=9)
    ax_a.set_title("(a) Mean balanced accuracy ± 1 SD", fontsize=9)
    ax_a.axvline(means[-1], color=_ABLATION_COLORS["frs_default"],
                 linewidth=1.2, linestyle="--", alpha=0.7,
                 label=f"FRS-Default ({means[-1]:.3f})")
    ax_a.legend(fontsize=7.5, loc="lower right")
    ax_a.grid(True, axis="x", linewidth=0.4, color="#dddddd", zorder=0)
    x_lo = max(0.0, min(means) - max(stds) - 0.02)
    ax_a.set_xlim(x_lo, min(1.0, max(means) + max(stds) + 0.06))

    # ---- Panel (b): delta heatmap ----
    # Pivot to datasets × variants
    ref_key = "frs_default"
    # Gather datasets in display order (sorted by FRS-Default value)
    ref_df = df[df["model_key"] == ref_key][["dataset", "balanced_accuracy_mean"]].copy()
    ref_df = ref_df.sort_values("balanced_accuracy_mean")
    datasets = ref_df["dataset"].tolist()

    # Build delta matrix: variant - default per dataset
    delta_mat = np.zeros((len(datasets), len(variant_keys)))
    for j, key in enumerate(variant_keys):
        sub = df[df["model_key"] == key].set_index("dataset")["balanced_accuracy_mean"]
        ref  = df[df["model_key"] == ref_key].set_index("dataset")["balanced_accuracy_mean"]
        for i, ds in enumerate(datasets):
            if ds in sub.index and ds in ref.index:
                delta_mat[i, j] = float(sub[ds]) - float(ref[ds])

    vmax = max(abs(delta_mat).max(), 0.03)
    im = ax_b.imshow(
        delta_mat.T,
        aspect="auto",
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax_b.set_xticks(np.arange(len(datasets)))
    ax_b.set_xticklabels(
        [_fmt_ds(ds) for ds in datasets],
        rotation=45, ha="right", fontsize=7,
    )
    ax_b.set_yticks(np.arange(len(variant_keys)))
    ax_b.set_yticklabels(variant_labels, fontsize=8)
    ax_b.set_title(
        "(b) Per-dataset balanced-accuracy delta vs FRS-Default\n"
        "(blue = variant better; red = variant worse)",
        fontsize=9,
    )

    # Annotate cells with values ≥ 0.01 in magnitude
    for i in range(len(datasets)):
        for j in range(len(variant_keys)):
            v = delta_mat[i, j]
            if abs(v) >= 0.01:
                ax_b.text(i, j, f"{v:+.2f}", ha="center", va="center",
                          fontsize=5.5, color="black" if abs(v) < 0.06 else "white")

    cbar = fig.colorbar(im, ax=ax_b, fraction=0.025, pad=0.02)
    cbar.set_label("Δ balanced accuracy", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout(pad=1.2)
    out = output_dir / "fig_ablation_suite.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _fmt_ds(slug: str) -> str:
    """Shorten a dataset slug for axis labels."""
    _short = {
        "breast_cancer_diagnostic": "BCD",
        "breast_cancer_original":   "BCO",
        "mammographic_mass":        "Mammo",
        "breast_tissue":            "BreastT",
        "heart_cleveland":          "HeartC",
        "statlog_heart":            "StatH",
        "spect_heart":              "SPECT",
        "spectf_heart":             "SPECTF",
        "pima_diabetes":            "Pima",
        "diabetic_retinopathy_debrecen": "DRD",
        "parkinsons":               "ParkS",
        "parkinsons_disease_classification": "ParkC",
        "ilpd":                     "ILPD",
        "dermatology":              "Derm",
        "haberman_survival":        "Hab",
        "vertebral_column_2c":      "VC2C",
        "arrhythmia_binary":        "Arrh",
        "iris":                     "Iris",
        "wine":                     "Wine",
        "digits":                   "Digits",
    }
    return _short.get(slug, slug[:6])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",   default="datasets/prepared")
    parser.add_argument("--runs-dir",   default="runs")
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--verify", action="store_true", help="Fail if paper references missing figures.")
    args = parser.parse_args(argv)

    data_dir   = Path(args.data_dir)
    runs_dir   = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fuzzy_baseline_metrics = runs_dir / "fuzzy-baselines-comparison" / "metrics.csv"

    print("Generating figures...")
    print("  [0/8] Architecture pipeline diagram (Fig. 0)...")
    figure_architecture(output_dir)

    print("  [1/8] Rule contribution waterfall (Fig. 1)...")
    figure_waterfall(data_dir, output_dir)

    print("  [2/8] Fuzzy membership functions (Fig. 2)...")
    figure_membership(data_dir, output_dir)

    if fuzzy_baseline_metrics.exists():
        print("  [3/8] Paired scatter vs FURIA (Fig. 3)...")
        figure_scatter(runs_dir, output_dir)
        print("  [4/8] Critical Difference diagram (Fig. 5)...")
        figure_cd(runs_dir, output_dir)
    else:
        print("  [3/8] Skipping fig_scatter.pdf (no fuzzy-baselines-comparison run).")
        print("  [4/8] Skipping fig_cd.pdf (no fuzzy-baselines-comparison run).")

    print("  [5/8] Interpretability analysis (Fig. 6)...")
    figure_interpretability(runs_dir, output_dir)

    print("  [6/8] Ablation semantic comparison (Fig. 7)...")
    figure_ablation_semantic(data_dir, output_dir)

    print("  [7/8] Ablation per-dataset delta chart (Fig. 8)...")
    figure_ablation_delta(runs_dir, output_dir)

    print("  [8/9] Accuracy--compactness Pareto analysis (Fig. 9)...")
    figure_pareto(runs_dir, output_dir)

    print("  [9/10] Forest plots of per-dataset deltas (Fig. 10)...")
    figure_forest_deltas(runs_dir, output_dir)

    highdim_metrics = runs_dir / "highdim-analysis" / "metrics.csv"
    if highdim_metrics.exists():
        print("  [10/11] High-dimensional failure mode: budget sensitivity (Fig. 11)...")
        figure_highdim_budget(runs_dir, output_dir)
    else:
        print("  [10/11] Skipping fig_highdim_budget.pdf (no highdim-analysis run).")

    print("  [11/12] FARC-HD/IVTURS GA budget sensitivity figure...")
    figure_budget_sensitivity(runs_dir, output_dir)

    ablation_suite_metrics = runs_dir / "ablation-suite" / "metrics.csv"
    if ablation_suite_metrics.exists():
        print("  [12/12] Broader ablation suite: bar chart + heatmap (Fig. 12)...")
        figure_ablation_suite(runs_dir, output_dir)
    else:
        print("  [12/12] Skipping fig_ablation_suite.pdf (no ablation-suite run).")

    if args.verify:
        _verify_included_figures(output_dir)

    print(f"\nAll figures written to {output_dir}/")


def _verify_included_figures(output_dir: Path) -> None:
    tex = Path("paper/paper.tex")
    content = tex.read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", content)))
    missing = []
    for name in referenced:
        path = output_dir / name
        if path.suffix:
            candidates = [path]
        else:
            candidates = [path.with_suffix(ext) for ext in (".pdf", ".png", ".jpg", ".jpeg")]
        if not any(candidate.exists() for candidate in candidates):
            missing.append(name)
    if missing:
        raise FileNotFoundError("Missing figures referenced by paper.tex: " + ", ".join(missing))
    print(f"  Verified {len(referenced)} figure references in {tex}.")


if __name__ == "__main__":
    main()
