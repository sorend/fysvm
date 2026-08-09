"""Independent scalar reference evaluator for MCLTA.

This module implements a brute-force grid search over the transition feasible
set to verify that the MILP solver's certified bounds contain the true extrema.

IMPORTANT: This module must NOT import any production MILP encoding helpers
from transition_envelopes.py. It uses only the model's public attributes and
independent scalar arithmetic.
"""

from __future__ import annotations

import numpy as np
from typing import Any


# ------------------------------------------------------------------ #
# Scalar membership (no production imports)                          #
# ------------------------------------------------------------------ #


def mu_scalar(term: str, v: float, q_low: float, q_mid: float, q_high: float) -> float:
    """Scalar membership evaluation for low/medium/high terms."""
    if term == "low":
        if q_mid <= q_low:
            return 1.0 if v <= q_low else 0.0
        if v <= q_low:
            return 1.0
        if v >= q_mid:
            return 0.0
        return (q_mid - v) / (q_mid - q_low)
    if term == "high":
        if q_high <= q_mid:
            return 1.0 if v >= q_high else 0.0
        if v <= q_mid:
            return 0.0
        if v >= q_high:
            return 1.0
        return (v - q_mid) / (q_high - q_mid)
    # medium
    if q_mid <= q_low:
        up = 1.0 if v >= q_mid else 0.0
    else:
        up = max(0.0, min(1.0, (v - q_low) / (q_mid - q_low)))
    if q_high <= q_mid:
        down = 1.0 if v <= q_mid else 0.0
    else:
        down = max(0.0, min(1.0, (q_high - v) / (q_high - q_mid)))
    return min(up, down)


def phi_scalar(rule_conditions: list[tuple[int, str]], x_vec: list[float], partitions: Any) -> float:
    """Scalar min rule activation.

    Parameters
    ----------
    rule_conditions:
        List of (feature_index, term) for each condition.
    x_vec:
        Screened-space feature values.
    partitions:
        Model's partitions_ list (has .low, .medium, .high attributes).
    """
    act = 1.0
    for feat, term in rule_conditions:
        p = partitions[feat]
        act = min(act, mu_scalar(term, x_vec[feat], p.low, p.medium, p.high))
    return act


def f_scalar(model: Any, x_vec: list[float]) -> float:
    """Scalar decision function evaluation.

    Parameters
    ----------
    model:
        FuzzyRuleSVM (uses .coef_, .intercept_, .rules_, .partitions_).
    x_vec:
        Screened-space feature values (length = n_screened_features_).
    """
    s = float(model.intercept_)
    for k, rule in enumerate(model.rules_):
        beta = float(model.coef_[k])
        if beta == 0.0:
            continue
        conds = [(c.feature, c.term) for c in rule.conditions]
        s += beta * phi_scalar(conds, x_vec, model.partitions_)
    return s


def f_t_scalar(model: Any, x_vec: list[float], target_sign: int) -> float:
    """Target-oriented score (f_t = target_sign * f)."""
    return target_sign * f_scalar(model, x_vec)


def transition_objective_scalar(
    model: Any,
    z_vec: list[float],
    u_val: float,
    v_val: float,
    j: int,
    target_sign: int,
) -> float:
    """Transition score change f_t(z, v) - f_t(z, u).

    Parameters
    ----------
    z_vec:
        Context feature values (length = n_screened_features_; position j is ignored).
    u_val:
        Source feature value for feature j.
    v_val:
        Destination feature value for feature j.
    j:
        Transition feature index.
    target_sign:
        +1 for positive class, -1 for negative class.
    """
    src = list(z_vec)
    src[j] = u_val
    dst = list(z_vec)
    dst[j] = v_val
    return f_t_scalar(model, dst, target_sign) - f_t_scalar(model, src, target_sign)


# ------------------------------------------------------------------ #
# Dense grid search                                                   #
# ------------------------------------------------------------------ #


def brute_force_envelope(
    model: Any,
    j: int,
    source_term: str,
    destination_term: str,
    source_alpha: float,
    destination_alpha: float,
    target_sign: int,
    lower: list[float],
    upper: list[float],
    context_lits: list[tuple[int, str, float]] | None = None,
    enforce_order: bool = True,
    min_displacement: float = 0.0,
    n_grid: int = 50,
) -> tuple[float, float]:
    """Brute-force dense grid search for transition envelope extrema.

    Returns (min_delta, max_delta) over the feasible set. This is a
    diagnostic tool, NOT a certificate — the true optimum may lie between
    grid points.

    Parameters
    ----------
    model:
        FuzzyRuleSVM.
    j:
        Transition feature index.
    source_term, destination_term, source_alpha, destination_alpha:
        Transition query parameters.
    target_sign:
        +1 or -1.
    lower, upper:
        Feature domain bounds (length = n_screened).
    context_lits:
        List of (feature_index, term, min_membership) context constraints.
    enforce_order:
        If True, enforce v > u (for low->high) or v < u (for high->low).
    min_displacement:
        Minimum |v - u| required.
    n_grid:
        Grid points per dimension for context features.

    Returns
    -------
    (min_delta, max_delta) over feasible grid points.
    """
    n_screened = len(model.partitions_)
    lower_arr = np.array(lower, dtype=np.float64)
    upper_arr = np.array(upper, dtype=np.float64)

    _TERM_ORDER = {"low": 0, "medium": 1, "high": 2}
    src_ord = _TERM_ORDER.get(source_term, 0)
    dst_ord = _TERM_ORDER.get(destination_term, 0)
    if enforce_order and src_ord != dst_ord:
        s_ab = 1 if dst_ord > src_ord else -1
    else:
        s_ab = 0

    # Grid for the transition feature
    j_grid = np.linspace(lower_arr[j], upper_arr[j], n_grid)

    # Grid for context features
    ctx_grids = []
    for k in range(n_screened):
        if k == j:
            ctx_grids.append(None)
        else:
            ctx_grids.append(np.linspace(lower_arr[k], upper_arr[k], max(n_grid // 4, 10)))

    min_delta = np.inf
    max_delta = -np.inf

    partitions = model.partitions_

    # Build context feature index sets
    ctx_feat_indices = [k for k in range(n_screened) if k != j]
    # Limit grid dimensions to avoid explosion
    max_ctx_dims = min(len(ctx_feat_indices), 3)
    used_ctx_feats = ctx_feat_indices[:max_ctx_dims]
    fixed_ctx = [float((lower_arr[k] + upper_arr[k]) / 2) for k in range(n_screened)]

    # For small problems, do a full grid; for larger ones, sample
    if len(used_ctx_feats) <= 2:
        ctx_grids_used = [ctx_grids[k] for k in used_ctx_feats]
        ctx_meshes = np.meshgrid(*ctx_grids_used, indexing="ij")
        ctx_flat = [m.ravel() for m in ctx_meshes]
        n_ctx = len(ctx_flat[0]) if ctx_flat else 1
    else:
        # Random sampling for higher dimensions
        rng = np.random.default_rng(0)
        n_ctx = 200
        ctx_flat = [
            rng.uniform(float(lower_arr[k]), float(upper_arr[k]), n_ctx)
            for k in used_ctx_feats
        ]

    for ci in range(n_ctx if ctx_flat else 1):
        # Build context vector
        z = list(fixed_ctx)
        for fi_idx, k in enumerate(used_ctx_feats):
            z[k] = float(ctx_flat[fi_idx][ci]) if ctx_flat else float(lower_arr[k])

        # Check context constraints
        ctx_ok = True
        if context_lits:
            for feat_c, term_c, gamma_c in context_lits:
                p = partitions[feat_c]
                # Check context: at z for context features
                if feat_c != j:
                    mu = mu_scalar(term_c, z[feat_c], p.low, p.medium, p.high)
                    if mu < gamma_c - 1e-10:
                        ctx_ok = False
                        break
        if not ctx_ok:
            continue

        # Source grid
        for u in j_grid:
            p_j = partitions[j]
            mu_src = mu_scalar(source_term, float(u), p_j.low, p_j.medium, p_j.high)
            if mu_src < source_alpha - 1e-10:
                continue

            # Destination grid
            for v in j_grid:
                mu_dst = mu_scalar(destination_term, float(v), p_j.low, p_j.medium, p_j.high)
                if mu_dst < destination_alpha - 1e-10:
                    continue

                # Order constraint
                if s_ab != 0:
                    if s_ab * (float(v) - float(u)) < min_displacement - 1e-10:
                        continue
                elif min_displacement > 0:
                    if abs(float(v) - float(u)) < min_displacement - 1e-10:
                        continue

                delta = transition_objective_scalar(model, z, float(u), float(v), j, target_sign)
                if delta < min_delta:
                    min_delta = delta
                if delta > max_delta:
                    max_delta = delta

    if np.isinf(min_delta) or np.isinf(max_delta):
        return float("nan"), float("nan")
    return float(min_delta), float(max_delta)


def verify_envelope_contains_grid(
    model: Any,
    j: int,
    source_term: str,
    destination_term: str,
    source_alpha: float,
    destination_alpha: float,
    target_sign: int,
    lower: list[float],
    upper: list[float],
    certified_lower: float | None,
    certified_upper: float | None,
    context_lits: list[tuple[int, str, float]] | None = None,
    n_grid: int = 50,
    atol: float = 1e-6,
) -> tuple[bool, float, float, str]:
    """Check that the certified interval contains all grid-search values.

    Returns (pass, grid_min, grid_max, message).
    """
    grid_min, grid_max = brute_force_envelope(
        model, j, source_term, destination_term,
        source_alpha, destination_alpha, target_sign,
        lower, upper, context_lits,
        n_grid=n_grid,
    )

    if np.isnan(grid_min):
        return True, grid_min, grid_max, "no feasible points found in grid"

    msg_parts = []
    ok = True

    if certified_lower is not None and grid_min < certified_lower - atol:
        ok = False
        msg_parts.append(
            f"grid_min={grid_min:.6f} < certified_lower={certified_lower:.6f} "
            f"(violation={certified_lower - grid_min:.6e})"
        )

    if certified_upper is not None and grid_max > certified_upper + atol:
        ok = False
        msg_parts.append(
            f"grid_max={grid_max:.6f} > certified_upper={certified_upper:.6f} "
            f"(violation={grid_max - certified_upper:.6e})"
        )

    return ok, grid_min, grid_max, "; ".join(msg_parts) if msg_parts else "OK"
