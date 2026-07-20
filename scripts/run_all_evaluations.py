"""Run the staged evaluation bundle used by the manuscript."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a fast subset for CI/local checks.")
    parser.add_argument("--skip-slow-fuzzy", action="store_true", help="Skip FARC-HD/IVTURS stages.")
    args = parser.parse_args(argv)

    datasets = ["iris", "pima_diabetes"] if args.smoke else []
    split_args = ["--outer-splits", "2", "--inner-splits", "2", "--max-samples", "120"] if args.smoke else []
    highdim_args = ["arrhythmia_binary", "--outer-splits", "2", "--max-samples", "120"] if args.smoke else []
    budget_args = ["--datasets", "pima_diabetes", "--outer-splits", "2", "--inner-splits", "2", "--max-samples", "120", "--budgets", "64", "256"] if args.smoke else []
    synthetic_args = ["--repeats", "1", "--outer-splits", "2", "--n-samples", "120", "--n-noise", "4"] if args.smoke else []

    stages = [
        ["uv", "run", "pytest"],
        ["uv", "run", "fysvm-prepare-datasets"],
        ["uv", "run", "python", "scripts/compare_recommendations.py", *datasets, *split_args],
        ["uv", "run", "python", "scripts/compare_modern_baselines.py", *datasets, *split_args],
        ["uv", "run", "python", "scripts/ablation_membership_svm.py", *datasets, *split_args],
        ["uv", "run", "python", "scripts/analyze_highdim.py", *budget_args],
        ["uv", "run", "python", "scripts/compare_highdim_adaptive.py", *highdim_args],
        ["uv", "run", "python", "scripts/synthetic_regime_experiments.py", *synthetic_args],
        ["uv", "run", "python", "scripts/regime_analysis.py"],
        ["uv", "run", "python", "scripts/generate_tables.py"],
        ["uv", "run", "python", "scripts/generate_figures.py", "--verify"],
    ]
    if not args.skip_slow_fuzzy:
        fuzzy_args = [*datasets, "--fast", *split_args] if args.smoke else []
        stages.insert(5, ["uv", "run", "python", "scripts/compare_fuzzy_baselines.py", *fuzzy_args])
        stages.insert(6, ["uv", "run", "python", "scripts/validate_furia_port.py", *datasets])

    Path("runs").mkdir(exist_ok=True)
    for command in stages:
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
