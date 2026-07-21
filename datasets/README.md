# Evaluation Datasets

Prepared datasets live in `datasets/prepared/*.npz` and are loaded by the
evaluation harness.

Prepare all registered datasets:

```bash
uv run fysvm-prepare-datasets
```

Prepare a subset:

```bash
uv run fysvm-prepare-datasets iris wine breast_cancer_diagnostic
```

Each `.npz` contains numeric `X`, string labels `y`, `feature_names`,
`target_names`, and JSON `metadata`. Missing values from UCI sources are kept as
`NaN`; the harness applies median imputation inside each training fold.
