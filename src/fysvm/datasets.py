"""Dataset registry and preparation utilities for evaluation runs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine


TaskType = Literal["binary", "multiclass"]


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for one benchmark dataset."""

    slug: str
    name: str
    domain: str
    expected_samples: int
    expected_features: int
    task: TaskType
    target: str
    source: str
    note: str = ""


@dataclass(frozen=True)
class PreparedDataset:
    """A numeric dataset ready for model evaluation."""

    spec: DatasetSpec
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]


_UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases"
_UCI_STATIC = "https://archive.ics.uci.edu/static/public"


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "breast_cancer_diagnostic",
        "Breast Cancer Wisconsin Diagnostic",
        "medical, breast cancer diagnosis",
        569,
        30,
        "binary",
        "malignant vs benign",
        "sklearn.load_breast_cancer",
    ),
    DatasetSpec(
        "breast_cancer_original",
        "Breast Cancer Wisconsin Original",
        "medical, breast cancer diagnosis",
        699,
        9,
        "binary",
        "malignant vs benign",
        f"{_UCI_BASE}/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        "Rows with missing values are retained and median-imputed by the harness.",
    ),
    DatasetSpec(
        "mammographic_mass",
        "Mammographic Mass",
        "medical, breast imaging",
        961,
        5,
        "binary",
        "benign vs malignant",
        f"{_UCI_BASE}/mammographic-masses/mammographic_masses.data",
    ),
    DatasetSpec(
        "breast_tissue",
        "Breast Tissue",
        "medical, breast tissue impedance",
        106,
        9,
        "multiclass",
        "6 tissue classes",
        f"{_UCI_BASE}/00192/BreastTissue.xls",
    ),
    DatasetSpec(
        "heart_cleveland",
        "Heart Disease Cleveland",
        "medical, cardiac diagnosis",
        303,
        13,
        "binary",
        "disease vs no disease",
        f"{_UCI_BASE}/heart-disease/processed.cleveland.data",
        "Targets greater than zero are binarized as disease present.",
    ),
    DatasetSpec(
        "statlog_heart",
        "Statlog Heart",
        "medical, cardiac diagnosis",
        270,
        13,
        "binary",
        "heart disease absent/present",
        f"{_UCI_BASE}/statlog/heart/heart.dat",
    ),
    DatasetSpec(
        "spect_heart",
        "SPECT Heart",
        "medical, cardiac SPECT",
        267,
        22,
        "binary",
        "normal vs abnormal",
        f"{_UCI_BASE}/spect/SPECT.train and SPECT.test",
    ),
    DatasetSpec(
        "spectf_heart",
        "SPECTF Heart",
        "medical, cardiac SPECT",
        267,
        44,
        "binary",
        "normal vs abnormal",
        f"{_UCI_BASE}/spect/SPECTF.train and SPECTF.test",
    ),
    DatasetSpec(
        "pima_diabetes",
        "Pima Indians Diabetes",
        "medical, diabetes screening",
        768,
        8,
        "binary",
        "diabetes positive/negative",
        "OpenML diabetes, with UCI-format CSV fallback",
    ),
    DatasetSpec(
        "diabetic_retinopathy_debrecen",
        "Diabetic Retinopathy Debrecen",
        "medical, retinopathy screening",
        1151,
        19,
        "binary",
        "signs of diabetic retinopathy vs none",
        f"{_UCI_BASE}/00329/messidor_features.arff",
    ),
    DatasetSpec(
        "parkinsons",
        "Parkinsons",
        "medical, voice-based Parkinson's detection",
        195,
        22,
        "binary",
        "Parkinson's vs healthy",
        f"{_UCI_BASE}/parkinsons/parkinsons.data",
    ),
    DatasetSpec(
        "parkinsons_disease_classification",
        "Parkinson's Disease Classification",
        "medical, voice-based Parkinson's detection",
        756,
        754,
        "binary",
        "Parkinson's vs healthy/control",
        f"{_UCI_STATIC}/470/parkinson+s+disease+classification.zip",
        "High-dimensional dataset; the default harness uses one-term rules for scalability.",
    ),
    DatasetSpec(
        "ilpd",
        "Indian Liver Patient Dataset",
        "medical, liver disease",
        583,
        10,
        "binary",
        "liver patient vs non-liver patient",
        f"{_UCI_BASE}/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
    ),
    DatasetSpec(
        "dermatology",
        "Dermatology",
        "medical, skin disease diagnosis",
        366,
        34,
        "multiclass",
        "6 erythemato-squamous disease classes",
        f"{_UCI_BASE}/dermatology/dermatology.data",
    ),
    DatasetSpec(
        "haberman_survival",
        "Haberman's Survival",
        "medical, breast cancer survival",
        306,
        3,
        "binary",
        "survived 5+ years vs not",
        f"{_UCI_BASE}/haberman/haberman.data",
    ),
    DatasetSpec(
        "vertebral_column_2c",
        "Vertebral Column 2-Class",
        "medical, orthopaedic diagnosis",
        310,
        6,
        "binary",
        "normal vs abnormal",
        f"{_UCI_BASE}/00212/vertebral_column_data.zip",
    ),
    DatasetSpec(
        "arrhythmia_binary",
        "Arrhythmia Binary",
        "medical, ECG/cardiac rhythm",
        452,
        279,
        "binary",
        "normal vs arrhythmia",
        f"{_UCI_BASE}/arrhythmia/arrhythmia.data",
        "Class 1 is normal; all other classes are binarized as arrhythmia.",
    ),
    DatasetSpec(
        "iris",
        "Iris",
        "classic numeric benchmark",
        150,
        4,
        "multiclass",
        "3 iris species",
        "sklearn.load_iris",
    ),
    DatasetSpec(
        "wine",
        "Wine",
        "classic chemistry benchmark",
        178,
        13,
        "multiclass",
        "3 wine cultivars",
        "sklearn.load_wine",
    ),
    DatasetSpec(
        "digits",
        "Digits",
        "image-derived numeric benchmark",
        1797,
        64,
        "multiclass",
        "10 digit classes",
        "sklearn.load_digits",
    ),
)

_SPECS_BY_SLUG = {spec.slug: spec for spec in DATASET_SPECS}


def list_datasets() -> list[DatasetSpec]:
    """Return the datasets known to the evaluation harness."""

    return list(DATASET_SPECS)


def load_dataset(slug: str, data_dir: str | Path = "datasets/prepared") -> PreparedDataset:
    """Load a prepared dataset from ``data_dir``."""

    path = Path(data_dir) / f"{slug}.npz"
    if not path.exists():
        if slug not in _SPECS_BY_SLUG:
            _require_spec(slug)
        raise FileNotFoundError(
            f"Prepared dataset not found: {path}. Run `fysvm-prepare-datasets {slug}` first."
        )

    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        X = data["X"].astype(np.float64, copy=False)
        y = data["y"].astype(str, copy=False)
        feature_names = data["feature_names"].astype(str).tolist()
        target_names = data["target_names"].astype(str).tolist()

    saved_slug = metadata.get("slug")
    if saved_slug != slug:
        raise ValueError(f"Dataset file {path} contains slug {saved_slug!r}, expected {slug!r}.")
    spec = _SPECS_BY_SLUG.get(slug) or DatasetSpec(
        slug=slug,
        name=str(metadata.get("name", slug)),
        domain=str(metadata.get("domain", "local")),
        expected_samples=int(metadata.get("expected_samples", X.shape[0])),
        expected_features=int(metadata.get("expected_features", X.shape[1])),
        task="multiclass" if len(np.unique(y)) > 2 else "binary",
        target=str(metadata.get("target", "local target")),
        source=str(metadata.get("source", path)),
        note=str(metadata.get("note", "")),
    )
    return PreparedDataset(spec, X, y, feature_names, target_names)


def prepare_dataset(
    slug: str,
    output_dir: str | Path = "datasets/prepared",
    *,
    force: bool = False,
) -> Path:
    """Prepare one dataset and write it as a compressed ``.npz`` file."""

    spec = _require_spec(slug)
    output_path = Path(output_dir) / f"{slug}.npz"
    if output_path.exists() and not force:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared = _LOADERS[slug](spec)
    _validate_prepared_dataset(prepared)
    _save_prepared_dataset(prepared, output_path)
    return output_path


def prepare_datasets(
    slugs: Iterable[str] | None = None,
    output_dir: str | Path = "datasets/prepared",
    *,
    force: bool = False,
) -> list[Path]:
    """Prepare multiple datasets and return their output paths."""

    selected = list(slugs) if slugs is not None else [spec.slug for spec in DATASET_SPECS]
    return [prepare_dataset(slug, output_dir, force=force) for slug in selected]


def write_manifest(output_path: str | Path = "datasets/manifest.json") -> Path:
    """Write the dataset registry manifest used by the harness."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "datasets": [
            {
                **asdict(spec),
                "prepared_file": f"prepared/{spec.slug}.npz",
            }
            for spec in DATASET_SPECS
        ]
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for dataset preparation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset slugs to prepare. Defaults to all registered datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/prepared",
        help="Directory where prepared .npz files are written.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument(
        "--manifest",
        default="datasets/manifest.json",
        help="Path for the JSON dataset manifest.",
    )
    args = parser.parse_args(argv)

    write_manifest(args.manifest)
    selected = args.datasets or None
    for path in prepare_datasets(selected, args.output_dir, force=args.force):
        print(path)


def _require_spec(slug: str) -> DatasetSpec:
    try:
        return _SPECS_BY_SLUG[slug]
    except KeyError as exc:
        known = ", ".join(sorted(_SPECS_BY_SLUG))
        raise KeyError(f"Unknown dataset slug {slug!r}. Known datasets: {known}") from exc


def _sklearn_dataset(
    spec: DatasetSpec,
    loader: Callable[[], Any],
) -> PreparedDataset:
    data = loader()
    y = np.asarray([str(data.target_names[target]) for target in data.target], dtype=str)
    return PreparedDataset(
        spec=spec,
        X=np.asarray(data.data, dtype=np.float64),
        y=y,
        feature_names=[str(name) for name in data.feature_names],
        target_names=[str(name) for name in data.target_names],
    )


def _load_breast_cancer_diagnostic(spec: DatasetSpec) -> PreparedDataset:
    return _sklearn_dataset(spec, load_breast_cancer)


def _load_iris(spec: DatasetSpec) -> PreparedDataset:
    return _sklearn_dataset(spec, load_iris)


def _load_wine(spec: DatasetSpec) -> PreparedDataset:
    return _sklearn_dataset(spec, load_wine)


def _load_digits_dataset(spec: DatasetSpec) -> PreparedDataset:
    data = load_digits()
    y = np.asarray([str(target) for target in data.target], dtype=str)
    return PreparedDataset(
        spec=spec,
        X=np.asarray(data.data, dtype=np.float64),
        y=y,
        feature_names=[f"pixel_{index}" for index in range(data.data.shape[1])],
        target_names=[str(target) for target in data.target_names],
    )


def _load_breast_cancer_original(spec: DatasetSpec) -> PreparedDataset:
    text = _download_text(spec.source)
    rows = _parse_csv_text(text)
    X = np.asarray([[float_or_nan(value) for value in row[1:10]] for row in rows], dtype=float)
    y = np.asarray(["benign" if row[10] == "2" else "malignant" for row in rows], dtype=str)
    return PreparedDataset(
        spec,
        X,
        y,
        [
            "clump_thickness",
            "uniformity_cell_size",
            "uniformity_cell_shape",
            "marginal_adhesion",
            "single_epithelial_cell_size",
            "bare_nuclei",
            "bland_chromatin",
            "normal_nucleoli",
            "mitoses",
        ],
        ["benign", "malignant"],
    )


def _load_mammographic_mass(spec: DatasetSpec) -> PreparedDataset:
    rows = _parse_csv_text(_download_text(spec.source))
    X = np.asarray([[float_or_nan(value) for value in row[:5]] for row in rows], dtype=float)
    y = np.asarray(["benign" if row[5] == "0" else "malignant" for row in rows], dtype=str)
    return PreparedDataset(
        spec,
        X,
        y,
        ["birads", "age", "shape", "margin", "density"],
        ["benign", "malignant"],
    )


def _load_breast_tissue(spec: DatasetSpec) -> PreparedDataset:
    raw = _download_bytes(spec.source)
    df = pd.read_excel(io.BytesIO(raw), sheet_name="Data")
    df = df.dropna(how="all")
    target_column = "Class"
    feature_columns = [
        column
        for column in df.columns
        if column not in {"Case #", target_column} and pd.api.types.is_numeric_dtype(df[column])
    ]
    X = df[feature_columns].to_numpy(dtype=float)
    y = df[target_column].astype(str).to_numpy()
    return PreparedDataset(
        spec,
        X,
        y,
        [str(column) for column in feature_columns],
        sorted(np.unique(y).astype(str).tolist()),
    )


def _load_heart_cleveland(spec: DatasetSpec) -> PreparedDataset:
    rows = _parse_csv_text(_download_text(spec.source))
    X = np.asarray([[float_or_nan(value) for value in row[:13]] for row in rows], dtype=float)
    y = np.asarray(["no_disease" if row[13] == "0" else "disease" for row in rows], dtype=str)
    return PreparedDataset(spec, X, y, _heart_feature_names(), ["no_disease", "disease"])


def _load_statlog_heart(spec: DatasetSpec) -> PreparedDataset:
    rows = [line.split() for line in _download_text(spec.source).splitlines() if line.strip()]
    X = np.asarray([[float(value) for value in row[:13]] for row in rows], dtype=float)
    y = np.asarray(["absent" if row[13] == "1" else "present" for row in rows], dtype=str)
    return PreparedDataset(spec, X, y, _heart_feature_names(), ["absent", "present"])


def _load_spect_heart(spec: DatasetSpec) -> PreparedDataset:
    rows = _load_two_uci_csv_files(
        f"{_UCI_BASE}/spect/SPECT.train",
        f"{_UCI_BASE}/spect/SPECT.test",
    )
    X = np.asarray([[float(value) for value in row[1:]] for row in rows], dtype=float)
    y = np.asarray(["normal" if row[0] == "1" else "abnormal" for row in rows], dtype=str)
    return PreparedDataset(
        spec,
        X,
        y,
        [f"spect_region_{index}" for index in range(1, X.shape[1] + 1)],
        ["abnormal", "normal"],
    )


def _load_spectf_heart(spec: DatasetSpec) -> PreparedDataset:
    rows = _load_two_uci_csv_files(
        f"{_UCI_BASE}/spect/SPECTF.train",
        f"{_UCI_BASE}/spect/SPECTF.test",
    )
    X = np.asarray([[float(value) for value in row[1:]] for row in rows], dtype=float)
    y = np.asarray(["normal" if row[0] == "1" else "abnormal" for row in rows], dtype=str)
    return PreparedDataset(
        spec,
        X,
        y,
        [f"spectf_feature_{index}" for index in range(1, X.shape[1] + 1)],
        ["abnormal", "normal"],
    )


def _load_pima_diabetes(spec: DatasetSpec) -> PreparedDataset:
    urls = [
        f"{_UCI_BASE}/pima-indians-diabetes/pima-indians-diabetes.data",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    ]
    last_error: Exception | None = None
    for url in urls:
        try:
            rows = _parse_csv_text(_download_text(url))
            break
        except Exception as exc:  # pragma: no cover - depends on external mirrors
            last_error = exc
    else:  # pragma: no cover - depends on external mirrors
        raise RuntimeError("Unable to download Pima Indians Diabetes dataset.") from last_error

    X = np.asarray([[float(value) for value in row[:8]] for row in rows], dtype=float)
    y = np.asarray(["negative" if row[8] == "0" else "positive" for row in rows], dtype=str)
    return PreparedDataset(
        spec,
        X,
        y,
        [
            "pregnancies",
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
            "diabetes_pedigree",
            "age",
        ],
        ["negative", "positive"],
    )


def _load_diabetic_retinopathy(spec: DatasetSpec) -> PreparedDataset:
    text = _download_text(spec.source)
    records, metadata = arff.loadarff(io.StringIO(text))
    df = pd.DataFrame(records)
    class_column = df.columns[-1]
    X = df.drop(columns=[class_column]).to_numpy(dtype=float)
    y = df[class_column].map(_decode_arff_value).map(
        {"0": "no_retinopathy", "1": "retinopathy"}
    ).to_numpy(dtype=str)
    return PreparedDataset(
        spec,
        X,
        y,
        [str(name) for name in metadata.names()[:-1]],
        ["no_retinopathy", "retinopathy"],
    )


def _load_parkinsons(spec: DatasetSpec) -> PreparedDataset:
    df = pd.read_csv(io.StringIO(_download_text(spec.source)))
    y = df["status"].map({0: "healthy", 1: "parkinsons"}).to_numpy(dtype=str)
    feature_columns = [column for column in df.columns if column not in {"name", "status"}]
    return PreparedDataset(
        spec,
        df[feature_columns].to_numpy(dtype=float),
        y,
        [str(column) for column in feature_columns],
        ["healthy", "parkinsons"],
    )


def _load_parkinsons_disease_classification(spec: DatasetSpec) -> PreparedDataset:
    raw = _download_bytes(spec.source)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if csv_names:
            df = pd.read_csv(archive.open(csv_names[0]))
        else:
            rar_name = next(
                name for name in archive.namelist() if name.lower().endswith(".rar")
            )
            df = pd.read_csv(io.BytesIO(_extract_csv_from_rar(archive.read(rar_name))))

    target_column = "class" if "class" in df.columns else df.columns[-1]
    drop_columns = [target_column]
    for candidate in ("id", "ID", "subject", "subject#"):
        if candidate in df.columns:
            drop_columns.append(candidate)

    y_raw = pd.to_numeric(df[target_column], errors="coerce")
    feature_frame = df.drop(columns=drop_columns).apply(pd.to_numeric, errors="coerce")
    valid = y_raw.notna()
    feature_frame = feature_frame.loc[valid]
    y = y_raw.loc[valid].astype(int).map({0: "healthy", 1: "parkinsons"}).to_numpy(dtype=str)
    return PreparedDataset(
        spec,
        feature_frame.to_numpy(dtype=float),
        y,
        [str(column) for column in feature_frame.columns],
        ["healthy", "parkinsons"],
    )


def _load_ilpd(spec: DatasetSpec) -> PreparedDataset:
    rows = _parse_csv_text(_download_text(spec.source))
    X_values: list[list[float]] = []
    y_values: list[str] = []
    for row in rows:
        gender = 1.0 if row[1].strip().lower().startswith("male") else 0.0
        X_values.append([float_or_nan(row[0]), gender, *[float_or_nan(value) for value in row[2:10]]])
        y_values.append("liver_patient" if row[10] == "1" else "non_liver_patient")
    return PreparedDataset(
        spec,
        np.asarray(X_values, dtype=float),
        np.asarray(y_values, dtype=str),
        [
            "age",
            "gender_male",
            "total_bilirubin",
            "direct_bilirubin",
            "alkaline_phosphotase",
            "alamine_aminotransferase",
            "aspartate_aminotransferase",
            "total_proteins",
            "albumin",
            "albumin_globulin_ratio",
        ],
        ["non_liver_patient", "liver_patient"],
    )


def _load_dermatology(spec: DatasetSpec) -> PreparedDataset:
    rows = _parse_csv_text(_download_text(spec.source))
    X = np.asarray([[float_or_nan(value) for value in row[:34]] for row in rows], dtype=float)
    y = np.asarray([f"class_{row[34]}" for row in rows], dtype=str)
    return PreparedDataset(
        spec,
        X,
        y,
        [f"dermatology_feature_{index}" for index in range(1, 35)],
        [f"class_{index}" for index in range(1, 7)],
    )


def _load_haberman_survival(spec: DatasetSpec) -> PreparedDataset:
    rows = _parse_csv_text(_download_text(spec.source))
    X = np.asarray([[float(value) for value in row[:3]] for row in rows], dtype=float)
    y = np.asarray(["survived_5_years" if row[3] == "1" else "died_within_5_years" for row in rows], dtype=str)
    return PreparedDataset(
        spec,
        X,
        y,
        ["age", "operation_year", "positive_axillary_nodes"],
        ["survived_5_years", "died_within_5_years"],
    )


def _load_vertebral_column_2c(spec: DatasetSpec) -> PreparedDataset:
    raw = _download_bytes(spec.source)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        name = next(name for name in archive.namelist() if name.endswith("column_2C.dat"))
        text = archive.read(name).decode("utf-8")
    rows = [line.split() for line in text.splitlines() if line.strip()]
    X = np.asarray([[float(value) for value in row[:6]] for row in rows], dtype=float)
    y = np.asarray(["normal" if row[6] == "NO" else "abnormal" for row in rows], dtype=str)
    return PreparedDataset(
        spec,
        X,
        y,
        [
            "pelvic_incidence",
            "pelvic_tilt",
            "lumbar_lordosis_angle",
            "sacral_slope",
            "pelvic_radius",
            "degree_spondylolisthesis",
        ],
        ["normal", "abnormal"],
    )


def _load_arrhythmia_binary(spec: DatasetSpec) -> PreparedDataset:
    rows = _parse_csv_text(_download_text(spec.source))
    X = np.asarray([[float_or_nan(value) for value in row[:279]] for row in rows], dtype=float)
    y = np.asarray(["normal" if row[279] == "1" else "arrhythmia" for row in rows], dtype=str)
    return PreparedDataset(
        spec,
        X,
        y,
        [f"arrhythmia_feature_{index}" for index in range(1, 280)],
        ["normal", "arrhythmia"],
    )


_LOADERS: dict[str, Callable[[DatasetSpec], PreparedDataset]] = {
    "breast_cancer_diagnostic": _load_breast_cancer_diagnostic,
    "breast_cancer_original": _load_breast_cancer_original,
    "mammographic_mass": _load_mammographic_mass,
    "breast_tissue": _load_breast_tissue,
    "heart_cleveland": _load_heart_cleveland,
    "statlog_heart": _load_statlog_heart,
    "spect_heart": _load_spect_heart,
    "spectf_heart": _load_spectf_heart,
    "pima_diabetes": _load_pima_diabetes,
    "diabetic_retinopathy_debrecen": _load_diabetic_retinopathy,
    "parkinsons": _load_parkinsons,
    "parkinsons_disease_classification": _load_parkinsons_disease_classification,
    "ilpd": _load_ilpd,
    "dermatology": _load_dermatology,
    "haberman_survival": _load_haberman_survival,
    "vertebral_column_2c": _load_vertebral_column_2c,
    "arrhythmia_binary": _load_arrhythmia_binary,
    "iris": _load_iris,
    "wine": _load_wine,
    "digits": _load_digits_dataset,
}


def _validate_prepared_dataset(dataset: PreparedDataset) -> None:
    if dataset.X.ndim != 2:
        raise ValueError(f"{dataset.spec.slug}: X must be two-dimensional.")
    if dataset.y.shape != (dataset.X.shape[0],):
        raise ValueError(f"{dataset.spec.slug}: y must have one label per row.")
    if len(dataset.feature_names) != dataset.X.shape[1]:
        raise ValueError(f"{dataset.spec.slug}: feature_names length does not match X.")
    if len(np.unique(dataset.y)) < 2:
        raise ValueError(f"{dataset.spec.slug}: dataset must have at least two classes.")


def _save_prepared_dataset(dataset: PreparedDataset, output_path: Path) -> None:
    metadata = {
        **asdict(dataset.spec),
        "actual_samples": int(dataset.X.shape[0]),
        "actual_features": int(dataset.X.shape[1]),
        "classes": dataset.target_names,
    }
    np.savez_compressed(
        output_path,
        X=np.asarray(dataset.X, dtype=np.float64),
        y=np.asarray(dataset.y, dtype=str),
        feature_names=np.asarray(dataset.feature_names, dtype=str),
        target_names=np.asarray(dataset.target_names, dtype=str),
        metadata=np.asarray(json.dumps(metadata), dtype=str),
    )


def _download_text(url: str) -> str:
    return _download_bytes(url).decode("utf-8", errors="replace")


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "fysvm-dataset-prep/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _parse_csv_text(text: str) -> list[list[str]]:
    reader = csv.reader(io.StringIO(text.strip()))
    return [row for row in reader if row]


def _load_two_uci_csv_files(train_url: str, test_url: str) -> list[list[str]]:
    return _parse_csv_text(_download_text(train_url)) + _parse_csv_text(_download_text(test_url))


def float_or_nan(value: str) -> float:
    stripped = value.strip()
    if stripped in {"", "?", "nan", "NaN"}:
        return float("nan")
    return float(stripped)


def _decode_arff_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _heart_feature_names() -> list[str]:
    return [
        "age",
        "sex",
        "chest_pain_type",
        "resting_blood_pressure",
        "serum_cholesterol",
        "fasting_blood_sugar",
        "resting_ecg",
        "max_heart_rate",
        "exercise_induced_angina",
        "oldpeak",
        "slope",
        "major_vessels",
        "thal",
    ]


def _download_to_tempfile(url: str) -> Path:
    raw = _download_bytes(url)
    handle = tempfile.NamedTemporaryFile(delete=False)
    handle.write(raw)
    handle.close()
    return Path(handle.name)


def _extract_csv_from_rar(raw: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".rar", delete=False) as handle:
        handle.write(raw)
        rar_path = Path(handle.name)
    try:
        if shutil.which("unrar"):
            result = subprocess.run(
                ["unrar", "p", "-inul", str(rar_path)],
                check=True,
                capture_output=True,
            )
            return result.stdout
        if shutil.which("7z"):
            result = subprocess.run(
                ["7z", "x", "-so", str(rar_path)],
                check=True,
                capture_output=True,
            )
            return result.stdout
    finally:
        rar_path.unlink(missing_ok=True)
    raise RuntimeError(
        "Parkinson's Disease Classification requires extracting a nested RAR; "
        "install `unrar` or `7z`, or provide a prepared .npz file."
    )
