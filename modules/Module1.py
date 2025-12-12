"""Module 1: Dataset Upload and Basic Information.

This module provides small, reusable helpers for basic dataset profiling.
It is UI-framework agnostic (works for Streamlit/CLI/notebooks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def upload_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV from disk."""
    return pd.read_csv(file_path)


def get_shape(df: pd.DataFrame) -> tuple[int, int]:
    """Return (rows, cols)."""
    return int(df.shape[0]), int(df.shape[1])


def get_column_types(df: pd.DataFrame) -> pd.Series:
    """Return pandas dtypes per column."""
    return df.dtypes


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric summary statistics."""
    return df.describe()


def get_class_distribution(df: pd.DataFrame, target_column: str) -> pd.Series:
    """Return class counts for the chosen target column."""
    return df[target_column].value_counts(dropna=False)


def get_missing_cell_count(df: pd.DataFrame) -> int:
    """Return the number of missing cells in the dataset."""
    return int(df.isna().sum().sum())


def get_duplicate_row_count(df: pd.DataFrame) -> int:
    """Return the number of duplicate rows."""
    return int(df.duplicated().sum())


def get_unique_counts(df: pd.DataFrame, dropna: bool = True) -> pd.Series:
    """Return unique counts per column."""
    return df.nunique(dropna=dropna)


def get_cardinality_buckets(
    df: pd.DataFrame,
    dropna: bool = True,
    low_max: int = 10,
    medium_max: int = 50,
    high_max: int = 200,
) -> pd.DataFrame:
    """Bucket columns by cardinality (unique value count).

    Buckets are useful for choosing encoders and for spotting ID-like columns.
    """
    uniq = get_unique_counts(df, dropna=dropna)
    buckets = []
    for col, n in uniq.items():
        if n <= low_max:
            bucket = 'low'
        elif n <= medium_max:
            bucket = 'medium'
        elif n <= high_max:
            bucket = 'high'
        else:
            bucket = 'very_high'
        buckets.append({'column': col, 'unique': int(n), 'bucket': bucket})
    return pd.DataFrame(buckets).sort_values(['bucket', 'unique', 'column']).reset_index(drop=True)


def get_dataset_schema(df: pd.DataFrame, sample_values: int = 3) -> pd.DataFrame:
    """Return a compact schema/profile table for reporting.

    Columns: name, dtype, non_null, missing_pct, unique, samples.
    """
    rows = []
    n_rows = int(len(df))
    uniq = get_unique_counts(df)
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        missing_pct = float(0.0 if n_rows == 0 else (1 - (non_null / n_rows)) * 100)
        samples = df[col].dropna().astype(str).head(sample_values).tolist()
        rows.append(
            {
                'column': col,
                'dtype': str(df[col].dtype),
                'non_null': non_null,
                'missing_pct': round(missing_pct, 3),
                'unique': int(uniq.get(col, 0)),
                'samples': samples,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class TargetValidation:
    ok: bool
    errors: list[str]
    warnings: list[str]
    missing_target: int
    n_classes: int
    class_counts: dict[str, int]


def validate_target_column(
    df: pd.DataFrame,
    target_column: Any,
    *,
    min_classes: int = 2,
    min_samples_per_class: int = 2,
    max_classes_warning: int = 50,
) -> TargetValidation:
    """Validate a target column for classification.

    Returns an object containing errors/warnings and class distribution.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if target_column not in df.columns:
        return TargetValidation(
            ok=False,
            errors=[f"Target column {target_column!r} not found."],
            warnings=[],
            missing_target=0,
            n_classes=0,
            class_counts={},
        )

    y = df[target_column]
    missing_target = int(y.isna().sum())
    if missing_target > 0:
        warnings.append(f'Target has {missing_target} missing values; those rows may be dropped during training.')

    counts = y.value_counts(dropna=True)
    n_classes = int(counts.shape[0])

    if n_classes < min_classes:
        errors.append('Target must contain at least 2 classes for classification.')

    rare = counts[counts < int(min_samples_per_class)]
    if not rare.empty:
        warnings.append(
            f"Some classes have fewer than {min_samples_per_class} samples: "
            + ', '.join([f"{idx}({int(v)})" for idx, v in rare.items()])
        )

    if n_classes > int(max_classes_warning):
        warnings.append(
            f'Target has {n_classes} classes; this may be hard to model and visualizations may be less readable.'
        )

    class_counts = {str(k): int(v) for k, v in counts.items()}
    ok = len(errors) == 0
    return TargetValidation(
        ok=ok,
        errors=errors,
        warnings=warnings,
        missing_target=missing_target,
        n_classes=n_classes,
        class_counts=class_counts,
    )


def infer_target_candidates(
    df: pd.DataFrame,
    *,
    name_hints: tuple[str, ...] = ('target', 'label', 'class', 'y'),
    max_unique_ratio: float = 0.2,
    max_unique_abs: int = 50,
) -> list[str]:
    """Suggest likely target columns.

    Heuristics:
    - Column name matches common hints.
    - Low-ish cardinality relative to dataset size.
    """
    cols = list(df.columns)
    n = max(int(len(df)), 1)
    uniq = get_unique_counts(df)

    scored: list[tuple[float, str]] = []
    for c in cols:
        name_score = 1.0 if any(h in str(c).lower() for h in name_hints) else 0.0
        u = float(uniq.get(c, 0))
        ratio = u / float(n)
        ratio_score = 1.0 if (u <= max_unique_abs or ratio <= max_unique_ratio) else 0.0
        score = (2.0 * name_score) + ratio_score
        if score > 0:
            scored.append((score, str(c)))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, c in scored]


def get_dataset_profile(df: pd.DataFrame, target_col: str | None = None) -> dict[str, Any]:
    """Return a small, JSON-serializable dataset profile."""
    profile: dict[str, Any] = {
        'rows': int(df.shape[0]),
        'cols': int(df.shape[1]),
        'missing_cells': get_missing_cell_count(df),
        'duplicate_rows': get_duplicate_row_count(df),
    }
    if target_col and target_col in df.columns:
        tv = validate_target_column(df, target_col)
        profile['target'] = {
            'column': target_col,
            'ok': tv.ok,
            'missing_target': tv.missing_target,
            'n_classes': tv.n_classes,
            'class_counts': tv.class_counts,
            'warnings': tv.warnings,
            'errors': tv.errors,
        }
    return profile