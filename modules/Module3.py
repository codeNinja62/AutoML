"""Module 3: Issue Detection and User Approval.

This module is UI-framework agnostic. It detects common data quality issues and
returns structured findings that the Streamlit layer (Module8) can render.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


Severity = Literal['info', 'warning', 'critical']


@dataclass(frozen=True)
class IssueFinding:
    key: str
    title: str
    severity: Severity
    description: str
    affected_columns: list[str]
    metrics: dict[str, Any]
    suggested_fixes: list[str]

# 1. Detect missing values
def detect_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return missing counts per column (only columns with missing values)."""
    missing_info = df.isnull().sum()
    missing_info = missing_info[missing_info > 0]
    return missing_info

# 2. Detect outliers using Z-score
def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> dict[str, list[int]]:
    outlier_indices = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        series = df[column].astype(float)
        sigma = float(series.std(skipna=True))
        if not np.isfinite(sigma) or sigma == 0:
            continue
        z_scores = (series - float(series.mean(skipna=True))) / sigma
        mask = z_scores.abs() > float(threshold)
        if bool(mask.any()):
            outlier_indices[str(column)] = df.index[mask.fillna(False)].tolist()
    return outlier_indices


def detect_outliers_iqr(df: pd.DataFrame) -> dict[str, int]:
    """Detect outliers using IQR rule and return counts per numeric column."""
    out_counts: dict[str, int] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for column in numeric_cols:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr
        mask = (df[column] < lb) | (df[column] > ub)
        count = int(mask.sum())
        if count > 0:
            out_counts[str(column)] = count
    return out_counts

# 3. Detect class imbalance
def detect_class_imbalance(df: pd.DataFrame, target_column: Any, threshold: float = 0.7) -> pd.Series:
    """Return classes whose normalized frequency exceeds threshold.

    Note: This is a simple heuristic. The structured API below adds more context.
    """
    if target_column not in df.columns:
        return pd.Series(dtype=float)
    if len(df) == 0:
        return pd.Series(dtype=float)
    class_counts = df[target_column].value_counts(normalize=True, dropna=False)
    imbalanced_classes = class_counts[class_counts > float(threshold)]
    return imbalanced_classes

# 4. Detect high-cardinality categorical features
def detect_high_cardinality(df: pd.DataFrame, threshold: float = 0.1) -> list[str]:
    high_cardinality_features = []
    n = max(int(len(df)), 1)
    for column in df.select_dtypes(include=['object', 'category']).columns:
        unique_ratio = float(df[column].nunique(dropna=True)) / float(n)
        if unique_ratio > float(threshold):
            high_cardinality_features.append(str(column))
    return high_cardinality_features

# 5. Detect constant/near-constant features
def detect_constant_features(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    constant_features = []
    for column in df.columns:
        vc = df[column].value_counts(normalize=True, dropna=False)
        if vc.empty:
            continue
        top_freq = float(vc.iloc[0])
        if top_freq > float(threshold):
            constant_features.append(str(column))
    return constant_features

# 6. Show warnings and suggest fixes
def show_warnings_and_suggestions(df, target_column):
    warnings = {}
    
    missing_values = detect_missing_values(df)
    if not missing_values.empty:
        warnings['missing_values'] = missing_values

    outliers = detect_outliers_zscore(df)
    if outliers:
        warnings['outliers'] = outliers

    class_imbalance = detect_class_imbalance(df, target_column)
    if not class_imbalance.empty:
        warnings['class_imbalance'] = class_imbalance

    high_cardinality = detect_high_cardinality(df)
    if high_cardinality:
        warnings['high_cardinality'] = high_cardinality

    constant_features = detect_constant_features(df)
    if constant_features:
        warnings['constant_features'] = constant_features

    return warnings

# 7. Ask user confirmation before applying fixes
def get_user_confirmation(warnings):
    print("The following issues were detected in the dataset:")
    for issue, details in warnings.items():
        print(f"- {issue}: {details}")
    confirmation = input("Do you want to proceed with the suggested fixes? (yes/no): ")
    return confirmation.lower() == 'yes'


def detect_issues(
    df: pd.DataFrame,
    *,
    target_column: Any,
    missing_pct_warning: float = 5.0,
    missing_pct_critical: float = 30.0,
    outlier_pct_warning: float = 1.0,
    outlier_pct_critical: float = 10.0,
    class_imbalance_threshold: float = 0.7,
    high_cardinality_ratio_threshold: float = 0.1,
    constant_threshold: float = 0.95,
) -> dict[str, Any]:
    """Detect issues and return a structured summary.

    Returns a dict so it is JSON-serializable and easy to stash in Streamlit session.
    Keys are stable and can be used by other modules (e.g., class imbalance → class_weight).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')

    findings: list[IssueFinding] = []
    n_rows = int(len(df))
    n_rows_safe = max(n_rows, 1)

    # Missing values
    mv = detect_missing_values(df)
    missing_detail: dict[str, dict[str, Any]] = {}
    if not mv.empty:
        for col, count in mv.items():
            pct = (float(count) / float(n_rows_safe)) * 100.0
            missing_detail[str(col)] = {'count': int(count), 'pct': round(pct, 3)}

        worst_pct = max((d['pct'] for d in missing_detail.values()), default=0.0)
        severity: Severity = 'warning'
        if worst_pct >= float(missing_pct_critical):
            severity = 'critical'
        elif worst_pct < float(missing_pct_warning):
            severity = 'info'

        findings.append(
            IssueFinding(
                key='missing_values',
                title='Missing values',
                severity=severity,
                description='Some columns contain missing values.',
                affected_columns=sorted(missing_detail.keys()),
                metrics={'worst_missing_pct': worst_pct, 'total_missing_cells': int(df.isna().sum().sum())},
                suggested_fixes=[
                    'Impute numeric features using median/mean; categorical using most frequent.',
                    'If a column is mostly missing, consider dropping it (especially if non-informative).',
                ],
            )
        )

    # Outliers (IQR)
    out_iqr = detect_outliers_iqr(df)
    outlier_iqr_detail: dict[str, dict[str, Any]] = {}
    if out_iqr:
        for col, count in out_iqr.items():
            pct = (float(count) / float(n_rows_safe)) * 100.0
            outlier_iqr_detail[str(col)] = {'count': int(count), 'pct': round(pct, 3)}
        worst_pct = max((d['pct'] for d in outlier_iqr_detail.values()), default=0.0)
        severity = 'warning'
        if worst_pct >= float(outlier_pct_critical):
            severity = 'critical'
        elif worst_pct < float(outlier_pct_warning):
            severity = 'info'
        findings.append(
            IssueFinding(
                key='outliers_iqr',
                title='Outliers (IQR)',
                severity=severity,
                description='Numeric columns contain outliers under the IQR rule.',
                affected_columns=sorted(outlier_iqr_detail.keys()),
                metrics={'worst_outlier_pct': worst_pct},
                suggested_fixes=[
                    'Cap outliers using IQR bounds (safe default for demos).',
                    'Alternatively remove outlier rows if you have strong domain justification.',
                ],
            )
        )

    # Outliers (Z-score)
    out_z = detect_outliers_zscore(df)
    outlier_z_detail: dict[str, dict[str, Any]] = {}
    if out_z:
        for col, idxs in out_z.items():
            count = int(len(idxs))
            pct = (float(count) / float(n_rows_safe)) * 100.0
            outlier_z_detail[str(col)] = {'count': count, 'pct': round(pct, 3)}
        worst_pct = max((d['pct'] for d in outlier_z_detail.values()), default=0.0)
        severity = 'warning'
        if worst_pct >= float(outlier_pct_critical):
            severity = 'critical'
        elif worst_pct < float(outlier_pct_warning):
            severity = 'info'
        findings.append(
            IssueFinding(
                key='outliers_zscore',
                title='Outliers (Z-score)',
                severity=severity,
                description='Numeric columns contain outliers under Z-score thresholding.',
                affected_columns=sorted(outlier_z_detail.keys()),
                metrics={'threshold': 3.0, 'worst_outlier_pct': worst_pct},
                suggested_fixes=[
                    'If features are roughly normal, Z-score outlier removal can help.',
                    'Otherwise prefer IQR capping for robustness.',
                ],
            )
        )

    # Class imbalance
    class_imbalance = detect_class_imbalance(df, target_column, threshold=float(class_imbalance_threshold))
    class_imbalance_detail: dict[str, Any] = {}
    if not class_imbalance.empty:
        dist = df[target_column].value_counts(normalize=True, dropna=False).to_dict() if target_column in df.columns else {}
        max_ratio = float(max(dist.values())) if dist else 0.0
        class_imbalance_detail = {
            'threshold': float(class_imbalance_threshold),
            'dominant_classes': {str(k): float(v) for k, v in class_imbalance.to_dict().items()},
            'class_distribution': {str(k): float(v) for k, v in dist.items()},
            'max_class_ratio': round(max_ratio, 6),
        }
        findings.append(
            IssueFinding(
                key='class_imbalance',
                title='Class imbalance',
                severity='warning' if max_ratio < 0.9 else 'critical',
                description='The target distribution is imbalanced; accuracy can be misleading.',
                affected_columns=[str(target_column)],
                metrics=class_imbalance_detail,
                suggested_fixes=[
                    'Use F1-score (weighted/macro) as the primary metric.',
                    'Enable class_weight="balanced" for models that support it (LR/SVM/Tree/RF).',
                ],
            )
        )

    # High-cardinality categoricals
    high_card = detect_high_cardinality(df, threshold=float(high_cardinality_ratio_threshold))
    if high_card:
        findings.append(
            IssueFinding(
                key='high_cardinality',
                title='High-cardinality categoricals',
                severity='warning',
                description='Some categorical features have many unique values (can explode One-Hot features).',
                affected_columns=sorted([str(c) for c in high_card]),
                metrics={'threshold_ratio': float(high_cardinality_ratio_threshold)},
                suggested_fixes=[
                    'Prefer encoding="auto" (uses a heuristic to avoid huge one-hot expansions).',
                    'Consider dropping ID-like columns or grouping rare categories.',
                ],
            )
        )

    # Constant/near-constant features
    constant = detect_constant_features(df, threshold=float(constant_threshold))
    if constant:
        findings.append(
            IssueFinding(
                key='constant_features',
                title='Constant / near-constant features',
                severity='info',
                description='Some features are almost always the same value and add little signal.',
                affected_columns=sorted([str(c) for c in constant]),
                metrics={'threshold': float(constant_threshold)},
                suggested_fixes=[
                    'Drop these features using the “Remove columns” cleanup tool.',
                ],
            )
        )

    # Build a backwards-compatible summary dict too.
    summary: dict[str, Any] = {
        'missing_values': missing_detail,
        'outliers_iqr': outlier_iqr_detail,
        'outliers_zscore': outlier_z_detail,
        'class_imbalance': class_imbalance_detail,
        'high_cardinality': high_card,
        'constant_features': constant,
        'findings': [asdict(f) for f in findings],
    }
    # Remove empty keys for cleanliness.
    return {k: v for k, v in summary.items() if v not in ({}, [], None)}

