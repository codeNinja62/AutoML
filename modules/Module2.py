# Module 2: Automated Exploratory Data Analysis (EDA)
# Missing value analysis
# Outlier detection (IQR / Z-score)
# Correlation matrix
# Numerical feature distributions
# Categorical feature bar plots
# Train/test split summary


"""Module 2: Automated Exploratory Data Analysis (EDA)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _sample_for_plots(df: pd.DataFrame, *, max_rows: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Return a sampled dataframe for plotting (keeps UI responsive on large datasets)."""
    if len(df) <= int(max_rows):
        return df
    return df.sample(n=int(max_rows), random_state=int(random_state))


# 1. Missing value analysis
def missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value counts and percentages per column."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / max(len(df), 1)) * 100
    out = pd.DataFrame({'missing_count': missing_count, 'missing_pct': missing_pct})
    out = out[out['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    return out


def plot_missing_values(missing_df: pd.DataFrame):
    """Create a bar plot (matplotlib Figure) of missing percentages."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if missing_df.empty:
        ax.text(0.5, 0.5, 'No missing values detected.', ha='center', va='center')
        ax.axis('off')
        return fig
    plot_df = missing_df.reset_index().rename(columns={'index': 'column'})
    sns.barplot(x='column', y='missing_pct', data=plot_df, ax=ax)
    ax.set_title('Missing Values (%) by Feature')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Missing %')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig

# 2. Outlier detection using IQR
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# 3. Correlation matrix
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] == 0:
        return pd.DataFrame()
    return numeric_df.corr()


def plot_correlation_heatmap(corr_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    if corr_df.empty:
        ax.text(0.5, 0.5, 'No numeric features available for correlation.', ha='center', va='center')
        ax.axis('off')
        return fig
    sns.heatmap(corr_df, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    fig.tight_layout()
    return fig

# 4. Numerical feature distributions
def plot_numerical_distribution(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = _sample_for_plots(df)
    series = plot_df[column].dropna()
    if series.empty:
        ax.text(0.5, 0.5, 'No non-missing values to plot.', ha='center', va='center')
        ax.axis('off')
        return fig
    sns.histplot(series, kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    fig.tight_layout()
    return fig


def plot_categorical_distribution(df: pd.DataFrame, column: str, top_n: int = 20):
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = _sample_for_plots(df)
    counts = plot_df[column].astype(str).fillna('NaN').value_counts().head(top_n)
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title(f'Count Plot of {column} (top {top_n})')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig


def plot_outlier_boxplot(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    plot_df = _sample_for_plots(df)
    series = plot_df[column].dropna()
    if series.empty:
        ax.text(0.5, 0.5, 'No non-missing values to plot.', ha='center', va='center')
        ax.axis('off')
        return fig
    sns.boxplot(x=series, ax=ax)
    ax.set_title(f'Boxplot: {column}')
    fig.tight_layout()
    return fig


def outlier_summary_iqr(df: pd.DataFrame, *, columns: list[Any] | None = None) -> pd.DataFrame:
    """Return outlier counts/% per numeric column using the IQR rule."""
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()
    rows: list[dict[str, Any]] = []
    n = max(int(len(df)), 1)
    for col in columns:
        if col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr
        count = int(((series < lb) | (series > ub)).sum())
        if count <= 0:
            continue
        rows.append(
            {
                'column': str(col),
                'outlier_count': count,
                'outlier_pct': round((count / n) * 100.0, 3),
                'method': 'iqr',
            }
        )
    if not rows:
        return pd.DataFrame(columns=['column', 'outlier_count', 'outlier_pct', 'method'])
    return pd.DataFrame(rows).sort_values(['outlier_pct', 'outlier_count', 'column'], ascending=[False, False, True])


def outlier_summary_zscore(
    df: pd.DataFrame,
    *,
    threshold: float = 3.0,
    columns: list[Any] | None = None,
) -> pd.DataFrame:
    """Return outlier counts/% per numeric column using Z-score thresholding."""
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()
    rows: list[dict[str, Any]] = []
    n = max(int(len(df)), 1)
    thr = float(threshold)
    for col in columns:
        if col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        series = df[col].astype(float)
        mu = float(series.mean(skipna=True))
        sigma = float(series.std(skipna=True))
        if not np.isfinite(sigma) or sigma == 0:
            continue
        z = (series - mu) / sigma
        count = int((z.abs() > thr).sum(skipna=True))
        if count <= 0:
            continue
        rows.append(
            {
                'column': str(col),
                'outlier_count': count,
                'outlier_pct': round((count / n) * 100.0, 3),
                'method': 'zscore',
            }
        )
    if not rows:
        return pd.DataFrame(columns=['column', 'outlier_count', 'outlier_pct', 'method'])
    return pd.DataFrame(rows).sort_values(['outlier_pct', 'outlier_count', 'column'], ascending=[False, False, True])


def plot_outlier_counts(outlier_df: pd.DataFrame, *, title: str) :
    """Plot a simple bar chart of outlier % by feature."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if outlier_df is None or outlier_df.empty:
        ax.text(0.5, 0.5, 'No outliers detected with this method.', ha='center', va='center')
        ax.axis('off')
        return fig

    plot_df = outlier_df.copy()
    if 'column' not in plot_df.columns or 'outlier_pct' not in plot_df.columns:
        ax.text(0.5, 0.5, 'Outlier summary is unavailable.', ha='center', va='center')
        ax.axis('off')
        return fig

    plot_df = plot_df.head(25)
    sns.barplot(x='column', y='outlier_pct', data=plot_df, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Outliers (%)')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig


def train_test_split_summary(
    df: pd.DataFrame,
    *,
    target_col: Any,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Return a summary of the stratified train/test split.

    Uses the same split logic as the training step so the EDA "split summary" is consistent.
    """
    from project.modules.Module4 import split_train_test_stratified

    X_train, X_test, y_train, y_test = split_train_test_stratified(
        df,
        target_col,
        test_size=float(test_size),
        random_state=int(random_state),
    )
    return {
        'n_rows': int(len(df)),
        'n_features': int(X_train.shape[1]),
        'test_size': float(test_size),
        'train_rows': int(X_train.shape[0]),
        'test_rows': int(X_test.shape[0]),
        'train_class_counts': y_train.value_counts(dropna=False).to_dict(),
        'test_class_counts': y_test.value_counts(dropna=False).to_dict(),
    }