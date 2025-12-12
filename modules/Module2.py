# Module 2: Automated Exploratory Data Analysis (EDA)
# Missing value analysis
# Outlier detection (IQR / Z-score)
# Correlation matrix
# Numerical feature distributions
# Categorical feature bar plots
# Train/test split summary


"""Module 2: Automated Exploratory Data Analysis (EDA)."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    sns.histplot(df[column].dropna(), kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    fig.tight_layout()
    return fig


def plot_categorical_distribution(df: pd.DataFrame, column: str, top_n: int = 20):
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = df[column].astype(str).fillna('NaN').value_counts().head(top_n)
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title(f'Count Plot of {column} (top {top_n})')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig


def plot_outlier_boxplot(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f'Boxplot: {column}')
    fig.tight_layout()
    return fig