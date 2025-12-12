# Module 4: Preprocessing
# Missing value handling (mean/median/mode/constant)
# Outlier handling if needed
# Scaling
# Encoding categorical variables (One-Hot/Ordinal)
# Train-test split based on user ratio


"""Module 4: Preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler


def _numeric_feature_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude_set = set(exclude or [])
    cols = [c for c in df.select_dtypes(include='number').columns.tolist() if c not in exclude_set]
    return cols


def _iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float] | None:
    s = series.dropna()
    if s.empty:
        return None
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    lb = q1 - multiplier * iqr
    ub = q3 + multiplier * iqr
    return lb, ub


def _row_outlier_mask_iqr(
    df: pd.DataFrame,
    numeric_columns: list[str],
    multiplier: float = 1.5,
) -> pd.Series:
    if df.empty or not numeric_columns:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for col in numeric_columns:
        bounds = _iqr_bounds(df[col], multiplier=multiplier)
        if bounds is None:
            continue
        lb, ub = bounds
        mask = mask | (df[col] < lb) | (df[col] > ub)
    return mask


def _row_outlier_mask_zscore(
    df: pd.DataFrame,
    numeric_columns: list[str],
    threshold: float = 3.0,
) -> pd.Series:
    if df.empty or not numeric_columns:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for col in numeric_columns:
        s = df[col]
        s_non_null = s.dropna()
        if s_non_null.empty:
            continue
        mean = float(s_non_null.mean())
        std = float(s_non_null.std(ddof=0))
        if std == 0.0:
            continue
        z = (s - mean).abs() / std
        mask = mask | (z > threshold)
    return mask


def apply_outlier_handling(
    df: pd.DataFrame,
    action: str = 'no_action',
    method: str = 'iqr',
    *,
    exclude_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> tuple[pd.DataFrame, dict]:
    """Apply simple dataset-level outlier handling.

    Supported actions:
    - no_action: return df unchanged
    - cap_iqr: cap each numeric column to its IQR bounds
    - remove_rows: drop any row that is an outlier by the selected method

    Returns a (new_df, summary) tuple.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')

    action = str(action)
    method = str(method)

    if numeric_columns is None:
        numeric_columns = _numeric_feature_columns(df, exclude=exclude_columns)
    else:
        numeric_columns = [c for c in numeric_columns if c in df.columns]

    before_rows = int(df.shape[0])
    summary = {
        'action': action,
        'method': method,
        'numeric_columns': list(numeric_columns),
        'rows_before': before_rows,
        'rows_after': before_rows,
        'rows_removed': 0,
        'values_capped': 0,
    }

    if action == 'no_action' or before_rows == 0 or not numeric_columns:
        return df.copy(), summary

    if action == 'cap_iqr':
        capped = df.copy()
        values_capped = 0
        for col in numeric_columns:
            bounds = _iqr_bounds(capped[col], multiplier=iqr_multiplier)
            if bounds is None:
                continue
            lb, ub = bounds
            s = capped[col]
            # Count how many non-null values would be capped.
            to_cap = s.notna() & ((s < lb) | (s > ub))
            values_capped += int(to_cap.sum())
            capped[col] = s.clip(lower=lb, upper=ub)
        summary['values_capped'] = int(values_capped)
        return capped, summary

    if action == 'remove_rows':
        if method == 'zscore':
            mask = _row_outlier_mask_zscore(df, numeric_columns=numeric_columns, threshold=zscore_threshold)
        else:
            mask = _row_outlier_mask_iqr(df, numeric_columns=numeric_columns, multiplier=iqr_multiplier)
        kept = df.loc[~mask].copy()
        summary['rows_after'] = int(kept.shape[0])
        summary['rows_removed'] = int(before_rows - kept.shape[0])
        return kept, summary

    raise ValueError(f"Unknown outlier action: {action!r}")

# 1. Missing value handling
def handle_missing_values(df, strategy_dict):
    for column, strategy in strategy_dict.items():
        if strategy == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
        elif strategy == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif strategy == 'mode':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(strategy, inplace=True)
    return df

# 2. Outlier handling by capping
def handle_outliers_capping(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# 3. Scaling numerical features
def scale_numerical_features(df, numerical_columns):
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

# 4. Encoding categorical variables
def encode_categorical_variables(df, categorical_columns, encoding_type='onehot'):
    if encoding_type == 'onehot':
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = encoder.fit_transform(df[categorical_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
        df = df.drop(columns=categorical_columns).reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)
    elif encoding_type == 'ordinal':
        encoder = OrdinalEncoder()
        df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    return df

# 5. Train-test split
def split_train_test(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    X: pd.DataFrame,
    numeric_impute: str = 'median',
    categorical_impute: str = 'most_frequent',
    numeric_fill_value: float | int | None = None,
    categorical_fill_value: str | None = None,
    scaling: str = 'standard',
    encoding: str = 'onehot',
):
    """Create a ColumnTransformer preprocessor for mixed-type tabular data."""
    numeric_features = X.select_dtypes(include='number').columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    if scaling == 'minmax':
        scaler = MinMaxScaler()
    elif scaling == 'standard':
        scaler = StandardScaler()
    else:
        scaler = 'passthrough'

    if encoding == 'auto':
        # Simple heuristic: if total one-hot features would explode, prefer ordinal.
        # (Keeps the system fast for high-cardinality categoricals.)
        estimated_onehot_dims = 0
        for c in categorical_features:
            estimated_onehot_dims += int(X[c].nunique(dropna=True))
        encoding = 'ordinal' if estimated_onehot_dims > 200 else 'onehot'

    if encoding == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    else:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    numeric_imputer_kwargs = {}
    if numeric_impute == 'constant':
        numeric_imputer_kwargs['fill_value'] = 0 if numeric_fill_value is None else numeric_fill_value

    categorical_imputer_kwargs = {}
    if categorical_impute == 'constant':
        categorical_imputer_kwargs['fill_value'] = '' if categorical_fill_value is None else categorical_fill_value

    numeric_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_impute, **numeric_imputer_kwargs)),
        ('scaler', scaler),
    ])

    categorical_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_impute, **categorical_imputer_kwargs)),
        ('encoder', encoder),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, numeric_features),
            ('cat', categorical_pipe, categorical_features),
        ],
        remainder='drop',
    )
    return preprocessor


def split_train_test_stratified(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')
    if df.empty:
        raise ValueError('Dataset is empty')
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} not found")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    if X.shape[1] == 0:
        raise ValueError('No feature columns remain after removing the target column')

    y_non_null = y.dropna()
    n_classes = int(y_non_null.nunique())
    if n_classes < 2:
        raise ValueError('Target must have at least 2 classes for classification')

    class_counts = y_non_null.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    if min_class_count < 2:
        raise ValueError('Each class must have at least 2 samples for a stratified split')

    n_samples = int(len(y_non_null))
    if isinstance(test_size, float):
        if not (0.0 < test_size < 1.0):
            raise ValueError('test_size must be between 0 and 1 when provided as a float')
        test_n = int(np.ceil(test_size * n_samples))
    else:
        test_n = int(test_size)
    train_n = n_samples - test_n
    if test_n < n_classes or train_n < n_classes:
        raise ValueError(
            'Test/train split is too small for stratification: each split must contain at least one sample per class. '
            'Increase dataset size or adjust the test ratio.'
        )

    stratify = y
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

