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

def handle_missing_values(df: pd.DataFrame, strategy_dict: dict) -> pd.DataFrame:
    """Return a copy of df with missing values filled per-column.

    strategy_dict maps column -> strategy where strategy is one of:
    - 'mean', 'median' (numeric columns)
    - 'mode' (any dtype)
    - any other value is treated as a constant fill_value
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')

    out = df.copy()
    for column, strategy in (strategy_dict or {}).items():
        if column not in out.columns:
            continue

        s = out[column]
        strat = str(strategy)
        if strat in {'mean', 'median'}:
            if not pd.api.types.is_numeric_dtype(s):
                # Fall back to mode for non-numeric columns.
                strat = 'mode'

        if strat == 'mean':
            fill_value = float(s.dropna().mean()) if not s.dropna().empty else 0.0
        elif strat == 'median':
            fill_value = float(s.dropna().median()) if not s.dropna().empty else 0.0
        elif strat == 'mode':
            modes = s.mode(dropna=True)
            if not modes.empty:
                fill_value = modes.iloc[0]
            else:
                fill_value = 0.0 if pd.api.types.is_numeric_dtype(s) else ''
        else:
            # Treat unknown strategies as constant fill values.
            fill_value = strategy

        out[column] = s.fillna(fill_value)

    return out

def handle_outliers_capping(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Cap a single numeric column using IQR bounds (returns a copy)."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')
    if column not in df.columns:
        raise ValueError(f"Column {column!r} not found")

    out = df.copy()
    if not pd.api.types.is_numeric_dtype(out[column]):
        return out

    bounds = _iqr_bounds(out[column], multiplier=1.5)
    if bounds is None:
        return out
    lb, ub = bounds
    out[column] = out[column].clip(lower=lb, upper=ub)
    return out

def scale_numerical_features(
    df: pd.DataFrame,
    numerical_columns: list[str],
    scaling: str = 'standard',
) -> pd.DataFrame:
    """Scale numeric columns (returns a copy).

    scaling: 'standard' | 'minmax' | 'none'
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')
    out = df.copy()
    cols = [c for c in (numerical_columns or []) if c in out.columns]
    if not cols or scaling == 'none':
        return out

    scaler = MinMaxScaler() if scaling == 'minmax' else StandardScaler()
    out[cols] = scaler.fit_transform(out[cols])
    return out

def encode_categorical_variables(
    df: pd.DataFrame,
    categorical_columns: list[str],
    encoding_type: str = 'onehot',
) -> pd.DataFrame:
    """Encode categorical columns (returns a copy).

    encoding_type: 'onehot' | 'ordinal' | 'none'
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')
    out = df.copy()
    cols = [c for c in (categorical_columns or []) if c in out.columns]
    if not cols or encoding_type == 'none':
        return out

    if encoding_type == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        out[cols] = encoder.fit_transform(out[cols])
        return out

    # default: onehot
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded = encoder.fit_transform(out[cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cols), index=out.index)
    out = out.drop(columns=cols)
    out = pd.concat([out, encoded_df], axis=1)
    return out

def split_train_test(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
):
    """Train-test split with optional stratification."""

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

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
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

