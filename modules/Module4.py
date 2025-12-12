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
    X = df.drop(columns=[target_column])
    y = df[target_column]
    stratify = y if y.nunique() > 1 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

