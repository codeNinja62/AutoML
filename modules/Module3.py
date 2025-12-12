# Module 3: Issue Detection and User Approval
# Detect missing values
# Detect outliers
# Detect class imbalance
# Detect high-cardinality categorical features
# Detect constant/near-constant features
# Show warnings and suggest fixes
# Ask user confirmation before applying fixes



# Imports
import pandas as pd
import numpy as np

# 1. Detect missing values
def detect_missing_values(df):
    missing_info = df.isnull().sum()
    missing_info = missing_info[missing_info > 0]
    return missing_info

# 2. Detect outliers using Z-score
def detect_outliers_zscore(df, threshold=3):
    outlier_indices = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        outliers = df[np.abs(z_scores) > threshold]
        if not outliers.empty:
            outlier_indices[column] = outliers.index.tolist()
    return outlier_indices

# 3. Detect class imbalance
def detect_class_imbalance(df, target_column, threshold=0.7):
    class_counts = df[target_column].value_counts(normalize=True)
    imbalanced_classes = class_counts[class_counts > threshold]
    return imbalanced_classes

# 4. Detect high-cardinality categorical features
def detect_high_cardinality(df, threshold=0.1):
    high_cardinality_features = []
    for column in df.select_dtypes(include=['object', 'category']).columns:
        unique_ratio = df[column].nunique() / len(df)
        if unique_ratio > threshold:
            high_cardinality_features.append(column)
    return high_cardinality_features

# 5. Detect constant/near-constant features
def detect_constant_features(df, threshold=0.95):
    constant_features = []
    for column in df.columns:
        top_freq = df[column].value_counts(normalize=True).iloc[0]
        if top_freq > threshold:
            constant_features.append(column)
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

